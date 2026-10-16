"""How precisely do vision encoders localize objects in LIBERO start frames? (simulator ground truth)

Scenes come from render_scenes.py: for every task, the 50 evaluation initial states after the settling steps,
with exact object positions. Within a task the layout is fixed and objects are perturbed by a few cm, so the
question is whether an encoder's patch features resolve those displacements.

Probe: per task and per movable object, kernel ridge (linear kernel on 14x14-pooled patch features of both
views) predicts the object's xy displacement from its task mean; 5-fold CV over initial states, inner CV for the
ridge strength. Reported per suite: mean R^2 and RMSE (cm) over (task, object), plus the same for the task's
target object (first argument of the first binary goal predicate).

Encoders: SDT-V3 19M ImageNet (frozen) and after Spike-Driven VLA training; DeiT-S/16 (frozen, ImageNet
supervised, accuracy-matched ANN); DINOv3 ViT-S/16 and ViT-B/16 (frozen); TurboVLA's DINOv3-B after VLA
training; 28x28 pixels. Each encoder gets the preprocessing it was trained with.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
BERT_PATH = ""


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True)
    p.add_argument("--base_ckpt", required=True, help="Spike-Driven VLA checkpoint (model_config paths valid here)")
    p.add_argument("--turbovla_ckpt", required=True)
    p.add_argument("--deit", required=True)
    p.add_argument("--dinov3_s", required=True)
    p.add_argument("--dinov3_b", required=True)
    p.add_argument("--bert", required=True, help="bert-base-uncased directory (TurboVLA text encoder)")
    p.add_argument("--extra", nargs="*", default=[], help="name=checkpoint of further trained policies (D runs)")
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only", nargs="*", default=None, help="debug: run only these encoder names")
    p.add_argument("--allow_partial", action="store_true", help="debug: skip tasks not fully rendered")
    return p.parse_args()


def grid(x, size=14):
    """[N, C, H, W] -> [N, size*size*C], average-pooled when larger."""
    if x.shape[-1] != size:
        x = F.adaptive_avg_pool2d(x.float(), size)
    return x.flatten(1)


def tokens_to_map(tokens, prefix):
    tokens = tokens[:, prefix:]
    side = int(round(tokens.shape[1] ** 0.5))
    return tokens.transpose(1, 2).reshape(len(tokens), -1, side, side)


class Policy:
    """Vision encoder of a trained TurboVLA-family checkpoint, fed exactly as at evaluation."""

    def __init__(self, ckpt, dinov3_path, device):
        from turbovla.evaluation.policy import build_dinov3_manual_processor, build_sdtv3_manual_processor
        from turbovla.models.configuration import TurboVLAConfig
        from turbovla.models.turbovla import build_turbovla as build_model
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        config = TurboVLAConfig.from_mapping(raw["model_config"])
        if config.vision.encoder_type == "dinov3" and not Path(config.vision.model_name_or_path or "").is_dir():
            config.vision.model_name_or_path = dinov3_path
        if config.text.encoder_type == "bert":  # as in turbovla.evaluation.policy
            config.text.model_name_or_path = BERT_PATH
        config.vision.compute_precision = "fp32"
        self.kind = config.vision.encoder_type
        encoder = build_model(config).vision_encoder
        state = raw.get("ema_model_state_dict") or raw["model_state_dict"]
        own = {k[len("vision_encoder."):]: v for k, v in state.items() if k.startswith("vision_encoder.")}
        missing, unexpected = encoder.load_state_dict(own, strict=False)
        assert not unexpected and all("position" in k or "view" in k for k in missing), (missing, unexpected)
        self.backbone = encoder.backbone.to(device).eval()
        self.processor = (build_sdtv3_manual_processor() if self.kind == "sdtv3_19m"
                          else build_dinov3_manual_processor(dinov3_path, config.vision.image_size))
        self.device = device

    @torch.inference_mode()
    def __call__(self, images):
        x = self.processor(list(images))["pixel_values"].float().to(self.device)
        if self.kind == "sdtv3_19m":
            # the continuous residual map [N, 360, 14, 14] that SDTV3VisionEncoder hands to the projection
            return grid(self.backbone.forward_features(x)).cpu()
        hidden = self.backbone(pixel_values=x).last_hidden_state
        prefix = hidden.shape[1] - (x.shape[-1] // 16) ** 2
        return grid(tokens_to_map(hidden, prefix)).cpu()


class FrozenHF:
    def __init__(self, path, processor_path, device):
        from transformers import AutoModel
        from turbovla.evaluation.policy import build_dinov3_manual_processor
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to(device).eval()
        self.processor = build_dinov3_manual_processor(processor_path, 224)
        self.device = device

    @torch.inference_mode()
    def __call__(self, images):
        x = self.processor(list(images))["pixel_values"].float().to(self.device)
        hidden = self.model(pixel_values=x).last_hidden_state
        return grid(tokens_to_map(hidden, hidden.shape[1] - 196)).cpu()


class FrozenSDT(Policy):
    """SDT-V3 as initialised from its ImageNet checkpoint (no VLA training)."""

    def __init__(self, ckpt, dinov3_path, device):
        from turbovla.evaluation.policy import build_sdtv3_manual_processor
        from turbovla.models.configuration import TurboVLAConfig
        from turbovla.models.turbovla import build_turbovla as build_model
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        config = TurboVLAConfig.from_mapping(raw["model_config"])
        self.kind = "sdtv3_19m"
        self.backbone = build_model(config).vision_encoder.backbone.to(device).eval()
        self.processor = build_sdtv3_manual_processor()
        self.device = device


def centered(gram, train, rows, cols):
    mean_row = gram[:, train].mean(1)
    mean_all = gram[np.ix_(train, train)].mean()
    return gram[np.ix_(rows, cols)] - mean_row[rows, None] - mean_row[None, cols] + mean_all


def ridge_predict(gram, y, train, test, lam):
    values, vectors = np.linalg.eigh(centered(gram, train, train, train))
    proj = vectors.T @ (y[train] - y[train].mean(0))
    alpha = vectors @ (proj / (values + lam)[:, None])
    return centered(gram, train, test, train) @ alpha + y[train].mean(0)


def cv_predict(gram, y, folds):
    """Outer k-fold over scenes; ridge strength chosen per fold by inner k-fold on the training scenes."""
    n = len(y)
    fold = np.arange(n) % folds
    scale = np.trace(gram) / n
    lams = scale * np.logspace(-4, 3, 8)
    pred = np.zeros_like(y)
    for f in range(folds):
        train, test = np.where(fold != f)[0], np.where(fold == f)[0]
        inner = np.arange(len(train)) % folds
        err = np.zeros(len(lams))
        for g in range(folds):
            tr, te = train[inner != g], train[inner == g]
            for i, lam in enumerate(lams):
                err[i] += ((ridge_predict(gram, y, tr, te, lam) - y[te]) ** 2).sum()
        pred[test] = ridge_predict(gram, y, train, test, lams[err.argmin()])
    return pred


def main():
    args = parse()
    global BERT_PATH
    BERT_PATH = args.bert
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenes = {}
    for suite in SUITES:
        parts = defaultdict(list)  # render_scenes.py writes each task in chunks of initial states
        for f in sorted((Path(args.scenes) / suite).glob("task*_s*.npz")):
            try:
                parts[int(f.stem[4:6])].append(dict(np.load(f)))
            except (EOFError, ValueError, OSError):
                if not args.allow_partial:  # a chunk still being written
                    raise
        for task, chunks in sorted(parts.items()):
            chunks.sort(key=lambda z: int(z["init_index"][0]))
            scenes[(suite, task)] = {k: np.concatenate([z[k] for z in chunks]) for k in ("agent", "wrist", "object_pos")}
            scenes[(suite, task)]["meta"] = json.loads(str(chunks[0]["meta"]))
            if len(scenes[(suite, task)]["agent"]) != 50:
                assert args.allow_partial, (suite, task)
                del scenes[(suite, task)]
    print("scenes", len(scenes), "tasks", flush=True)

    encoders = {
        "pixels 28x28": None,
        "SDT-V3 ImageNet (frozen)": lambda: FrozenSDT(args.base_ckpt, args.dinov3_b, device),
        "SDT-V3 after Spike-Driven VLA": lambda: Policy(args.base_ckpt, args.dinov3_b, device),
        "DeiT-S ImageNet (frozen)": lambda: FrozenHF(args.deit, args.dinov3_b, device),
        "DINOv3-S (frozen)": lambda: FrozenHF(args.dinov3_s, args.dinov3_b, device),
        "DINOv3-B (frozen)": lambda: FrozenHF(args.dinov3_b, args.dinov3_b, device),
        "DINOv3-B after TurboVLA": lambda: Policy(args.turbovla_ckpt, args.dinov3_b, device),
    }
    for item in args.extra:
        name, ckpt = item.split("=", 1)
        encoders[name] = lambda ckpt=ckpt: Policy(ckpt, args.dinov3_b, device)

    if args.only:
        encoders = {k: v for k, v in encoders.items() if k in args.only}
    results = {}
    for name, make in encoders.items():
        encoder = make() if make else None
        per_suite = defaultdict(lambda: defaultdict(list))
        for (suite, task), s in scenes.items():
            feats = []
            for view in ("agent", "wrist"):
                if encoder is None:
                    x = torch.from_numpy(s[view]).permute(0, 3, 1, 2).float() / 255
                    feats.append(grid(x, 28))
                else:
                    feats.append(torch.cat([encoder(s[view][i:i + 25]) for i in range(0, len(s[view]), 25)]))
            grams = []
            for x in feats:
                x = x.double()
                x = x / x.norm(dim=1).mean().clamp_min(1e-12)
                grams.append((x @ x.T).numpy())
            gram = grams[0] + grams[1]
            meta = s["meta"]
            target = next((g[1] for g in meta["goal"] if len(g) == 3 and g[1] in meta["objects"]), None)
            for k, obj in enumerate(meta["objects"]):
                y = s["object_pos"][:, k, :2].astype(np.float64)
                y = y - y.mean(0)
                if y.std(0).max() < 1e-4:  # object not perturbed in this task
                    continue
                pred = cv_predict(gram, y, args.folds)
                r2 = 1 - ((pred - y) ** 2).sum() / (y ** 2).sum()
                rmse = float(np.sqrt(((pred - y) ** 2).sum(1).mean()) * 100)
                spread = float(np.sqrt((y ** 2).sum(1).mean()) * 100)
                per_suite[suite]["r2"].append(r2)
                per_suite[suite]["rmse_cm"].append(rmse)
                per_suite[suite]["spread_cm"].append(spread)
                if obj == target:
                    per_suite[suite]["target_r2"].append(r2)
                    per_suite[suite]["target_rmse_cm"].append(rmse)
        results[name] = {suite: {k: float(np.mean(v)) for k, v in d.items()} | {"n": len(d["r2"])}
                         for suite, d in per_suite.items()}
        print(name, json.dumps(results[name]), flush=True)
        del encoder
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("\n" + f"{'encoder':32s} " + " ".join(f"{s[7:]:>16s}" for s in SUITES) + "   (object xy R^2 / RMSE cm)")
    for name, r in results.items():
        print(f"{name:32s} " + " ".join(
            f"{r[s]['r2']:7.3f}/{r[s]['rmse_cm']:5.2f}cm" if s in r else f"{'-':>16s}" for s in SUITES))


if __name__ == "__main__":
    main()
