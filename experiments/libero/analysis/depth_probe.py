"""Object-position decodability along depth: SDT-V3 residual stream vs DeiT-S hidden states (both frozen, ImageNet
supervised, ~20M). Same probe and scenes as object_probe.py (within-task object xy displacement, kernel ridge on
14x14-pooled maps of both views, 5-fold CV over the 50 initial states).

SDT-V3 taps: stage outputs before each downsampling, then the stream after every block of stage 3 (14x14,
6 spike-driven attention blocks) and stage 4 (2 blocks). DeiT-S taps: patch embedding and all 12 layers.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import object_probe as op  # noqa: E402


def load_scenes(root, suites):
    scenes = {}
    for suite in suites:
        parts = defaultdict(list)
        for f in sorted((Path(root) / suite).glob("task*_s*.npz")):
            parts[int(f.stem[4:6])].append(dict(np.load(f)))
        for task, chunks in sorted(parts.items()):
            chunks.sort(key=lambda z: int(z["init_index"][0]))
            s = {k: np.concatenate([z[k] for z in chunks]) for k in ("agent", "wrist", "object_pos")}
            s["meta"] = json.loads(str(chunks[0]["meta"]))
            scenes[(suite, task)] = s
    return scenes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True)
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--deit", required=True)
    p.add_argument("--dinov3_b", required=True)
    p.add_argument("--bert", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--suites", nargs="*", default=list(op.SUITES))
    args = p.parse_args()
    op.BERT_PATH = args.bert
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenes = load_scenes(args.scenes, args.suites)

    def sdt_taps(enc):
        bb, captured = enc.backbone, {}
        points = {"stage1 out": (bb.downsample2, "pre"), "stage2 out": (bb.downsample3, "pre"),
                  **{f"block3.{i}": (b, "post") for i, b in enumerate(bb.block3)},
                  **{f"block4.{i}": (b, "post") for i, b in enumerate(bb.block4)}}
        hooks = []
        for name, (module, when) in points.items():
            if when == "pre":
                hooks.append(module.register_forward_pre_hook(
                    lambda m, inp, name=name: captured.__setitem__(name, inp[0].detach())))
            else:
                hooks.append(module.register_forward_hook(
                    lambda m, inp, out, name=name: captured.__setitem__(name, out.detach())))

        def run(images):
            x = enc.processor(list(images))["pixel_values"].float().to(device)
            captured.clear()
            with torch.inference_mode():
                bb.forward_features(x)
            return {k: op.grid(v).cpu() for k, v in captured.items()}
        return run, list(points)

    def deit_taps(enc):
        def run(images):
            x = enc.processor(list(images))["pixel_values"].float().to(device)
            with torch.inference_mode():
                hidden = enc.model(pixel_values=x, output_hidden_states=True).hidden_states
            return {("embed" if i == 0 else f"layer{i}"): op.grid(op.tokens_to_map(h, h.shape[1] - 196)).cpu()
                    for i, h in enumerate(hidden)}
        return run, ["embed"] + [f"layer{i}" for i in range(1, 13)]

    models = {
        "SDT-V3 ImageNet (frozen)": lambda: sdt_taps(op.FrozenSDT(args.base_ckpt, args.dinov3_b, device)),
        "DeiT-S ImageNet (frozen)": lambda: deit_taps(op.FrozenHF(args.deit, args.dinov3_b, device)),
    }
    results = {}
    for name, make in models.items():
        run, order = make()
        per = defaultdict(lambda: defaultdict(list))
        for (suite, task), s in scenes.items():
            feats = defaultdict(list)
            for view in ("agent", "wrist"):
                parts = [run(s[view][i:i + 25]) for i in range(0, 50, 25)]
                for k in parts[0]:
                    feats[k].append(torch.cat([q[k] for q in parts]))
            ys = []
            for k, obj in enumerate(s["meta"]["objects"]):
                y = s["object_pos"][:, k, :2].astype(np.float64)
                y = y - y.mean(0)
                if y.std(0).max() >= 1e-4:
                    ys.append(y)
            for key, views in feats.items():
                gram = 0
                for x in views:
                    x = x.double()
                    x = x / x.norm(dim=1).mean().clamp_min(1e-12)
                    gram = gram + (x @ x.T).numpy()
                for y in ys:
                    pred = op.cv_predict(gram, y, 5)
                    per[key][suite].append(1 - ((pred - y) ** 2).sum() / (y ** 2).sum())
        results[name] = {k: {suite: float(np.mean(v)) for suite, v in per[k].items()} for k in order}
        print("\n==", name)
        for k in order:
            print(f"  {k:12s} {np.mean(list(results[name][k].values())):.3f}  " +
                  " ".join(f"{suite[7:]}={v:.3f}" for suite, v in results[name][k].items()), flush=True)
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
