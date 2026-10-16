"""Where does SDT-V3 lose fine object position: at its spike quantizers? (zero training, simulator ground truth)

SDT-V3 keeps a continuous residual stream, but every downsampling layer first passes the whole stream through
MultiSpike: round(clamp(x, 0, 4)) / 4, i.e. negatives dropped and 5 levels. At each of these bottlenecks (and at
the backbone output) we take the quantizer's input x and probe several transforms of the same x:
    pre      x itself (continuous)
    relu     clamp(x, 0, inf)            -> cost of dropping the negative half
    clip     clamp(x, 0, 4)              -> plus upper saturation
    spike    round(clamp(x, 0, 4))       -> the actual quantizer (5 levels)
    signed   round(clamp(x, -4, 4))      -> 9 signed levels
    fine     round(4 * clamp(x, 0, 4))   -> 17 unsigned levels
Probe as in object_probe.py: per task and object, kernel ridge on 14x14-pooled maps of both views predicts the
object's xy displacement from its task mean; 5-fold CV over the 50 initial states. Reported: mean R^2 per suite.
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

VARIANTS = {
    "pre": lambda x: x,
    "relu": lambda x: x.clamp(min=0),
    "clip": lambda x: x.clamp(0, 4),
    "spike": lambda x: x.clamp(0, 4).round(),
    "signed": lambda x: x.clamp(-4, 4).round(),
    "fine": lambda x: (4 * x.clamp(0, 4)).round(),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True)
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--dinov3_b", required=True)
    p.add_argument("--bert", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--suites", nargs="*", default=list(op.SUITES))
    args = p.parse_args()
    op.BERT_PATH = args.bert
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class A:  # minimal namespace for op's scene loader
        allow_partial = False
    scenes = {}
    for suite in args.suites:
        parts = defaultdict(list)
        for f in sorted((Path(args.scenes) / suite).glob("task*_s*.npz")):
            parts[int(f.stem[4:6])].append(dict(np.load(f)))
        for task, chunks in sorted(parts.items()):
            chunks.sort(key=lambda z: int(z["init_index"][0]))
            s = {k: np.concatenate([z[k] for z in chunks]) for k in ("agent", "wrist", "object_pos")}
            s["meta"] = json.loads(str(chunks[0]["meta"]))
            assert len(s["agent"]) == 50
            scenes[(suite, task)] = s

    encoders = {
        "SDT-V3 ImageNet (frozen)": lambda: op.FrozenSDT(args.base_ckpt, args.dinov3_b, device),
        "SDT-V3 after Spike-Driven VLA": lambda: op.Policy(args.base_ckpt, args.dinov3_b, device),
    }
    results = {}
    for name, make in encoders.items():
        enc = make()
        bb = enc.backbone
        taps = {"down1_2": bb.downsample1_2, "down2": bb.downsample2, "down3": bb.downsample3,
                "down4": bb.downsample4}
        captured = {}
        hooks = [m.encode_spike.register_forward_hook(
            lambda mod, inp, out, key=key: captured.__setitem__(key, inp[0].detach()))
            for key, m in taps.items()]
        per = defaultdict(lambda: defaultdict(list))  # (layer, variant) -> suite -> [r2]
        for (suite, task), s in scenes.items():
            feats = defaultdict(list)  # (layer, variant) -> [per view tensor]
            for view in ("agent", "wrist"):
                chunks = defaultdict(list)
                for i in range(0, 50, 25):
                    x = enc.processor(list(s[view][i:i + 25]))["pixel_values"].float().to(device)
                    with torch.inference_mode():
                        final = bb.forward_features(x)
                    chunks[("output", "pre")].append(op.grid(final).cpu())
                    for key, value in captured.items():
                        for vname, fn in VARIANTS.items():
                            chunks[(key, vname)].append(op.grid(fn(value)).cpu())
                for k, v in chunks.items():
                    feats[k].append(torch.cat(v))
            meta = s["meta"]
            ys = []
            for k, obj in enumerate(meta["objects"]):
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
                    pred = op.cv_predict(gram, y, args.folds)
                    per[key][suite].append(1 - ((pred - y) ** 2).sum() / (y ** 2).sum())
            captured.clear()
        for h in hooks:
            h.remove()
        results[name] = {f"{k[0]}/{k[1]}": {suite: float(np.mean(v)) for suite, v in d.items()} for k, d in per.items()}
        print("\n==", name)
        layers = ["down1_2", "down2", "down3", "down4"]
        print(f"{'layer':8s} " + " ".join(f"{v:>8s}" for v in VARIANTS) + "   (mean R^2 over suites)")
        for layer in layers:
            row = [np.mean(list(results[name][f"{layer}/{v}"].values())) for v in VARIANTS]
            print(f"{layer:8s} " + " ".join(f"{r:8.3f}" for r in row))
        print(f"{'output':8s} {np.mean(list(results[name]['output/pre'].values())):8.3f}")
        del enc
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
