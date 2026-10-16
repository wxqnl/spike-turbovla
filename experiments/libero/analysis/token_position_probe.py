"""Do the visual tokens the action head reads carry their own image position in their content?

The head reads the visual tokens through attention, a weighted sum that ignores token order. Neither model adds a patch
position embedding at the VLA level (position_embedding = "view": one embedding per camera). If a token's content does
not say where its patch is, the attention read cannot recover where the target is, however well it routes ("index-only
position"). Per token, a linear ridge probe predicts the patch's (row, column) in its camera grid from the
token's content, at two stages: projected visual tokens (features["visual"]) and fused visual tokens (the head's
memory). Frames from 4 suites x 10 demos x 5 frames; probes trained on half of the demos, tested on the other half.

Arguments: the usual evaluator arguments (policy built by the evaluator, as in reliance_policy.py), plus --data_root,
--out [--episodes_per_suite 10 --frames_per_episode 5].
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliance_policy import build_policy, demo_frames  # noqa: E402


def ridge_r2(X_tr, Y_tr, X_te, Y_te, lam=1e-2):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
    A = X_tr.T @ X_tr + lam * len(X_tr) * np.eye(X_tr.shape[1])
    W = np.linalg.solve(A, X_tr.T @ (Y_tr - Y_tr.mean(0)))
    pred = X_te @ W + Y_tr.mean(0)
    return 1 - ((pred - Y_te) ** 2).sum(0) / ((Y_te - Y_te.mean(0)) ** 2).sum(0)


def main():
    argv = sys.argv[1:]
    opts = {"--data_root": None, "--out": None, "--episodes_per_suite": "10", "--frames_per_episode": "5"}
    for flag in list(opts):
        if flag in argv:
            i = argv.index(flag)
            opts[flag] = argv[i + 1]
            del argv[i:i + 2]
    sys.argv = [sys.argv[0], *argv]
    policy = build_policy()
    model = policy.model
    stats = json.loads(Path(policy.stats_path).read_text())[policy.stats_key]
    fused_store = []
    model.vision_language_interaction.register_forward_hook(
        lambda m, a, out: fused_store.append((out[0] if isinstance(out, (tuple, list)) else out).detach()))
    feats = {"projected": [], "fused": []}
    episode_of = []
    for k, (suite, instruction, frame, _) in enumerate(demo_frames(
            opts["--data_root"], stats, model.config.action.horizon, int(opts["--episodes_per_suite"]),
            int(opts["--frames_per_episode"]), 1)):
        samples, states = policy._build_batch([frame["primary"]], [frame["wrist"]], [frame["state"]])
        samples, states = policy._prepare_model_inputs(samples, states)
        fused_store.clear()
        with torch.inference_mode():
            _, features = model([instruction], samples, states, return_distillation_features=True)
        visual = features["visual"][0].float().cpu().numpy()  # [Nvis, C]
        feats["projected"].append(visual)
        feats["fused"].append(fused_store[-1][0, :len(visual)].float().cpu().numpy())
        episode_of.append(k // int(opts["--frames_per_episode"]))
    nvis = feats["projected"][0].shape[0]
    side = int(round((nvis // 2) ** 0.5))  # two cameras, square grids
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), -1).reshape(-1, 2) / (side - 1)
    target = np.concatenate([grid, grid])  # [Nvis, 2] row, column within each camera
    episode_of = np.array(episode_of)
    train = (episode_of % 2) == 0
    out = {"checkpoint": policy.ckpt_path, "tokens_per_camera": side * side, "frames": len(episode_of), "r2": {}}
    for stage, values in feats.items():
        F = np.stack(values)  # [frames, Nvis, C]
        r = {}
        for cam, sl in (("primary", slice(0, side * side)), ("wrist", slice(side * side, nvis))):
            X_tr = F[train][:, sl].reshape(-1, F.shape[-1]).astype(np.float64)
            X_te = F[~train][:, sl].reshape(-1, F.shape[-1]).astype(np.float64)
            Y_tr = np.tile(target[sl], (train.sum(), 1))
            Y_te = np.tile(target[sl], ((~train).sum(), 1))
            row, col = ridge_r2(X_tr, Y_tr, X_te, Y_te)
            r[cam] = {"row": float(row), "column": float(col)}
        out["r2"][stage] = r
        print(f"[position] {stage:9s} " + "  ".join(f"{c}: row {v['row']:.2f} col {v['column']:.2f}" for c, v in r.items()),
              flush=True)
    Path(opts["--out"]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
