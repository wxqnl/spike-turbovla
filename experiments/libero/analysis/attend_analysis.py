"""Analysis of attend_probe.py records: how much of the head's cross-attention falls on the plate, the target bowl
and the gripper, per model, frame (t0 / transport / approach), camera and decoder layer.

For every probed frame the episode's simulator state is restored in the same LIBERO scene, the three points (plate
and bowl body centers, end effector) are projected into both cameras with robosuite's camera matrices, and mapped to
the policy's input image (rotated by 180 degrees: row r, column 255 - c at 256 px) and then to the model's patch grid
(14 x 14 for SDT-V3 at 224 px, 16 x 16 for DINOv3 at 256 px; both cover the full image). Only the unshifted render
(the first of the probe renders) is used.

Reported per camera and layer, from the attention weights over the visual tokens:
  share      that camera's part of all cross-attention weight (visual + text + state = 1)
  lift(X)    weight on the patches whose centers lie within RADIUS px (at 256 px) of X, as a fraction of the
             camera's weight, divided by the fraction of that camera's patches inside the disc (1 = no preference)
Points outside the image are skipped for that camera (n counts the frames used).

Heads specialize, so the same lift is also computed for every (layer, head) from attn_hq (averaged over queries):
  best head  the head with the highest mean plate lift on one half of the episodes (even / odd init states), scored
             on the other half and averaged over the two splits, so picking the maximum does not inflate it.
Usage: python attend_analysis.py <root> [--radius 32]
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = 256


def projector(env):
    from robosuite.utils import camera_utils as cu
    inner = env.env

    def project(cam, points):
        T = cu.get_camera_transform_matrix(inner.sim, cam, RES, RES)
        rc = cu.project_points_from_world_to_camera(np.asarray(points), T, RES, RES).astype(np.float64)
        return np.stack([rc[:, 0], RES - 1 - rc[:, 1]], -1)  # policy input is the render rotated by 180 degrees
    return project


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--radius", type=float, default=32.0)
    args = p.parse_args()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl_root = Path(get_libero_path("bddl_files"))
    cams = ("agentview", "robot0_eye_in_hand")
    stats = defaultdict(list)  # (model, frame, cam, layer, metric) -> values
    heads = defaultdict(list)  # (model, frame, cam, target) -> [(init_index, lift [layers, heads])]
    for model_dir in sorted(Path(args.root).iterdir()):
        if not model_dir.is_dir():
            continue
        for path in sorted(model_dir.glob("task*.pkl")):
            rec = pickle.load(open(path, "rb"))
            task = rec["task"]
            env = OffScreenRenderEnv(bddl_file_name=str(next(bddl_root.rglob(task["bddl"]))),
                                     camera_heights=RES, camera_widths=RES)
            env.seed(7)
            env.reset()
            inner, project = env.env, projector(env)
            bowl_name, plate_name = task["goal"][0][1], task["goal"][0][2]
            for ep in rec["episodes"]:
                for frame, probe in ep["probes"].items():
                    if "attn_map" not in probe["features"]:
                        continue
                    attn = probe["features"]["attn_map"][0].astype(np.float64)  # unshifted render: [layers, nvis]
                    side = int(round((attn.shape[1] // 2) ** 0.5))
                    centers = (np.arange(side) + 0.5) * RES / side
                    grid = np.stack(np.meshgrid(centers, centers, indexing="ij"), -1).reshape(-1, 2)  # (row, col)
                    obs = env.set_init_state(ep["sim_states"][ep["frames"][frame]])
                    points = np.stack([inner.sim.data.body_xpos[inner.obj_body_id[plate_name]],
                                       inner.sim.data.body_xpos[inner.obj_body_id[bowl_name]],
                                       np.asarray(obs["robot0_eef_pos"])])
                    for c, cam in enumerate(cams):
                        px = project(cam, points)
                        w = attn[:, c * side * side:(c + 1) * side * side]  # [layers, side^2]
                        share = w.sum(-1)
                        for layer in range(w.shape[0]):
                            stats[(model_dir.name, frame, cam, layer, "share")].append(share[layer])
                        for name, xy in zip(("plate", "bowl", "gripper"), px):
                            if not (0 <= xy[0] < RES and 0 <= xy[1] < RES):
                                continue
                            inside = np.linalg.norm(grid - xy, axis=-1) <= args.radius
                            if not inside.any():
                                continue
                            frac = inside.mean()
                            for layer in range(w.shape[0]):
                                mass = w[layer, inside].sum() / max(w[layer].sum(), 1e-12)
                                stats[(model_dir.name, frame, cam, layer, f"lift_{name}")].append(mass / frac)
                            if "attn_hq" in probe["features"]:
                                hq = probe["features"]["attn_hq"].astype(np.float64).mean(2)  # [layers, heads, nvis]
                                wh = hq[..., c * side * side:(c + 1) * side * side]
                                lift = wh[..., inside].sum(-1) / np.maximum(wh.sum(-1), 1e-12) / frac
                                heads[(model_dir.name, frame, cam, name)].append((ep["init_index"], lift))
            env.close()
    report = defaultdict(dict)
    models = sorted({k[0] for k in stats})
    for model in models:
        print(f"\n== {model}")
        for frame in ("t0", "transport", "approach"):
            for cam in cams:
                rows = []
                layers = sorted({k[3] for k in stats if k[:3] == (model, frame, cam)})
                if not layers:
                    continue
                for layer in layers:
                    cell = {m: (float(np.mean(stats[(model, frame, cam, layer, m)])),
                                len(stats[(model, frame, cam, layer, m)]))
                            for m in ("share", "lift_plate", "lift_bowl", "lift_gripper")
                            if stats.get((model, frame, cam, layer, m))}
                    report[model][f"{frame}/{cam}/L{layer}"] = cell
                    rows.append(f"L{layer} " + " ".join(f"{m.replace('lift_', '')} {v:.2f}" for m, (v, _) in cell.items()))
                n = {m: v[1] for m, v in cell.items()}
                print(f"  {frame:9s} {cam:18s} n={n.get('lift_plate', 0):3d}  " + " | ".join(rows))
                best = {}
                for target in ("plate", "bowl", "gripper"):
                    entries = heads.get((model, frame, cam, target))
                    if not entries or len(entries) < 4:
                        continue
                    idx = np.array([e[0] for e in entries])
                    lifts = np.stack([e[1] for e in entries])  # [episodes, layers, heads]
                    scores, picks = [], []
                    for half in (0, 1):
                        sel, held = lifts[idx % 2 == half], lifts[idx % 2 != half]
                        if not len(sel) or not len(held):
                            continue
                        layer, head = np.unravel_index(np.argmax(sel.mean(0)), sel.shape[1:])
                        scores.append(held[:, layer, head].mean())
                        picks.append(f"L{layer}h{head}")
                    if scores:
                        best[target] = (float(np.mean(scores)), picks)
                if best:
                    report[model][f"{frame}/{cam}/best_head"] = best
                    print(f"  {'':9s} {'':18s}        best head (held-out): " + "  ".join(
                        f"{t} {v:.2f} ({'/'.join(p)})" for t, (v, p) in best.items()))
    Path(args.root, "attend_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
