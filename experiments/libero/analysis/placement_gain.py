"""Does the policy place the object where the target actually is? (LIBERO-Spatial, failure_rollout.py records)

Each LIBERO-Spatial task is "pick up the black bowl ... and place it on the plate"; across the 50 initial states the
plate and bowl move by ~1 cm. Within each task we regress the bowl's final xy on the initial plate xy and the initial
bowl xy (both centered):
    bowl_final = c + G_plate * plate0 + G_bowl * bowl0
A policy that places on the plate it sees has G_plate = 1 and G_bowl = 0. Also reported: the mean offset from the
plate (bias), its spread, and the offset at success (LIBERO's On predicate needs xy distance < 3 cm).
Usage: python placement_gain.py <root>   (expects <model>/libero_spatial/taskNN_eSS.pkl)
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def task_stats(rec):
    t = rec["task"]
    objs = t["objects"]
    b, p = objs.index(t["goal"][0][1]), objs.index(t["goal"][0][2])
    P0, B0, BF, succ, init = [], [], [], [], []
    for ep in rec["episodes"]:
        w = ep["wait_steps"] + 1
        P0.append(ep["object_pos"][w - 1, p, :2])
        B0.append(ep["object_pos"][w - 1, b, :2])
        BF.append(ep["object_pos"][-1, b, :2])
        succ.append(ep["success"])
        init.append(ep["init_index"])
    P0, B0, BF, succ = map(np.array, (P0, B0, BF, succ))
    off = BF - P0
    X = np.c_[np.ones(len(P0)), P0 - P0.mean(0), B0 - B0.mean(0)]
    coef, *_ = np.linalg.lstsq(X, BF, rcond=None)
    dist = np.linalg.norm(off, axis=1) * 100
    return {
        "n": len(succ), "success": int(succ.sum()),
        "gain_plate": np.diag(coef[1:3]).tolist(), "gain_bowl": np.diag(coef[3:5]).tolist(),
        "bias_cm": (off.mean(0) * 100).tolist(), "offset_sd_cm": (off.std(0) * 100).tolist(),
        "success_dist_median_cm": float(np.median(dist[succ])) if succ.any() else None,
        "success_dist_gt2.5": float(np.mean(dist[succ] > 2.5)) if succ.any() else None,
        "dist_cm": dict(zip(map(int, init), dist.round(2).tolist())),
    }


def main():
    root = Path(sys.argv[1])
    out = defaultdict(dict)
    for f in sorted(root.glob("*/libero_spatial/task*_e*.pkl")):
        out[f.parts[-3]][int(f.stem[4:6])] = task_stats(pickle.loads(f.read_bytes()))
    print(f"{'model':9s} {'task':4s} {'succ':>4s} {'G_plate x,y':>12s} {'G_bowl x,y':>12s} {'bias cm':>12s} "
          f"{'sd cm':>11s} {'med|succ':>8s} {'>2.5|succ':>9s}")
    for m, tasks in out.items():
        for t, s in sorted(tasks.items()):
            print(f"{m:9s} t{t:02d}  {s['success']:4d} {s['gain_plate'][0]:6.2f}{s['gain_plate'][1]:6.2f} "
                  f"{s['gain_bowl'][0]:6.2f}{s['gain_bowl'][1]:6.2f} {s['bias_cm'][0]:6.2f}{s['bias_cm'][1]:6.2f} "
                  f"{s['offset_sd_cm'][0]:5.2f}{s['offset_sd_cm'][1]:6.2f} {s['success_dist_median_cm']:8.2f} "
                  f"{s['success_dist_gt2.5']:9.2f}")
        g = np.array([s["gain_plate"] for s in tasks.values()])
        gb = np.array([s["gain_bowl"] for s in tasks.values()])
        bias = np.array([np.linalg.norm(s["bias_cm"]) for s in tasks.values()])
        print(f"{m:9s} mean: success {sum(s['success'] for s in tasks.values())}/{sum(s['n'] for s in tasks.values())}"
              f"  G_plate median {np.median(g):.2f}  G_bowl median {np.median(gb):.2f}  |bias| median {np.median(bias):.2f} cm"
              f"  success-dist median {np.median([s['success_dist_median_cm'] for s in tasks.values()]):.2f} cm\n")
    # Paired: same task and initial state.
    models = list(out)
    if len(models) == 2:
        a, b = models
        diff = []
        for t in out[a]:
            if t in out[b]:
                da, db = out[a][t]["dist_cm"], out[b][t]["dist_cm"]
                diff += [da[i] - db[i] for i in da if i in db]
        diff = np.array(diff)
        print(f"paired final distance to plate, {a} - {b}: mean {diff.mean():.2f} cm, median {np.median(diff):.2f}, "
              f"{a} farther in {np.mean(diff > 0):.0%} of {len(diff)} initial states")
    (root / "placement_gain.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
