"""Classify LIBERO failures from failure_rollout.py records, and compare models on the same initial states.

Rules (fixed before looking at results):
- picked(k): the gripper grasps movable object k while k is lifted >= 2 cm above its initial height.
- targets: first argument of every binary goal predicate (on / in) that is a movable object.
- Failed episode categories, checked in this order:
    partial        multi-predicate goal, >= 1 predicate true at the end but not all
    articulated    goal has only unary predicates (open / close / turnon ...) and failed
    wrong_object   the first picked object is not a target (e.g. the other bowl in LIBERO-Spatial)
    no_pick_near   nothing picked, but the end effector came within 4 cm of a target (reached, grasp failed)
    no_pick_far    nothing picked and never within 4 cm of a target
    not_placed     a target was picked but its predicate never held
    undone         a predicate held at some step but not at the end
    other
Usage: python failure_taxonomy.py <root>/<tag>   (expects <model>/<suite>/taskNN_eSS.pkl from failure_runs.sh)
"""
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LIFT, NEAR = 0.02, 0.04
CATEGORIES = ["partial", "articulated", "wrong_object", "no_pick_near", "no_pick_far", "not_placed", "undone", "other"]


def classify(task, ep):
    goal = task["goal"]
    objects = task["objects"]
    w = ep["wait_steps"] + 1  # snapshots before the first policy action
    pos, grasp, preds, eef = ep["object_pos"][w:], ep["grasp"][w:], ep["goal"][w:], ep["eef"][w:]
    z0 = ep["object_pos"][w - 1, :, 2]
    lifted = pos[:, :, 2] - z0[None] >= LIFT
    picked_any = grasp & lifted
    first_pick = None
    hits = np.argwhere(picked_any)
    if len(hits):
        first_pick = objects[hits[hits[:, 0].argmin(), 1]]
    targets = [p[1] for p in goal if len(p) == 3 and p[1] in objects]
    info = {"first_pick": first_pick, "targets": targets,
            "final_true": int(preds[-1].sum()), "n_pred": len(goal)}
    if ep["success"]:
        return "success", info
    if len(goal) > 1 and 0 < preds[-1].sum() < len(goal):
        return "partial", info
    if not targets:
        return "articulated", info
    if first_pick is not None and first_pick not in targets:
        return "wrong_object", info
    target_idx = [objects.index(t) for t in targets]
    if first_pick is None:
        dist = np.linalg.norm(eef[:, None, :] - pos[:, target_idx, :], axis=-1).min()
        info["min_target_dist"] = float(dist)
        return ("no_pick_near" if dist < NEAR else "no_pick_far"), info
    if not preds.any(axis=0).all():
        return "not_placed", info
    if preds.any():
        return "undone", info
    return "other", info


def main():
    root = Path(sys.argv[1])
    rows = []
    for f in sorted(root.glob("*/*/task*.pkl")):
        model, suite, task_id = f.parts[-3], f.parts[-2], int(f.stem[4:6])
        rec = pickle.loads(f.read_bytes())
        for ep in rec["episodes"]:
            cat, info = classify(rec["task"], ep)
            rows.append({"model": model, "suite": suite, "task": task_id, "episode": int(ep["init_index"]),
                         "language": rec["task"]["language"], "category": cat, **info})
    (root / "taxonomy_rows.json").write_text(json.dumps(rows, indent=1))

    table = defaultdict(Counter)
    for r in rows:
        table[(r["model"], r["suite"])][r["category"]] += 1
    print("\n== categories per model and suite (episodes)")
    print(f"{'model':10s} {'suite':15s} {'n':>4s} {'succ':>5s} " + " ".join(f"{c:>12s}" for c in CATEGORIES))
    for (m, s), c in sorted(table.items()):
        n = sum(c.values())
        print(f"{m:10s} {s:15s} {n:4d} {c['success']:5d} " + " ".join(f"{c[k]:12d}" for k in CATEGORIES))

    # Paired: same (suite, task, initial-state index) across models.
    by_key = defaultdict(dict)
    for r in rows:
        by_key[(r["suite"], r["task"], r["episode"])][r["model"]] = r
    models = sorted({r["model"] for r in rows})
    for a in models:
        for b in models:
            if a == b:
                continue
            paired = Counter()
            for v in by_key.values():
                if a in v and b in v and v[a]["category"] != "success" and v[b]["category"] == "success":
                    paired[(v[a]["suite"], v[a]["category"])] += 1
            if paired:
                print(f"\n== {a} fails where {b} succeeds (same initial state)")
                for (s, c), k in sorted(paired.items()):
                    print(f"  {s:15s} {c:13s} {k}")

    per_task = defaultdict(Counter)
    for r in rows:
        per_task[(r["model"], r["suite"], r["task"], r["language"])][r["category"]] += 1
    print("\n== per task (failures only)")
    for (m, s, t, lang), c in sorted(per_task.items()):
        fails = {k: v for k, v in c.items() if k != "success"}
        if fails:
            print(f"{m:9s} {s:15s} t{t:02d} succ={c['success']:2d} {dict(fails)}  | {lang}")


if __name__ == "__main__":
    main()
