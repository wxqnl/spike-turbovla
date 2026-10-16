"""What do added objects do to each model in LIBERO-plus "Objects Layout: added objects" tasks?

Added objects are the objects of a perturbed task that the original LIBERO task does not have (original object lists
from the LIBERO records of failure_runs.sh). Per model, over the added-object tasks:
- failure categories of failure_taxonomy.classify (wrong_object = the first picked object is not a target);
- picked_added: an added object was grasped and lifted >= 2 cm (the policy went for it);
- reached_added_first: the end effector came within 4 cm of an added object before it came within 4 cm of a target;
- pushed_added: an added object moved >= 2 cm without being grasped (contact, the arm ran into it).
Paired: the same counts for the tasks one model fails and the other succeeds.
Usage: python libero_plus_distractor.py <layout_out_dir> <libero_records_root> <model_a> <model_b>
   e.g. ... analysis/libero_plus/layout analysis/failure/full/base f7 base
"""
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from failure_taxonomy import classify  # noqa: E402

LIFT, NEAR = 0.02, 0.04
out, orig_root, models = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:5]

original = {}  # original bddl stem -> object names
for f in orig_root.glob("*/task*_e00.pkl"):
    task = pickle.loads(f.read_bytes())["task"]
    original[Path(task["bddl"]).stem] = set(task["objects"])


def measures(ep):
    task = ep["task"]
    objs = task["objects"]
    stem = re.sub(r"_add_\d+$", "", ep["task_name"])
    added = [i for i, o in enumerate(objs) if o not in original[stem]]
    targets = [objs.index(p[1]) for p in task["goal"] if len(p) == 3 and p[1] in objs]
    w = ep["wait_steps"] + 1
    pos, grasp, eef = ep["object_pos"][w:], ep["grasp"][w:], ep["eef"][w:]
    start = ep["object_pos"][w - 1]
    lifted = pos[:, :, 2] - start[None, :, 2] >= LIFT
    picked = (grasp & lifted).any(0)
    moved = np.linalg.norm(pos[:, :, :2] - start[None, :, :2], axis=-1).max(0) >= LIFT
    dist = np.linalg.norm(eef[:, None, :] - pos, axis=-1)
    first_near = lambda idx: min([int(np.argmax(dist[:, i] < NEAR)) if (dist[:, i] < NEAR).any() else 10 ** 9
                                  for i in idx] or [10 ** 9])
    category = classify(task, ep)[0]
    return {"category": category,
            "picked_added": bool(picked[added].any()),
            "reached_added_first": first_near(added) < first_near(targets),
            "pushed_added": bool((moved & ~grasp.any(0))[added].any())}


rows = {m: {} for m in models}
for m in models:
    for f in sorted((out / m).glob("*/chunk*.pkl")):
        for ep in pickle.loads(f.read_bytes())["episodes"]:
            if "_add_" in ep["task_name"] and not ep.get("reset_failed"):
                rows[m][(f.parts[-2], ep["task_id"])] = measures(ep)

shared = rows[models[0]].keys() & rows[models[1]].keys()


def report(title, keys, m):
    rs = [rows[m][k] for k in keys]
    n = len(rs)
    fails = [r for r in rs if r["category"] != "success"]
    cats = Counter(r["category"] for r in fails)
    flags = {f: sum(r[f] for r in rs) for f in ("picked_added", "reached_added_first", "pushed_added")}
    flags_fail = {f: sum(r[f] for r in fails) for f in flags}
    print(f"{title:38s} {m:5s} n={n:4d} success {n - len(fails):4d}  failures {len(fails):4d} {dict(cats.most_common())}")
    print(f"{'':38s} {'':5s} all episodes: {flags}   failures only: {flags_fail}")


for m in models:
    report("added-object tasks", sorted(shared), m)
a, b = models
report(f"{a} fails, {b} succeeds", [k for k in shared if rows[a][k]["category"] != "success" and rows[b][k]["category"] == "success"], a)
report(f"{b} fails, {a} succeeds", [k for k in shared if rows[b][k]["category"] != "success" and rows[a][k]["category"] == "success"], b)
