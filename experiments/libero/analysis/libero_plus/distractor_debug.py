"""Sanity check for libero_plus_distractor.py: are added objects found, where are they, what do wrong picks pick?"""
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[3])
from failure_taxonomy import classify  # noqa: E402

out, orig_root = Path(sys.argv[1]), Path(sys.argv[2])
original = {}
for f in orig_root.glob("*/task*_e00.pkl"):
    task = pickle.loads(f.read_bytes())["task"]
    original[Path(task["bddl"]).stem] = set(task["objects"])

for m in ("f7", "base"):
    n_added, d_target, d_eef, wrong, missing = [], [], [], Counter(), 0
    for f in sorted((out / m).glob("*/chunk*.pkl")):
        for ep in pickle.loads(f.read_bytes())["episodes"]:
            if "_add_" not in ep["task_name"] or ep.get("reset_failed"):
                continue
            task, objs = ep["task"], ep["task"]["objects"]
            stem = re.sub(r"_add_\d+$", "", ep["task_name"])
            if stem not in original:
                missing += 1
                continue
            added = [i for i, o in enumerate(objs) if o not in original[stem]]
            targets = [objs.index(p[1]) for p in task["goal"] if len(p) == 3 and p[1] in objs]
            n_added.append(len(added))
            w = ep["wait_steps"] + 1
            start = ep["object_pos"][w - 1]
            if added and targets:
                d_target.append(min(np.linalg.norm(start[a, :2] - start[t, :2]) for a in added for t in targets))
                d_eef.append(np.linalg.norm(ep["eef"][w:, None, :] - ep["object_pos"][w:, added, :], axis=-1).min())
            cat, info = classify(task, ep)
            if cat == "wrong_object":
                wrong[(info["first_pick"] in [objs[a] for a in added], info["first_pick"])] += 1
    print(f"== {m}: tasks {len(n_added)}, stem not found {missing}, added per task {Counter(n_added)}")
    print(f"   start distance added -> nearest target (cm): median {100 * np.median(d_target):.1f}, "
          f"10th pct {100 * np.percentile(d_target, 10):.1f}")
    print(f"   closest end-effector approach to an added object (cm): median {100 * np.median(d_eef):.1f}, "
          f"share < 4 cm {np.mean(np.array(d_eef) < 0.04):.2f}")
    print(f"   wrong_object first picks (is_added, name): {wrong.most_common(8)}")
