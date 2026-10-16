"""How far are the original objects from their LIBERO positions in LIBERO-plus layout tasks (added vs moved)?

Reference: the mean start position of each object over the 50 LIBERO initial states of the original task (base
records of failure_runs.sh). For each perturbed episode: the largest xy displacement among the goal's target
objects, and among all original objects.
"""
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

out, orig_root = Path(sys.argv[1]), Path(sys.argv[2])
ref = {}
for f in orig_root.glob("*/task*_e00.pkl"):
    rec = pickle.loads(f.read_bytes())
    w = rec["episodes"][0]["wait_steps"]
    start = np.stack([ep["object_pos"][w] for ep in rec["episodes"]])
    ref[Path(rec["task"]["bddl"]).stem] = (rec["task"]["objects"], start.mean(0), start.std(0).max())

stats = defaultdict(list)
for f in sorted((out / "base").glob("*/chunk*.pkl")):
    for ep in pickle.loads(f.read_bytes())["episodes"]:
        if ep.get("reset_failed"):
            continue
        name = ep["task_name"]
        kind = "added" if re.search(r"_add_\d+$", name) else "moved" if re.search(r"_level\d+_sample\d+$", name) else None
        if kind is None:
            continue
        stem = re.sub(r"(_add_\d+|_level\d+_sample\d+)$", "", name)
        if stem not in ref:
            stats[(kind, "no reference")].append(0)
            continue
        objs0, mean0, _ = ref[stem]
        objs = ep["task"]["objects"]
        w = ep["wait_steps"]
        start = ep["object_pos"][w]
        shift = {o: np.linalg.norm(start[objs.index(o), :2] - mean0[i, :2]) for i, o in enumerate(objs0) if o in objs}
        targets = [p[1] for p in ep["task"]["goal"] if len(p) == 3 and p[1] in shift]
        stats[(kind, "max target shift")].append(max([shift[t] for t in targets] or [0]))
        stats[(kind, "max object shift")].append(max(shift.values()))
for (kind, what), v in sorted(stats.items()):
    v = 100 * np.array(v)
    print(f"{kind:6s} {what:18s} n={len(v):4d} median {np.median(v):5.1f} cm  75th {np.percentile(v, 75):5.1f}  "
          f"share > 3 cm {np.mean(v > 3):.2f}")
print("LIBERO's own spread over 50 initial states: max std per task (cm):",
      np.round(100 * np.array(sorted(r[2] for r in ref.values())), 1)[[0, len(ref) // 2, -1]])
