"""Success on LIBERO-plus added-object tasks by the number of added objects, per model and suite."""
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

out, orig_root = Path(sys.argv[1]), Path(sys.argv[2])
original = {}
for f in orig_root.glob("*/task*_e00.pkl"):
    task = pickle.loads(f.read_bytes())["task"]
    original[Path(task["bddl"]).stem] = set(task["objects"])

table = defaultdict(lambda: [0, 0])
for m in ("f7", "base"):
    for f in sorted((out / m).glob("*/chunk*.pkl")):
        for ep in pickle.loads(f.read_bytes())["episodes"]:
            if "_add_" not in ep["task_name"] or ep.get("reset_failed"):
                continue
            stem = re.sub(r"_add_\d+$", "", ep["task_name"])
            k = sum(o not in original[stem] for o in ep["task"]["objects"])
            for key in ((m, k, "all"), (m, k, f.parts[-2])):
                table[key][0] += ep["success"]
                table[key][1] += 1
for k in range(1, 6):
    cells = [f"{m} {table[(m, k, 'all')][0]:3d}/{table[(m, k, 'all')][1]:3d} "
             f"{100 * table[(m, k, 'all')][0] / table[(m, k, 'all')][1]:5.1f}%" for m in ("f7", "base")]
    print(f"{k} added: " + "   ".join(cells))
