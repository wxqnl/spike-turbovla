"""Paired comparison of two models on LIBERO-plus layout tasks, split into moved and added-object tasks (exact McNemar)."""
import json
import pickle
import re
import sys
from math import comb
from pathlib import Path

out, table = Path(sys.argv[1]), json.load(open(sys.argv[2]))
a, b = sys.argv[3], sys.argv[4]
names = {(s, r["id"] - 1): r["name"] for s, rows in table.items() for r in rows}
res = {a: {}, b: {}}
for m in (a, b):
    for f in sorted((out / m).glob("*/chunk*.pkl")):
        for ep in pickle.loads(f.read_bytes())["episodes"]:
            if not ep.get("reset_failed"):
                res[m][(f.parts[-2], ep["task_id"])] = ep["success"]
shared = res[a].keys() & res[b].keys()
for kind in ("moved", "added objects"):
    keys = [k for k in shared if bool(re.search(r"_level\d+_sample\d+$", names[k])) == (kind == "moved")]
    only_a = sum(res[a][k] and not res[b][k] for k in keys)
    only_b = sum(res[b][k] and not res[a][k] for k in keys)
    n = only_a + only_b
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    sa, sb = sum(res[a][k] for k in keys), sum(res[b][k] for k in keys)
    print(f"{kind:14s} n={len(keys)}  {a} {sa} ({100 * sa / len(keys):.1f}%)  {b} {sb} ({100 * sb / len(keys):.1f}%)  "
          f"only {a} {only_a}, only {b} {only_b}, exact McNemar p={p:.2g}")
