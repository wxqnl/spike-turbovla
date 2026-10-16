"""Success rates of LIBERO-plus records (libero_plus_runs.sh) per model, suite, category and difficulty level.

Objects Layout holds two kinds of tasks, told apart by the task name: added distractor objects ("..._add_N") and
moved objects ("..._levelL_sampleS", L = 1 to 5). Paired counts compare two models on the same perturbed tasks.
Records are found below <out_dir> at any depth as <model>/<suite>/chunkNNN.pkl, so one call can pool the output
directories of several categories (e.g. analysis/libero_plus).
Usage: python libero_plus_summary.py <out_dir> <task_classification.json>
"""
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

out, table = Path(sys.argv[1]), json.load(open(sys.argv[2]))
meta = {(suite, row["id"] - 1): row for suite, rows in table.items() for row in rows}


def kind(row):
    if row["category"] != "Objects Layout":
        return row["category"]
    return "Layout: moved" if re.search(r"_level\d+_sample\d+$", row["name"]) else "Layout: added objects"


results = defaultdict(dict)  # model -> (suite, task_id) -> success
failed_reset = defaultdict(set)  # tasks whose scene could not be placed (libero_plus_rollout.py), per model
for f in sorted(out.rglob("chunk*.pkl")):
    model, suite = f.parts[-3], f.parts[-2]
    for ep in pickle.loads(f.read_bytes())["episodes"]:
        if ep.get("reset_failed"):
            failed_reset[model].add((suite, ep["task_id"]))
        else:
            results[model][(suite, ep["task_id"])] = ep["success"]

models = sorted(results)
# A task that could not be placed for any model is left out for all of them.
excluded = set().union(*failed_reset.values()) if failed_reset else set()
for model in models:
    for key in excluded:
        results[model].pop(key, None)
print("scene placement failed (left out): " + ", ".join(f"{m} {len(failed_reset[m])}" for m in models)
      + f"; {len(excluded)} tasks excluded for every model\n")
groups = defaultdict(lambda: defaultdict(list))
for model in models:
    for key, success in results[model].items():
        row = meta[key]
        groups[(kind(row), key[0])][model].append(success)
        groups[(kind(row), "all suites")][model].append(success)
        groups[(f"{kind(row)}, level {row['difficulty_level']}", "all suites")][model].append(success)

print(f"{'group':40s} {'suite':15s} " + " ".join(f"{m:>18s}" for m in models))
for (group, suite), by_model in sorted(groups.items()):
    cells = []
    for m in models:
        v = by_model.get(m, [])
        cells.append(f"{sum(v):5d}/{len(v):<5d}{100 * sum(v) / max(len(v), 1):5.1f}%" if v else " " * 18)
    print(f"{group:40s} {suite:15s} " + " ".join(f"{c:>18s}" for c in cells))

for a in models:
    for b in models:
        if a < b:
            shared = results[a].keys() & results[b].keys()
            only_a = sum(results[a][k] and not results[b][k] for k in shared)
            only_b = sum(results[b][k] and not results[a][k] for k in shared)
            print(f"\npaired on {len(shared)} tasks: {a} only {only_a}, {b} only {only_b}")
