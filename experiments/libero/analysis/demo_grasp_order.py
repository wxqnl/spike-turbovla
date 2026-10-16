"""Which object do the LIBERO-Long demos grasp first? Two-object tasks only."""
import json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

root = Path(sys.argv[1])
tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in (root / "meta/tasks.jsonl").read_text().splitlines()}
grasps = {}
for path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
    t = pq.read_table(path, columns=["observation.state", "action", "task_index"])
    state = np.array(t.column("observation.state").to_pylist())
    grip = np.array(t.column("action").to_pylist())[:, 6]
    task = t.column("task_index")[0].as_py()
    opened = grip[0]
    closes = [i for i in range(1, len(grip)) if grip[i] != opened and grip[i - 1] == opened]
    grasps.setdefault(task, []).append([state[i, :3] for i in closes])

for task in sorted(grasps):
    eps = [g for g in grasps[task] if len(g) >= 2]
    if len(eps) < len(grasps[task]) // 2:
        print(f"task {task}: single-grasp task, skipped | {tasks[task]}")
        continue
    first = np.array([g[0] for g in eps]); second = np.array([g[1] for g in eps])
    pts = np.concatenate([first, second])[:, :2]
    # 2-means on all grasp locations
    c = pts[[np.argmin(pts[:, 0]), np.argmax(pts[:, 0])]].copy()
    for _ in range(50):
        lab = np.argmin(((pts[:, None] - c[None]) ** 2).sum(-1), 1)
        c = np.array([pts[lab == j].mean(0) for j in range(2)])
    lf = np.argmin(((first[:, :2, None].transpose(0, 2, 1) - c[None]) ** 2).sum(-1), 1)
    ls = np.argmin(((second[:, :2, None].transpose(0, 2, 1) - c[None]) ** 2).sum(-1), 1)
    order_a = int(((lf == 0) & (ls == 1)).sum()); order_b = int(((lf == 1) & (ls == 0)).sum())
    sep = np.linalg.norm(c[0] - c[1])
    print(f"task {task}: {len(grasps[task])} demos, >=2 grasps {len(eps)}, grasp sites {sep*100:.0f} cm apart, "
          f"order A->B {order_a}, B->A {order_b} | {tasks[task]}")
