"""Render the LIBERO evaluation start frames with exact object positions, for the object-localization probe.

For each task and each of its 50 initial states: reset, set the init state, run the same 10 settling steps
as the evaluation, then save both camera images (rotated exactly as the policy sees them) and the world
positions of every movable object and fixture. One .npz per task.

  python render_scenes.py --libero_root .../LIBERO --suite libero_spatial --task 3 --out scenes/
(run with LIBERO_CONFIG_PATH, MUJOCO_GL=osmesa, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--libero_root", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--wait_steps", type=int, default=10)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=50)
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.libero_root).resolve()))
    sys.modules.pop("libero", None)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from turbovla.evaluation.suite_policy import get_libero_dummy_action, rotate_libero_image

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    states = suite.get_task_init_states(args.task)
    states = states[args.start:args.start + args.count]
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=args.res, camera_widths=args.res)
    env.seed(args.seed)
    inner = env.env
    objects, fixtures = list(inner.objects_dict), list(inner.fixtures_dict)
    dummy = list(get_libero_dummy_action())
    start = time.time()
    agent, wrist, obj_pos, fix_pos, eef = [], [], [], [], []
    for i, state in enumerate(states):
        if i % 10 == 0:
            print(f"state {i}/{len(states)} {time.time() - start:.1f}s", flush=True)
        env.reset()
        obs = env.set_init_state(state)
        for _ in range(args.wait_steps):
            obs, _, _, _ = env.step(dummy)
        agent.append(rotate_libero_image(obs["agentview_image"]))
        wrist.append(rotate_libero_image(obs["robot0_eye_in_hand_image"]))
        obj_pos.append([inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in objects])
        fix_pos.append([inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in fixtures])
        eef.append(np.asarray(obs["robot0_eef_pos"]))
    env.close()
    out = Path(args.out) / args.suite
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"task{args.task:02d}_s{args.start:02d}.npz",
        agent=np.stack(agent).astype(np.uint8), wrist=np.stack(wrist).astype(np.uint8),
        object_pos=np.asarray(obj_pos, np.float32), fixture_pos=np.asarray(fix_pos, np.float32),
        eef=np.asarray(eef, np.float32), init_index=np.arange(args.start, args.start + len(states)),
        meta=json.dumps({"language": task.language, "objects": objects, "fixtures": fixtures,
                         "goal": [list(g) for g in inner.parsed_problem["goal_state"]]}),
    )
    print("saved", out / f"task{args.task:02d}_s{args.start:02d}.npz", len(states))


if __name__ == "__main__":
    main()
