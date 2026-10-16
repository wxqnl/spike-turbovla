"""Official LIBERO evaluation plus a per-step record of the simulator state, for failure analysis.

The rollout itself is `vla_adapter.rollout.eval_libero`, unchanged (same seeds, init states, wait steps,
open-loop chunking and success test); we only wrap the environment it creates. After every env step we
record the end-effector, every movable object's position, which objects the gripper is grasping, and the
truth value of each BDDL goal predicate. One pickle per task, one entry per episode.

Episode order is part of the protocol: every hard reset regenerates the scene model from the RNG seeded once per
task, and set_init_state overwrites only the joint state, so episode k depends on the k resets before it (checked:
the same initial state run third in one process and first in another gives different actions, for the spiking
policy and for TurboVLA alike). So each task's 50 episodes run in one process, in order, as in the official runs.
Each hard reset leaks ~50 open mesh files, so the process needs a high open-file limit (launch/env.example.sh sets
ulimit -n 65536); at the default 1024 the 21st reset fails and LIBERO's reset loop retries it forever.
--episode_start s (default 0) is kept for debugging only.

Run inside the eval environment (LIBERO_CONFIG_PATH, MUJOCO_GL=osmesa, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1),
with the usual evaluate.py arguments plus --record_path [--episode_start s]:
  python failure_rollout.py --record_path out/task03_e13.pkl --task_ids 3 --episode_start 13 \
      --num_trials_per_task 13 --ckpt_path ... [evaluate.py args]
"""
import pickle
import sys
from pathlib import Path

import numpy as np

from vla_adapter import rollout


class RecordingEnv:
    """Forwards everything to the LIBERO env; logs simulator state after each step."""

    def __init__(self, env, task, sink):
        self._env, self._sink = env, sink
        inner = env.env
        self._inner = inner
        self._movable = list(inner.objects_dict)
        self._goal = [list(p) for p in inner.parsed_problem["goal_state"]]
        sink["task"] = {"language": task.language, "bddl": task.bddl_file, "goal": self._goal,
                        "objects": self._movable, "fixtures": list(inner.fixtures_dict)}
        self._steps = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _snapshot(self, obs):
        inner = self._inner
        gripper = inner.robots[0].gripper
        positions = np.stack([inner.sim.data.body_xpos[inner.obj_body_id[n]] for n in self._movable]).astype(np.float32)
        grasp = np.array([bool(inner._check_grasp(gripper, inner.objects_dict[n])) for n in self._movable])
        goal = np.array([bool(inner._eval_predicate(p)) for p in self._goal])
        self._steps.append((np.asarray(obs["robot0_eef_pos"], np.float32),
                            np.asarray(obs["robot0_gripper_qpos"], np.float32), positions, grasp, goal))

    def set_init_state(self, state):
        obs = self._env.set_init_state(state)
        self._steps = []
        self._snapshot(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._snapshot(obs)
        return obs, reward, done, info

    def episode(self):
        eef, grip, pos, grasp, goal = (np.stack(x) for x in zip(*self._steps))
        return {"eef": eef, "gripper_qpos": grip, "object_pos": pos, "grasp": grasp, "goal": goal}


def main():
    argv = sys.argv[1:]
    options = {}
    for flag in ("--record_path", "--episode_start"):
        if flag in argv:
            i = argv.index(flag)
            options[flag] = argv[i + 1]
            del argv[i:i + 2]
    record_path = Path(options["--record_path"])
    start = int(options.get("--episode_start", 0))
    sys.argv = [sys.argv[0], *argv]
    sink = {"episodes": [], "episode_start": start}
    current = {}
    make_env, run_episode = rollout._make_libero_env, rollout._run_episode

    def recording_make_env(task, cfg):
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
        task_id = [t.name for t in suite.tasks].index(task.name)
        current["states"] = suite.get_task_init_states(task_id)
        current["count"] = 0
        env, description = make_env(task, cfg)
        current["env"] = RecordingEnv(env, task, sink)
        return current["env"], description

    def recording_run_episode(cfg, env, policy, description, initial_state, *args, **kwargs):
        index = start + current["count"]
        current["count"] += 1
        initial_state = current["states"][index]
        success, images, actions = run_episode(cfg, env, policy, description, initial_state, *args, **kwargs)
        sink["episodes"].append({"success": bool(success), "init_index": index,
                                 "actions": np.asarray(actions, np.float32),
                                 "wait_steps": cfg.num_steps_wait, **env.episode()})
        return success, images, actions

    rollout._make_libero_env, rollout._run_episode = recording_make_env, recording_run_episode
    rollout.main()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "wb") as f:
        pickle.dump(sink, f)


if __name__ == "__main__":
    main()
