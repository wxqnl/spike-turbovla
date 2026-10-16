"""LIBERO-plus evaluation with the per-step simulator records of failure_rollout.py.

LIBERO-plus (arXiv 2510.13626) perturbs the LIBERO tasks along seven dimensions (object layout, camera, robot
initial state, language, light, background, sensor noise). Each perturbed task has its own initial state and runs
once (num_trials_per_task 1, as the benchmark prescribes), so one process runs a list of task ids in order and one
record file holds many tasks: every episode carries its task id and task metadata. The record is rewritten after
every episode, so a crash keeps the finished episodes, and a rerun with the same record path skips the task ids
already in it.

LIBERO-plus's ControlEnv.reset retries forever, and its `finally: continue` swallows every exception, not only
RandomizationError. On 2026-10-15 processes spun for 16 h at the 18th to 22nd task of a chunk, the same chunks for F7
and the v1 base: each new scene leaks file handles, and once the 65536 limit was gone the mesh loads inside reset
failed and were retried forever (the same tasks run fine in a fresh process). libero_plus_runs.sh raises the limit;
here reset also retries placement at most MAX_RESET_TRIES times and lets any other error end the process, so a rerun
resumes it. An episode whose scene cannot be placed is recorded with reset_failed=True, and summaries leave it out.

Run inside the eval environment with LIBERO_CONFIG_PATH pointing at the LIBERO-plus config and PYTHONPATH holding
its extra packages (wand, scikit-image), with the usual evaluate.py arguments plus --record_path:
  python libero_plus_rollout.py --record_path out/spatial_000.pkl --libero_root .../LIBERO-plus \
      --task_suite_name libero_spatial --task_ids 1725,1726 --num_trials_per_task 1 --ckpt_path ... [evaluate.py args]
"""
try:  # LIBERO-plus imports wand (sensor noise); load MagickWand before the policy pulls in older system libraries
    import wand.image  # noqa: F401
except ImportError:
    pass
import pickle
import sys
from pathlib import Path

import numpy as np
from robosuite.utils.errors import RandomizationError

from vla_adapter import rollout
from failure_rollout import RecordingEnv

MAX_RESET_TRIES = 100


class ResetFailed(RuntimeError):
    pass


class FailedEnv:
    def close(self):
        pass


def bounded_reset(self):
    for _ in range(MAX_RESET_TRIES):
        try:
            return self.env.reset()
        except RandomizationError:
            pass
    raise ResetFailed(f"object placement failed {MAX_RESET_TRIES} times")


def main():
    argv = sys.argv[1:]
    i = argv.index("--record_path")
    record_path = Path(argv[i + 1])
    del argv[i:i + 2]
    record_path.parent.mkdir(parents=True, exist_ok=True)
    sink = {"episodes": []}
    if record_path.exists():
        sink = pickle.loads(record_path.read_bytes())
        done = {ep["task_id"] for ep in sink["episodes"]}
        j = argv.index("--task_ids")
        left = [t for t in argv[j + 1].split(",") if int(t) not in done]
        if not left:  # every task already recorded: leave the completion marker the runner looks for
            k = argv.index("--result_json_path")
            Path(argv[k + 1]).write_text('{"note": "all tasks already in the record"}\n')
            return
        argv[j + 1] = ",".join(left)
    sys.argv = [sys.argv[0], *argv]
    current = {}
    make_env, run_episode = rollout._make_libero_env, rollout._run_episode

    def recording_make_env(task, cfg):
        from libero.libero import benchmark
        from libero.libero.envs.env_wrapper import ControlEnv
        ControlEnv.reset = bounded_reset
        suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
        current["task_id"] = [t.name for t in suite.tasks].index(task.name)
        current["task_name"] = task.name
        current["count"] = 0
        current["make_failed"] = False
        try:
            env, description = make_env(task, cfg)
        except RandomizationError:  # placement already fails when the env is built
            current["make_failed"] = True
            sink["task"] = {"language": task.language, "bddl": task.bddl_file}
            return FailedEnv(), task.language
        current["env"] = RecordingEnv(env, task, sink)
        return current["env"], description

    def recording_run_episode(cfg, env, policy, description, initial_state, *args, **kwargs):
        index = current["count"]
        current["count"] += 1
        base = {"task_id": current["task_id"], "task_name": current["task_name"], "init_index": index,
                "task": dict(sink["task"]), "wait_steps": cfg.num_steps_wait}
        try:
            if current["make_failed"]:
                raise ResetFailed("object placement failed when the env was built")
            success, images, actions = run_episode(cfg, env, policy, description, initial_state, *args, **kwargs)
        except ResetFailed:
            success, images, actions = False, [], []
            sink["episodes"].append({"success": False, "reset_failed": True, **base})
        else:
            sink["episodes"].append({"success": bool(success), "actions": np.asarray(actions, np.float32),
                                     **base, **env.episode()})
        tmp = record_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(sink, f)
        tmp.replace(record_path)
        return success, images, actions

    rollout._make_libero_env, rollout._run_episode = recording_make_env, recording_run_episode
    rollout.main()


if __name__ == "__main__":
    main()
