"""Fixed public MT50 50-task x 50-episode evaluation of a final 70k EMA."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PREP = Path(__file__).resolve().parent
REPO = PREP.parents[2]
sys.path.insert(0, str(PREP))
from episode_store import EpisodeStore, durable_publish, encode

ROOT = Path(os.environ["MT50_RUN_ROOT"]).resolve()
OUT = Path(os.environ.get("MT50_EVAL_OUT", str(ROOT / "evaluation_public2500_70k_ema"))).resolve()
EXT = ROOT / "continuation_70k_anneal_after_60k"
EMA = Path(os.environ["MT50_EMA"]).resolve()
PY = sys.executable
GPUS = {int(index): uuid for index, uuid in json.loads(os.environ["MT50_GPU_MAP"]).items()}
INDICES = sorted(GPUS)
if len(INDICES) != 4 or len(set(GPUS.values())) != 4:
    raise ValueError("Provide exactly four distinct GPU indices and UUIDs")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    durable_publish(path, encode(value), replace=True)


def environment(gpu):
    env = os.environ.copy()
    model_source = Path(env["MT50_MODEL_SOURCE"]).resolve()
    env["CUDA_VISIBLE_DEVICES"] = GPUS[gpu]
    env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    env["MUJOCO_GL"] = "egl"
    env["PYTHONPATH"] = ":".join((str(model_source), str(PREP), str(REPO), env.get("PYTHONPATH", "")))
    env.update(PYOPENGL_PLATFORM="egl", MKL_NUM_THREADS="2", TRANSFORMERS_OFFLINE="1")
    env.pop("NVIDIA_TF32_OVERRIDE", None)
    env.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    return env


def require_free(gpu):
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    row = [list(map(str.strip, item.split(","))) for item in rows.splitlines()
           if int(item.split(",")[0]) == gpu][0]
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    if row[1] != GPUS[gpu]:
        raise RuntimeError(f"Assigned GPU{gpu} UUID changed")
    if int(row[2]) > 100 or GPUS[gpu] in occupied:
        raise RuntimeError(f"Assigned GPU{gpu} is unavailable")
    import xml.etree.ElementTree as ET
    node = ET.fromstring(subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "-q", "-x"], text=True
    )).find("gpu")
    if (node.findtext("gpu_recovery_action") != "None"
            or node.findtext("fabric/state") != "Completed"
            or node.findtext("fabric/status") != "Success"):
        raise RuntimeError(f"GPU{gpu} recovery/fabric gate failed")


def wait_training_ema():
    start = time.monotonic()
    while True:
        status = json.loads((EXT / "status.json").read_text())
        if status.get("stage") == "needs_autonomous_repair":
            raise RuntimeError("Own 70k training needs repair before evaluation")
        if ((EXT / "overall.exit_code").is_file()
                and status.get("stage") == "70k_training_ema_complete_pending_evaluation"):
            break
        write(OUT / "status.json", {
            "stage": "waiting_for_70k_training_ema",
            "pid": os.getpid(),
            "training_pid": status.get("pid"),
            "training_stage": status.get("stage"),
            "total_episodes": 2500,
            "time": time.time(),
        })
        if time.monotonic() - start > 1209600:
            raise TimeoutError("70k training/EMA wait exceeded fourteen days")
        time.sleep(15)
    require_training_ema()


def wait_gpus():
    start = time.monotonic()
    while True:
        try:
            for gpu in INDICES:
                require_free(gpu)
            return
        except RuntimeError as error:
            if "is unavailable" not in str(error):
                raise
        write(OUT / "status.json", {
            "stage": "waiting_for_assigned_gpus", "pid": os.getpid(),
            "physical_gpus": INDICES, "total_episodes": 2500, "time": time.time(),
        })
        if time.monotonic() - start > 172800:
            raise TimeoutError("Authorized evaluation GPUs occupied for two days")
        time.sleep(15)


def require_training_ema():
    if (EXT / "overall.exit_code").read_text().strip() != "0":
        raise RuntimeError("Own 70k training/EMA overall receipt is not successful")
    status = json.loads((EXT / "status.json").read_text())
    progress = json.loads((EXT / "training/progress.json").read_text())
    if (status.get("stage") != "70k_training_ema_complete_pending_evaluation"
            or progress.get("step") != 70000 or progress.get("samples_seen") != 17920000
            or not EMA.is_file()):
        raise RuntimeError("Own final 70k EMA is incomplete")
    for name in ("train", "refresh"):
        receipt = json.loads((EXT / "training" / f"{name}.success.json").read_text())
        if receipt["exit_code"] != 0 or Path(receipt["exit_file"]).read_text().strip() != "0":
            raise RuntimeError(f"Own 70k {name} success receipt is inconsistent")


def manifest():
    mapping = json.loads((PREP / "task_mapping.json").read_text())
    tasks = {item["native_task_id"]: item for item in mapping["tasks"]}
    if set(tasks) != set(range(50)):
        raise RuntimeError("Expected exactly 50 native MT50 tasks")
    episodes = [
        {"native_task_id": task, "episode": episode,
         "reset_seed": 202609210 + task * 50 + episode,
         "env_name": tasks[task]["env_name"]}
        for task in range(50) for episode in range(50)
    ]
    value = {
        "episodes": episodes,
        "protocol": {
            "variant": "spike_driven_vla_ema70k",
            "source": os.environ.get("MT50_TRAINING_SOURCE", os.environ["MT50_MODEL_SOURCE"]),
            "ema": str(EMA),
            "global_step": 70000,
            "samples_seen": 17920000,
            "weight_kind": "training_data_refreshed_ema",
            "ema_refresh": {"batches": 128, "batch_size": 16, "optimizer_steps": 0},
            "dataset_repo": "lerobot/metaworld_mt50",
            "dataset_revision": "v3.0",
            "tasks": 50,
            "episodes_per_task": 50,
            "total_episodes": 2500,
            "reset_seed_rule": "202609210 + native_task_id*50 + episode",
            "mt1_seed": 42,
            "policy_seed": 7,
            "camera": "corner2",
            "camera_position": [0.75, 0.075, 0.7],
            "flip_both_axes": True,
            "native_rgb": 480,
            "resize": "PIL bilinear256, no JPEG",
            "state": "training state4 zscore",
            "action": "training minmax inverse, continuous gripper",
            "openloop": 12,
            "max_steps": 500,
            "precision": "fp32",
            "matmul": "highest",
            "tf32": False,
            "physical_gpus": INDICES,
            "worker_assignment": "native_task_id modulo 4",
        },
    }
    path = OUT / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError("Stored public MT50 manifest changed")
    else:
        durable_publish(path, encode(value))
    return value


def worker(index, resume):
    import torch
    from adapter import PublicMT50Environment
    from rollout import rollout_episode
    from policy import FinalEMA70kPolicy

    gpu = INDICES[index]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPUS[gpu]:
        raise RuntimeError("Wrong visible GPU UUID")
    policy = FinalEMA70kPolicy(7)
    plan = json.loads((OUT / "manifest.json").read_text())
    folder = OUT / f"worker{index}"
    with EpisodeStore(
        folder,
        plan,
        policy.ready,
        resume=resume,
        task_ids=list(range(index, 50, 4)),
        expected_variants=("spike_driven_vla_ema70k",),
        expected_global_step=70000,
        expected_samples_seen=17920000,
    ) as store:
        write(folder / "ready.json", {
            **policy.ready,
            "worker": index,
            "physical_gpu": gpu,
            "gpu_uuid": GPUS[gpu],
            "cuda_name": torch.cuda.get_device_name(0),
            "renderer_backend": "egl",
            "renderer_verified": bool(store.completed),
            "time": time.time(),
        })
        env = None
        try:
            for item in store.pending():
                task = item["native_task_id"]
                if env is None or env.native_task_id != task:
                    if env is not None:
                        env.close()
                    env = PublicMT50Environment(policy.adapter.tasks[task])
                if env.name != item["env_name"]:
                    raise RuntimeError("Manifest/native environment mismatch")
                result = rollout_episode(policy, env, item["reset_seed"])
                result["episode"] = item["episode"]
                store.commit(result)
                write(folder / "progress.json", {
                    "worker": index,
                    "completed": len(store.completed),
                    "total": len(store.expected),
                    "successes": sum(item["success"] for item in store.completed.values()),
                    "last": result,
                    "time": time.time(),
                })
                if len(store.completed) == 1:
                    ready = json.loads((folder / "ready.json").read_text())
                    ready["renderer_verified"] = True
                    write(folder / "ready.json", ready)
                print(json.dumps({"event": "episode", **result}), flush=True)
        finally:
            if env is not None:
                env.close()
        if len(store.completed) != len(store.expected):
            raise RuntimeError("Incomplete 70k evaluation worker")


def controller(resume):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "evaluation.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / "status.json").exists() and not resume:
        raise RuntimeError("Existing 70k evaluation requires explicit resume")
    if resume and (OUT / "status.json").exists():
        previous = json.loads((OUT / "status.json").read_text())
        if (OUT / "overall.exit_code").exists():
            raise RuntimeError("Completed 70k evaluation must not restart")
        for pid in previous.get("running", {}).values():
            if Path(f"/proc/{pid}").exists():
                raise RuntimeError("A previous worker may still be alive")
    processes = {}
    attempts = {}
    codes = {}
    try:
        wait_training_ema()
        plan = manifest()
        wait_gpus()
        for index, gpu in enumerate(INDICES):
            folder = OUT / f"worker{index}"
            attempt = 1
            while (OUT / f"worker{index}.attempt{attempt}.log").exists():
                attempt += 1
            attempts[index] = attempt
            command = [PY, "-u", str(Path(__file__)), "--worker", str(index)]
            if (folder / "binding.json").exists():
                if not resume:
                    raise RuntimeError("Existing worker results require explicit resume")
                command.append("--resume")
            with (OUT / f"worker{index}.attempt{attempt}.log").open("xb", buffering=0) as stream:
                process = subprocess.Popen(
                    command,
                    cwd=str(REPO),
                    env=environment(gpu),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes[index] = process
        start = time.monotonic()
        while processes:
            for index, process in list(processes.items()):
                return_code = process.poll()
                if return_code is not None:
                    durable_publish(
                        OUT / f"worker{index}.attempt{attempts[index]}.exit_code",
                        f"{return_code}\n".encode(),
                    )
                    codes[index] = return_code
                    del processes[index]
            counts = {}
            successes = {}
            for index in range(4):
                result_path = OUT / f"worker{index}/result.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    counts[str(index)] = result["completed_episodes"]
                    successes[str(index)] = result["successes"]
            write(OUT / "status.json", {
                "stage": "evaluating",
                "pid": os.getpid(),
                "time": time.time(),
                "running": {str(index): process.pid for index, process in processes.items()},
                "worker_exit_codes": codes,
                "completed_episodes": sum(counts.values()),
                "successes": sum(successes.values()),
                "counts": counts,
                "total_episodes": 2500,
            })
            if processes and time.monotonic() - start > 172800:
                for process in processes.values():
                    os.killpg(process.pid, signal.SIGTERM)
                for index, process in processes.items():
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    durable_publish(
                        OUT / f"worker{index}.attempt{attempts[index]}.exit_code",
                        f"{process.returncode}\n".encode(),
                    )
                raise TimeoutError("70k evaluation exceeded its two-day hard timeout")
            if processes:
                time.sleep(5)
        if any(codes.get(index) != 0 for index in range(4)):
            raise RuntimeError("A worker failed; explicit safe resume is required")
        unique = {}
        by_task = {}
        successes = 0
        for index in range(4):
            binding = json.loads((OUT / f"worker{index}/binding.json").read_text())
            with EpisodeStore(
                OUT / f"worker{index}",
                plan,
                binding["provenance"],
                resume=True,
                task_ids=list(range(index, 50, 4)),
                expected_variants=("spike_driven_vla_ema70k",),
                expected_global_step=70000,
                expected_samples_seen=17920000,
            ) as store:
                if len(store.completed) != len(store.expected):
                    raise RuntimeError("Final worker coverage is incomplete")
                for key, result in store.completed.items():
                    if key in unique or key[0] % 4 != index:
                        raise RuntimeError("Duplicate or wrong worker assignment")
                    unique[key] = result
                    successes += result["success"]
                    entry = by_task.setdefault(str(key[0]), {"episodes": 0, "successes": 0})
                    entry["episodes"] += 1
                    entry["successes"] += result["success"]
        if (set(unique) != {(task, episode) for task in range(50) for episode in range(50)}
                or any(item["episodes"] != 50 for item in by_task.values())):
            raise RuntimeError("Final 70k coverage changed")
        write(OUT / "summary.json", {
            "total_episodes": 2500,
            "total_successes": successes,
            "success_rate": successes / 2500,
            "tasks": by_task,
            "protocol": plan["protocol"],
            "unique_episode_count": 2500,
            "worker_exit_codes": {str(index): codes[index] for index in range(4)},
            "time": time.time(),
        })
        write(OUT / "status.json", {
            "stage": "complete",
            "pid": os.getpid(),
            "completed_episodes": 2500,
            "total_episodes": 2500,
            "successes": successes,
            "time": time.time(),
        })
        durable_publish(OUT / "overall.exit_code", b"0\n")
    except BaseException as error:
        write(OUT / "status.json", {
            "stage": "failed",
            "pid": os.getpid(),
            "error": repr(error),
            "time": time.time(),
            "running": {str(index): process.pid for index, process in processes.items()},
            "worker_exit_codes": codes,
        })
        failed = 1
        while (OUT / f"failed_attempt{failed}.json").exists():
            failed += 1
        durable_publish(
            OUT / f"failed_attempt{failed}.json",
            encode({"error": repr(error), "time": time.time(), "exit_code": 1}),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approve-evaluation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", type=int, choices=range(4))
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker, args.resume)
    elif args.approve_evaluation:
        controller(args.resume)
    else:
        raise RuntimeError("Explicit evaluation approval is required")
