"""Crash-safe completed-episode storage; no model, simulator or CUDA imports.

The caller supplies a fixed, approved manifest. This module neither chooses a
budget/seed list nor grants evaluation permission. Use a separate directory for
each policy and hold this lock for the entire worker/controller lifetime.
Immutable episode receipts are authoritative; events/result are derived views.
"""
import fcntl
import json
import math
import os
import tempfile
from pathlib import Path


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       allow_nan=False) + "\n").encode()


def durable_publish(path, data, *, replace=False):
    """Publish a fully fsynced file; never overwrite a completed receipt."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # EEXIST is an error, not permission to replace.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class EpisodeStore:
    """Single-writer, manifest-bound storage for a policy's full MT50 results.

    manifest: {"protocol": {...}, "episodes": [
        {"native_task_id": int, "episode": int, "reset_seed": int,
         "env_name": str}, ...]}
    Each of all 50 native tasks must have the same positive episode budget,
    episode IDs 0..N-1 and distinct reset seeds. No default budget is supplied.
    provenance must identify variant, source, EMA, policy_seed and numerical
    settings. A different manifest/provenance cannot resume this directory.
    """
    def __init__(self, directory, manifest, provenance, *, resume=False, task_ids=None,
                 expected_variants=("spike_driven_vla_ema70k",), expected_global_step=70000,
                 expected_samples_seen=17920000):
        self.directory = Path(directory)
        self.expected_variants = tuple(expected_variants)
        self.expected_global_step = int(expected_global_step)
        self.expected_samples_seen = int(expected_samples_seen)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "results.lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._open(manifest, provenance, resume, task_ids)
        except BaseException:
            self.close()
            raise

    def _open(self, manifest, provenance, resume, task_ids):
        if not isinstance(manifest.get("protocol"), dict) or not manifest["protocol"]:
            raise ValueError("Explicit fixed protocol required")
        required = {"variant", "source", "policy_seed", "global_step",
                    "samples_seen", "strict_load", "precision", "matmul",
                    "cuda_tf32", "cudnn_tf32"}
        if not isinstance(provenance, dict) or not required.issubset(provenance):
            raise ValueError("Complete policy provenance required")
        weight_fields = [k for k in ("ema", "checkpoint")
                         if isinstance(provenance.get(k), str) and provenance[k]]
        if len(weight_fields) != 1:
            raise ValueError("Exactly one EMA or raw checkpoint provenance path required")
        if (provenance["variant"] not in self.expected_variants
                or provenance["global_step"] != self.expected_global_step
                or provenance["samples_seen"] != self.expected_samples_seen
                or provenance["strict_load"] is not True
                or provenance["precision"] != "fp32"
                or provenance["matmul"] != "highest"
                or provenance["cuda_tf32"] is not False
                or provenance["cudnn_tf32"] is not False):
            raise ValueError("Unexpected policy training/numerical protocol")
        self.expected = {}
        by_task = {}
        seeds = set()
        for item in manifest["episodes"]:
            for field in ("native_task_id", "episode", "reset_seed"):
                if type(item[field]) is not int or item[field] < 0:
                    raise ValueError("Episode IDs and seeds must be nonnegative integers")
            if not isinstance(item["env_name"], str) or not item["env_name"]:
                raise ValueError("Native environment name required")
            key = (item["native_task_id"], item["episode"])
            if key in self.expected or item["reset_seed"] in seeds:
                raise ValueError("Duplicate episode or reset seed")
            seeds.add(item["reset_seed"])
            self.expected[key] = dict(item)
            by_task.setdefault(key[0], []).append(key[1])
        if set(by_task) != set(range(50)):
            raise ValueError("All 50 native task IDs required")
        count = len(by_task[0])
        if not count or any(sorted(ids) != list(range(count)) for ids in by_task.values()):
            raise ValueError("Equal per-task budgets and contiguous episode IDs required")
        binding = {"manifest": manifest, "provenance": provenance}
        if task_ids is not None:
            if (not task_ids or len(set(task_ids)) != len(task_ids)
                    or any(type(t) is not int or t not in range(50) for t in task_ids)):
                raise ValueError("Invalid fixed worker task assignment")
            binding["task_ids"] = sorted(task_ids)
            self.expected = {k:v for k,v in self.expected.items() if k[0] in task_ids}
        binding_path = self.directory / "binding.json"
        if binding_path.exists():
            if not resume:
                raise FileExistsError("Existing state requires explicit resume")
            if json.loads(binding_path.read_text()) != binding:
                raise ValueError("Manifest or policy provenance changed")
        else:
            if resume:
                raise FileNotFoundError("Cannot resume without binding receipt")
            if any(p.name != "results.lock" for p in self.directory.iterdir()):
                raise FileExistsError("Unbound existing artifacts; preserve and diagnose")
            durable_publish(binding_path, encode(binding))
        self.binding = binding
        self.receipts = self.directory / "episodes"
        self.receipts.mkdir(exist_ok=True)
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.completed = {}
        for path in sorted(self.receipts.iterdir()):
            if path.name.startswith(".pending-"):
                # Interrupted unpublished temporary file is kept for diagnosis.
                continue
            record = json.loads(path.read_text())
            key = self.validate(record)
            if path.name != self.filename(key) or key in self.completed:
                raise ValueError("Unexpected or duplicate completed receipt")
            self.completed[key] = record
        self.reconcile()

    @staticmethod
    def filename(key):
        return "task%02d_episode%06d.json" % key

    def validate(self, record):
        for field in ("native_task_id", "episode", "reset_seed"):
            if type(record[field]) is not int:
                raise ValueError("Result episode identity must use integers")
        key = (record["native_task_id"], record["episode"])
        expected = self.expected.get(key)
        if expected is None or any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("Result does not match the fixed episode manifest")
        if type(record["success"]) is not bool:
            raise ValueError("Success must be an actual boolean")
        if type(record["steps"]) is not int or not 1 <= record["steps"] <= 500:
            raise ValueError("Invalid native step count")
        initial = record["initial"]
        for field in ("random_vector", "target", "state4"):
            values = initial[field]
            if not isinstance(values, list) or not values:
                raise ValueError("Missing actual initial state")
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
                raise ValueError("Invalid initial state values")
        if len(initial["state4"]) != 4 or len(initial["target"]) != 3:
            raise ValueError("Wrong initial state dimensions")
        encode(record)  # Reject non-finite metadata anywhere, before committing.
        return key

    def pending(self):
        return [dict(self.expected[k]) for k in sorted(self.expected)
                if k not in self.completed]

    def commit(self, record):
        """Call only after rollout_episode returns a completed legal episode."""
        if self.lock.closed:
            raise RuntimeError("Episode store is closed")
        key = self.validate(record)
        if key in self.completed:
            raise FileExistsError("Completed episodes, including failures, cannot be rerun")
        durable_publish(self.receipts / self.filename(key), encode(record))
        self.completed[key] = dict(record)
        self.reconcile()

    def reconcile(self):
        # An interrupted derived-view write never invalidates a committed episode.
        # On explicit resume, rebuild views from immutable authoritative receipts.
        records = [self.completed[k] for k in sorted(self.completed)]
        durable_publish(self.directory / "events.jsonl",
                        b"".join(encode(r) for r in records), replace=True)
        result = {"completed_episodes": len(records),
                  "total_episodes": len(self.expected),
                  "successes": sum(r["success"] for r in records),
                  "complete": len(records) == len(self.expected),
                  "counts_by_task": {
                      str(t): sum(r["native_task_id"] == t for r in records)
                      for t in range(50)}}
        # This is a count view, NOT an evaluator exit receipt or final audit.
        durable_publish(self.directory / "result.json", encode(result), replace=True)

    def close(self):
        if hasattr(self, "lock") and not self.lock.closed:
            self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
