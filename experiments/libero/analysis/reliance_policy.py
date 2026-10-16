"""State vs vision reliance for any checkpoint the LIBERO evaluator loads: spiking base and TurboVLA.

Same frames as reliance_probe.py (25 demos per suite, 20 evenly spaced frames each, partner = the frame `--offset`
picks later in the same demo, cyclic). For each frame the policy acts on (own images, own state), (own images,
partner state) and (partner images, own state); whichever input the action follows (smaller arm-action L1 to the
partner's demo chunk than to the own chunk) is the one it relies on. Frames are also split by phase: "holding" when
the demo's gripper command at that frame is close (transport and placement), else "free". Also reported: how far
each swap moves the predicted arm actions (mean |change|) next to the gap between the two demo chunks.

The policy is built by the evaluator itself (vla_adapter.rollout.eval_libero, interrupted right after the policy is
constructed), so preprocessing, normalization and weights are exactly those of the LIBERO runs. Arguments: the usual
evaluator arguments, plus --data_root, --out [--episodes_per_suite 25 --frames_per_episode 20 --offset 2].
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ckpt_routing_probe import decode  # noqa: E402
from vla_adapter import rollout  # noqa: E402


class Captured(Exception):
    pass


def build_policy():
    original = rollout._import_turbovla_adapter

    def patched():
        policy_cls, dummy, rotate, seed = original()

        class Grab(policy_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                raise Captured(self)
        return Grab, dummy, rotate, seed

    rollout._import_turbovla_adapter = patched
    try:
        rollout.eval_libero(rollout.parse_args())
    except Captured as c:
        return c.args[0]
    raise RuntimeError("policy was not constructed")


def demo_frames(data_root, stats, chunk, per_suite, per_episode, offset):
    import pyarrow.parquet as pq
    a_min, a_max = np.asarray(stats["action"]["min"], np.float32), np.asarray(stats["action"]["max"], np.float32)

    def target(actions, step):
        out = actions[step:step + chunk]
        mask = np.zeros(chunk, np.float32)
        mask[:len(out)] = 1
        out = np.concatenate([out, np.repeat(out[-1:], chunk - len(out), 0)])
        return np.clip(2 * (out[:, :6] - a_min[:6]) / (a_max[:6] - a_min[:6] + 1e-6) - 1, -1, 1), mask

    for suite in sorted(Path(data_root).glob("libero_*_no_noops_1.0.0_lerobot")):
        info = json.loads((suite / "meta" / "info.json").read_text())
        rows = [json.loads(line) for line in (suite / "meta" / "tasks.jsonl").read_text().splitlines()]
        tasks = {row["task_index"]: row["task"] for row in rows}
        meta = [json.loads(line) for line in (suite / "meta" / "episodes.jsonl").read_text().splitlines()]
        for e in np.linspace(0, len(meta) - 1, per_suite).round().astype(int):
            index = int(meta[e]["episode_index"])
            fmt = dict(episode_chunk=index // info["chunks_size"], episode_index=index)
            table = pq.read_table(suite / info["data_path"].format(**fmt),
                                  columns=["observation.state", "action", "frame_index", "task_index"])
            rank = np.argsort(np.asarray(table.column("frame_index").to_pylist()), kind="stable")
            states = np.asarray(table.column("observation.state").to_pylist(), np.float32)[rank]
            actions = np.asarray(table.column("action").to_pylist(), np.float32)[rank]
            task = int(np.asarray(table.column("task_index").to_pylist())[rank[0]])
            picks = np.linspace(0, len(actions) - 1, per_episode).round().astype(int)
            primary = decode(suite / info["video_path"].format(video_key="observation.images.image", **fmt), picks)
            wrist = decode(suite / info["video_path"].format(video_key="observation.images.wrist_image", **fmt),
                           picks)
            frames = []
            for step in picks:
                goal, mask = target(actions, step)
                frames.append(dict(primary=primary[step], wrist=wrist[step], target=goal, mask=mask,
                                   state=states[step], holding=bool(actions[step, 6] < 0.5)))
            for i, frame in enumerate(frames):
                yield suite.name.split("_")[1], tasks[task], frame, frames[(i + offset) % len(frames)]


def main():
    argv = sys.argv[1:]
    opts = {"--data_root": None, "--out": None, "--episodes_per_suite": "25", "--frames_per_episode": "20",
            "--offset": "2"}
    for flag in list(opts):
        if flag in argv:
            i = argv.index(flag)
            opts[flag] = argv[i + 1]
            del argv[i:i + 2]
    sys.argv = [sys.argv[0], *argv]
    policy = build_policy()
    model = policy.model
    stats = json.loads(Path(policy.stats_path).read_text())[policy.stats_key]
    chunk = model.config.action.horizon

    def l1(a, frame):
        return float((np.abs(a[:, :6] - frame["target"]).mean(-1) * frame["mask"]).sum() / frame["mask"].sum())

    records = defaultdict(lambda: defaultdict(list))
    count = 0
    for suite, instruction, own, partner in demo_frames(opts["--data_root"], stats, chunk,
                                                        int(opts["--episodes_per_suite"]),
                                                        int(opts["--frames_per_episode"]), int(opts["--offset"])):
        samples, states = policy._build_batch([own["primary"], own["primary"], partner["primary"]],
                                              [own["wrist"], own["wrist"], partner["wrist"]],
                                              [own["state"], partner["state"], own["state"]])
        samples, states = policy._prepare_model_inputs(samples, states)
        with torch.inference_mode():
            pred = model([instruction] * 3, samples, states).float().cpu().numpy()
        exact, state_swap, vision_swap = pred
        phase = "holding" if own["holding"] else "free"
        for key in (suite, f"{suite}/{phase}", "all", f"all/{phase}"):
            r = records[key]
            for name, a in (("exact", exact), ("state swap", state_swap), ("vision swap", vision_swap)):
                r[f"{name}|own"].append(l1(a, own))
                r[f"{name}|partner"].append(l1(a, partner))
                r[f"{name}|shift"].append(float(np.abs(a[:, :6] - exact[:, :6]).mean()))
            # how far the two demo chunks are apart: the scale a full switch to the partner would move the action
            r["demo gap"].append(float((np.abs(own["target"] - partner["target"]).mean(-1) * own["mask"]).sum()
                                       / own["mask"].sum()))
        count += 1
        if count % 200 == 0:
            print(f"[reliance] {count} frames", flush=True)
    out = {"frames": count, "checkpoint": policy.ckpt_path, "results": {}}
    for key, r in sorted(records.items()):
        entry = {"frames": len(r["exact|own"])}
        for name in ("exact", "state swap", "vision swap"):
            own_l1, partner_l1 = np.array(r[f"{name}|own"]), np.array(r[f"{name}|partner"])
            entry[name] = dict(l1_own=float(own_l1.mean()), l1_partner=float(partner_l1.mean()),
                               follows_partner=float((partner_l1 < own_l1).mean()),
                               shift=float(np.mean(r[f"{name}|shift"])))
        entry["demo_gap"] = float(np.mean(r["demo gap"]))
        out["results"][key] = entry
        print(f"{key:18s} n={entry['frames']:4d}  exact L1 {entry['exact']['l1_own']:.4f}  "
              f"follows state {entry['state swap']['follows_partner']:.2f}  "
              f"follows vision {entry['vision swap']['follows_partner']:.2f}  shift state {entry['state swap']['shift']:.4f}"
              f" vision {entry['vision swap']['shift']:.4f}  demo gap {entry['demo_gap']:.4f}", flush=True)
    Path(opts["--out"]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
