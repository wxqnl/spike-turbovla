"""Second-round premise checks on the trained v1 base, zero training.

Per frame (same 2000 frames as the other probes), against the demo action chunk:
  C1 proprioceptive shortcut: swap in the state, or the two camera images, of another frame of the same episode
     (`--offset` picks later, cyclic). Whichever input the action follows is the one the policy relies on.
  C2 visual reading: effective number of visual keys per head query, 1 / sum(p^2) over the Shiftmax weights
     restricted to the 392 visual tokens, and the share of the top visual key.
  C3 instruction use: replace the instruction by the next task of the same suite.
  C4 threshold discontinuity: add uniform pixel noise of +-`--noise` intensity levels (out of 255).
  C5 chunk collapse: within-chunk standard deviation of the predicted vs the demo arm actions, and L1 per horizon.

    python reliance_probe.py <ckpt_routing_probe arguments> --out reliance.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ckpt_routing_probe import build_model, decode  # noqa: E402

VISUAL = 392


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--checkpoint", "--v1-root", "--bert", "--data-root", "--work", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--episodes-per-suite", type=int, default=25)
    p.add_argument("--frames-per-episode", type=int, default=20)
    p.add_argument("--offset", type=int, default=2, help="partner frame = this many picks later (cyclic)")
    p.add_argument("--noise", type=float, default=2.0)
    p.add_argument("--threads", type=int, default=24)
    return p.parse_args()


def episodes(args, stats, chunk):
    import pyarrow.parquet as pq
    section = stats["proprio"] if "proprio" in stats else stats["state"]
    s_mean, s_std = np.asarray(section["mean"], np.float32), np.asarray(section["std"], np.float32)
    a_min, a_max = np.asarray(stats["action"]["min"], np.float32), np.asarray(stats["action"]["max"], np.float32)

    def target(actions, step):
        out = actions[step:step + chunk]
        mask = np.zeros(chunk, np.float32)
        mask[:len(out)] = 1
        out = np.concatenate([out, np.repeat(out[-1:], chunk - len(out), 0)])
        out[:, :6] = np.clip(2 * (out[:, :6] - a_min[:6]) / (a_max[:6] - a_min[:6] + 1e-6) - 1, -1, 1)
        return out, mask

    for suite in sorted(Path(args.data_root).glob("libero_*_no_noops_1.0.0_lerobot")):
        info = json.loads((suite / "meta" / "info.json").read_text())
        rows = [json.loads(line) for line in (suite / "meta" / "tasks.jsonl").read_text().splitlines()]
        tasks = {row["task_index"]: row["task"] for row in rows}
        order = [row["task_index"] for row in rows]
        meta = [json.loads(line) for line in (suite / "meta" / "episodes.jsonl").read_text().splitlines()]
        for e in np.linspace(0, len(meta) - 1, args.episodes_per_suite).round().astype(int):
            index = int(meta[e]["episode_index"])
            fmt = dict(episode_chunk=index // info["chunks_size"], episode_index=index)
            table = pq.read_table(suite / info["data_path"].format(**fmt),
                                  columns=["observation.state", "action", "frame_index", "task_index"])
            rank = np.argsort(np.asarray(table.column("frame_index").to_pylist()), kind="stable")
            states = np.asarray(table.column("observation.state").to_pylist(), np.float32)[rank]
            actions = np.asarray(table.column("action").to_pylist(), np.float32)[rank]
            task = int(np.asarray(table.column("task_index").to_pylist())[rank[0]])
            actions[:, 6] = 1.0 - 2.0 * actions[:, 6]
            picks = np.linspace(0, len(actions) - 1, args.frames_per_episode).round().astype(int)
            primary = decode(suite / info["video_path"].format(video_key="observation.images.image", **fmt), picks)
            wrist = decode(suite / info["video_path"].format(video_key="observation.images.wrist_image", **fmt),
                           picks)
            frames = []
            for step in picks:
                goal, mask = target(actions, step)
                frames.append(dict(primary=primary[step], wrist=wrist[step], target=goal, mask=mask,
                                   state=(states[step] - s_mean) / (s_std + 1e-6)))
            swap = tasks[order[(order.index(task) + 1) % len(order)]]
            for i, frame in enumerate(frames):
                yield dict(suite=suite.name.split("_")[1], instruction=tasks[task], swap_instruction=swap,
                           own=frame, partner=frames[(i + args.offset) % len(frames)])


class VisualReading:
    def __init__(self, model):
        from turbovla.models.shiftmax_action_head import ShiftmaxActionAttention, ShiftmaxRouting
        self.on, self.sums = False, defaultdict(lambda: np.zeros(3))
        self.sites = {}
        for name, module in model.named_modules():
            if isinstance(module, ShiftmaxActionAttention) and name.endswith("cross_attn"):
                self.sites[module.routing] = "L" + name.split("layers.")[1].split(".")[0]
        reader, original = self, ShiftmaxRouting.forward

        def forward(module, scores, allowed):
            weights = original(module, scores, allowed)
            if reader.on and module in reader.sites:
                visual = weights[0, ..., :VISUAL]
                p = visual / visual.sum(-1, keepdim=True).clamp_min(1e-12)
                site = reader.sites[module]
                reader.sums[site] += [(1 / p.square().sum(-1)).sum().item(), p.amax(-1).sum().item(), p[..., 0].numel()]
            return weights

        ShiftmaxRouting.forward = forward


def main():
    args = parse()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    model, checkpoint = build_model(args)
    from turbovla.evaluation.policy import build_sdtv3_manual_processor
    processor = build_sdtv3_manual_processor()
    stats_path = Path(args.v1_root) / "experiments" / "libero" / "configs" / "libero_all4_stats.json"
    stats = json.loads(stats_path.read_text())[checkpoint["args"]["stats_key"]]
    chunk = model.config.action.horizon
    reader = VisualReading(model)
    rng = np.random.default_rng(0)

    def pixels(primary, wrist, noise=0.0):
        images = [primary, wrist]
        if noise:
            images = [np.clip(im.astype(np.float32) + rng.uniform(-noise, noise, im.shape), 0, 255).round()
                      .astype(np.uint8) for im in images]
        return processor(images)["pixel_values"].view(1, 2, 3, 224, 224).float()

    def encode(instruction, images):
        condition, features = model.encode_condition([instruction], {"pixel_values": images},
                                                     return_distillation_features=True)
        valid = features["text_valid"]
        padding = torch.cat((torch.zeros((1, condition.shape[1] - valid.shape[1]), dtype=torch.bool), ~valid), 1)
        return condition, padding

    def act(encoded, state):
        condition, padding = encoded
        return model.action_head(condition, torch.from_numpy(state)[None], memory_key_padding_mask=padding)[0].numpy()

    def l1(a, frame):
        return float((np.abs(a - frame["target"]).mean(-1) * frame["mask"]).sum() / frame["mask"].sum())

    records = defaultdict(lambda: defaultdict(list))
    horizon = defaultdict(lambda: np.zeros((2, chunk)))
    start, count = time.time(), 0
    for sample in episodes(args, stats, chunk):
        own, partner = sample["own"], sample["partner"]
        with torch.inference_mode():
            reader.on = True
            base_memory = encode(sample["instruction"], pixels(own["primary"], own["wrist"]))
            exact = act(base_memory, own["state"])
            reader.on = False
            actions = {
                "state swap": act(base_memory, partner["state"]),
                "vision swap": act(encode(sample["instruction"], pixels(partner["primary"], partner["wrist"])),
                                   own["state"]),
                "instruction swap": act(encode(sample["swap_instruction"], pixels(own["primary"], own["wrist"])),
                                        own["state"]),
                f"pixel noise +-{args.noise:g}": act(
                    encode(sample["instruction"], pixels(own["primary"], own["wrist"], args.noise)), own["state"]),
            }
        valid = own["mask"] > 0
        for key in (sample["suite"], "all"):
            r = records[key]
            r["exact|l1_own"].append(l1(exact, own))
            r["exact|l1_partner"].append(l1(exact, partner))
            for name, a in actions.items():
                r[f"{name}|l1_own"].append(l1(a, own))
                r[f"{name}|l1_partner"].append(l1(a, partner))
                r[f"{name}|shift"].append(float(np.abs(a - exact)[:, :6].mean()))
                r[f"{name}|flip"].append(float((np.sign(a[:, 6]) != np.sign(exact[:, 6])).mean()))
            if valid.sum() >= 4:
                r["chunk std predicted"].append(float(exact[valid, :6].std(0).mean()))
                r["chunk std demo"].append(float(own["target"][valid, :6].std(0).mean()))
            horizon[key][0] += np.abs(exact - own["target"]).mean(-1) * own["mask"]
            horizon[key][1] += own["mask"]
        count += 1
        if count % 100 == 0:
            print(f"[reliance] {count} frames, {(time.time() - start) / count:.2f} s/frame", flush=True)

    summary = dict(frames=count, offset_picks=args.offset, noise_levels=args.noise, results={},
                   visual_reading={site: dict(effective_visual_keys=s[0] / s[2], top_visual_key_share=s[1] / s[2])
                                   for site, s in sorted(reader.sums.items())})
    for key, r in records.items():
        out = {"frames": len(r["exact|l1_own"])}
        base = np.array(r["exact|l1_own"])
        for name in ["exact"] + [n.split("|")[0] for n in r if n.endswith("|shift")]:
            own_l1, partner_l1 = np.array(r[f"{name}|l1_own"]), np.array(r[f"{name}|l1_partner"])
            entry = dict(l1_own=float(own_l1.mean()), l1_partner=float(partner_l1.mean()),
                         follows_partner=float((partner_l1 < own_l1).mean()),
                         delta_own=float((own_l1 - base).mean()),
                         delta_own_se=float((own_l1 - base).std(ddof=1) / np.sqrt(len(base))))
            if name != "exact":
                entry.update(shift=float(np.mean(r[f"{name}|shift"])), gripper_flip=float(np.mean(r[f"{name}|flip"])))
            out[name] = entry
        out["chunk_std_predicted"] = float(np.mean(r["chunk std predicted"]))
        out["chunk_std_demo"] = float(np.mean(r["chunk std demo"]))
        out["l1_by_horizon"] = (horizon[key][0] / np.maximum(horizon[key][1], 1)).round(4).tolist()
        summary["results"][key] = out
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
