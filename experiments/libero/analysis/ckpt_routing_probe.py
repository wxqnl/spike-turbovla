"""Routing statistics of a trained v1 spike-driven-vla checkpoint (premise checks P1-P3 and an open-loop P4 proxy).

The v1 base (`spike_text_only` fusion + Shiftmax action head) routes by hard top-k only in its six spiking text
layers (top-3 over <= 21 instruction tokens). Its action head weights keys by Shiftmax (powers of two), so ties
there share weight instead of being broken by order. For the head we therefore report
  * the Shiftmax mass each memory group receives (what a trained graded head reads), and
  * a counterfactual WTA_8 with the release rule (silent queries/keys excluded) on the head's real Q/K spikes.

Text WTA: ties at the k-th place, winners decided by token order, winner positions under index vs random
tie-breaking, and winner overlap across the T steps. P4 proxy: the action chunk with the text ties broken at
random instead of by order, against the demo chunk.

The model is built from the v1 code (read only); SmoothSpike and SDT-V3 weights come from the policy checkpoint,
so only a BERT-base-uncased tokenizer directory is needed.

    python ckpt_routing_probe.py --checkpoint spike-driven-vla-libero-40k-ema.pth --v1-root /path/to/cts \
        --bert ~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/<rev> \
        --data-root /path/to/libero --work /tmp/probe --out probe.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

GROUPS = ("agent view", "wrist view", "instruction", "state")
JITTER = 0.5  # affinities are integers, so jitter in [0, 0.5) only reorders ties


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--v1-root", required=True, help="checkout of the v1 code (turbovla package)")
    p.add_argument("--bert", required=True, help="bert-base-uncased tokenizer/config directory")
    p.add_argument("--data-root", required=True, help="directory with the four LIBERO LeRobot suites")
    p.add_argument("--episodes-per-suite", type=int, default=25)
    p.add_argument("--frames-per-episode", type=int, default=20)
    p.add_argument("--random-seeds", type=int, default=2, help="random tie-break reruns per frame (P4 proxy)")
    p.add_argument("--head-draws", type=int, default=4, help="random tie-break draws for the counterfactual WTA")
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


# ----------------------------------------------------------------------------------------------- model

def build_model(args):
    sys.path.insert(0, str(Path(args.v1_root).resolve()))
    from transformers import BertConfig
    import turbovla.models.text_encoder as text_encoder
    import turbovla.models.vision_encoder as vision_encoder
    from turbovla.models.configuration import TurboVLAConfig
    from turbovla.models.turbovla import build_turbovla
    from turbovla.models.components.wta_operators import verify_running_norms

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    text_dir = Path(args.work) / "smoothspike_config"
    text_dir.mkdir(parents=True, exist_ok=True)
    config_json = json.loads((Path(args.bert) / "config.json").read_text())
    config_json["T"] = int(checkpoint["model_config"]["text"]["timesteps"])
    (text_dir / "config.json").write_text(json.dumps(config_json))
    for name in ("vocab.txt", "tokenizer.json", "tokenizer_config.json"):
        (text_dir / name).write_bytes((Path(args.bert) / name).read_bytes())

    def smoothspike_skeleton(path, timesteps):
        from third_party.SmoothSpike.spikingbert_rot_inf import BertForMaskedLM
        config = BertConfig.from_pretrained(str(path), local_files_only=True)
        config.T, config._attn_implementation = int(timesteps), "eager"
        return BertForMaskedLM(config).bert  # weights are loaded from the policy checkpoint below

    source = (Path(args.v1_root) / "third_party" / "Spike-Driven-Transformer-V3" / "SDT_V3" / "Classification"
              / "Model_Base" / "models.py")

    def sdtv3_skeleton(self, config):
        spec = importlib.util.spec_from_file_location("probe_sdtv3_models", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Efficient_Spiking_Transformer_l()

    text_encoder.load_smoothspike_model = smoothspike_skeleton
    vision_encoder.SDTV3VisionEncoder._load_official_model = sdtv3_skeleton

    config = TurboVLAConfig.from_mapping(checkpoint["model_config"])
    config.text.model_name_or_path = str(text_dir)
    config.text.local_files_only = True
    model = build_turbovla(config)
    state = {k.removeprefix("module."): v for k, v in checkpoint["ema_model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    verify_running_norms(model)
    return model, checkpoint


# ------------------------------------------------------------------------------------------------ data

def decode(path, wanted):
    import av
    wanted, frames = set(int(i) for i in wanted), {}
    try:
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    frames[index] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(wanted):
                    break
        return frames
    except av.error.FFmpegError:
        # Some AV1 streams fail in PyAV's bundled decoder; system FFmpeg decodes them (as in the v1 loader).
        import subprocess
        with av.open(str(path)) as container:
            width, height = container.streams.video[0].codec_context.width, container.streams.video[0].codec_context.height
        raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
                              "-vsync", "0", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                             capture_output=True, check=True).stdout
        video = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
        return {i: video[i].copy() for i in wanted}


def frames(args, stats, chunk):
    import pyarrow.parquet as pq
    section = stats["proprio"] if "proprio" in stats else stats["state"]
    s_mean, s_std = np.asarray(section["mean"], np.float32), np.asarray(section["std"], np.float32)
    a_min, a_max = np.asarray(stats["action"]["min"], np.float32), np.asarray(stats["action"]["max"], np.float32)
    for suite in sorted(Path(args.data_root).glob("libero_*_no_noops_1.0.0_lerobot")):
        name = suite.name.split("_")[1]
        info = json.loads((suite / "meta" / "info.json").read_text())
        tasks = {}
        for line in (suite / "meta" / "tasks.jsonl").read_text().splitlines():
            row = json.loads(line)
            tasks[row["task_index"]] = row["task"]
        episodes = [json.loads(line) for line in (suite / "meta" / "episodes.jsonl").read_text().splitlines()]
        for e in np.linspace(0, len(episodes) - 1, args.episodes_per_suite).round().astype(int):
            index = int(episodes[e]["episode_index"])
            fmt = dict(episode_chunk=index // info["chunks_size"], episode_index=index)
            table = pq.read_table(suite / info["data_path"].format(**fmt),
                                  columns=["observation.state", "action", "frame_index", "task_index"])
            order = np.argsort(np.asarray(table.column("frame_index").to_pylist()), kind="stable")
            states = np.asarray(table.column("observation.state").to_pylist(), np.float32)[order]
            actions = np.asarray(table.column("action").to_pylist(), np.float32)[order]
            task = int(np.asarray(table.column("task_index").to_pylist())[order[0]])
            actions[:, 6] = 1.0 - 2.0 * actions[:, 6]  # binary_open (1 open) -> signed close (+1 close)
            picks = np.linspace(0, len(actions) - 1, args.frames_per_episode).round().astype(int)
            # Object geometry from the demo: end-effector position at the first grasp and the following release.
            closed = actions[:, 6] > 0
            grasp_step = int(np.argmax(closed)) if closed.any() else None
            after = np.where(~closed[grasp_step:])[0] if grasp_step is not None else []
            place_step = grasp_step + int(after[0]) if len(after) else None
            primary = decode(suite / info["video_path"].format(video_key="observation.images.image", **fmt), picks)
            wrist = decode(suite / info["video_path"].format(video_key="observation.images.wrist_image", **fmt),
                           picks)
            for step in picks:
                target = actions[step:step + chunk]
                mask = np.zeros(chunk, np.float32)
                mask[:len(target)] = 1
                target = np.concatenate([target, np.repeat(target[-1:], chunk - len(target), 0)])
                target[:, :6] = np.clip(2 * (target[:, :6] - a_min[:6]) / (a_max[:6] - a_min[:6] + 1e-6) - 1, -1, 1)
                yield dict(suite=name, episode=f"{name}/{index}", instruction=tasks[task], primary=primary[step],
                           wrist=wrist[step],
                           state=(states[step] - s_mean) / (s_std + 1e-6), target=target, mask=mask,
                           progress=step / max(len(actions) - 1, 1), step=int(step),
                           grasp=None if grasp_step is None else (grasp_step, states[grasp_step, :3]),
                           place=None if place_step is None else (place_step, states[place_step, :3]))


# ------------------------------------------------------------------------------------------- statistics

def topk_mask(scores, allowed, k, jitter=None):
    ranked = scores.masked_fill(~allowed, -torch.inf)
    if jitter is not None:
        ranked = ranked + jitter
    index = torch.argsort(ranked, dim=-1, descending=True, stable=True)[..., :k]
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, index, True) & allowed


def tie_stats(scores, allowed, k):
    """Per-row counts for the k-th place of a top-k over integer affinities. Inputs [..., Nk]."""
    ranked = scores.masked_fill(~allowed, -torch.inf).sort(-1, descending=True).values
    live, candidates = allowed.any(-1), allowed.sum(-1)
    selecting = live & (candidates > k)
    kth, nxt = ranked[..., k - 1], ranked[..., min(k, ranked.shape[-1] - 1)]
    tied = selecting & (kth == nxt)
    above = ((scores > kth[..., None]) & allowed).sum(-1)
    group = ((scores == kth[..., None]) & allowed).sum(-1)
    order_decided = torch.where(tied, k - above, torch.zeros_like(above))
    winners = torch.where(live, candidates.clamp(max=k), torch.zeros_like(candidates))
    distinct = torch.tensor([len(torch.unique(row[a])) for row, a in zip(scores.reshape(-1, scores.shape[-1]),
                                                                        allowed.reshape(-1, allowed.shape[-1]))
                             if a.any()], dtype=torch.float32)
    return dict(rows=int(live.sum()), selecting=int(selecting.sum()), tied=int(tied.sum()),
                winners=int(winners.sum()), order_decided=int(order_decided.sum()),
                tie_group=float(torch.where(tied, group, 0).sum()), candidates=float(candidates[live].sum()),
                distinct=float(distinct.sum()))


class Probe:
    def __init__(self, model, head_draws):
        from turbovla.models.components.wta_operators import MultiWinnerAttention
        from turbovla.models.shiftmax_action_head import ShiftmaxActionAttention, ShiftmaxRouting
        self.mode, self.generator, self.head_draws = "index", torch.Generator().manual_seed(0), head_draws
        self.recording, self.suite = True, ""
        self.text = defaultdict(lambda: defaultdict(float))
        self.text_positions = defaultdict(lambda: np.zeros((2, 21)))
        # Over rows that select (more candidates than k): Jaccard sum, step pairs, rows constant over T, rows selecting at all T
        self.overlap = defaultdict(lambda: np.zeros(4))
        self.head = defaultdict(lambda: defaultdict(float))
        self.head_mass = defaultdict(lambda: np.zeros(4))
        self.head_share = defaultdict(lambda: np.zeros((2, 4)))
        self.head_silent = defaultdict(lambda: np.zeros((2, 4)))
        self.head_fair = defaultdict(lambda: np.zeros(4))
        self.sample_winners = defaultdict(list)
        self.names, self.attention = {}, {}
        for name, module in model.named_modules():
            if isinstance(module, MultiWinnerAttention):
                self.names[module] = "text L" + name.split("text_layers.")[1].split(".")[0]
            if isinstance(module, ShiftmaxActionAttention) and name.endswith("cross_attn"):
                self.names[module.routing] = "head cross L" + name.split("layers.")[1].split(".")[0]
                self.attention[module.routing] = module
                for node in ("query", "key"):
                    module.spike_nodes[node].register_forward_hook(self._stash(module, node))
        probe = self
        original_wta, original_shiftmax = MultiWinnerAttention.forward, ShiftmaxRouting.forward

        def wta_forward(module, scores, allowed, query_spikes, key_spikes):
            winners = original_wta(module, scores, allowed, query_spikes, key_spikes)
            if probe.mode == "index":
                if probe.recording:
                    probe.record_text(module, scores, allowed, query_spikes, key_spikes, winners)
                return winners
            mask = allowed[:, None] if allowed.ndim == 3 else allowed
            mask = mask.expand_as(scores) & query_spikes.ne(0).any(-1, keepdim=True)
            mask = mask & key_spikes.ne(0).any(-1).unsqueeze(-2)
            jitter = torch.rand(scores.shape, generator=probe.generator) * JITTER
            return topk_mask(scores, mask, min(module.topk, scores.shape[-1]), jitter).to(scores.dtype)

        def shiftmax_forward(module, scores, allowed):
            weights = original_shiftmax(module, scores, allowed)
            if probe.mode == "index" and probe.recording and module in probe.names:
                probe.record_head(module, scores, allowed, weights)
            return weights

        MultiWinnerAttention.forward = wta_forward
        ShiftmaxRouting.forward = shiftmax_forward

    @staticmethod
    def _stash(module, node):
        def hook(_, __, output):
            setattr(module, f"_probe_{node}", output[0].detach())
        return hook

    def record_text(self, module, scores, allowed, q, k, winners):
        site = self.names[module]
        assert torch.equal(scores, scores.round()), "spike affinities must be integers"
        mask = allowed[:, None] if allowed.ndim == 3 else allowed
        mask = mask.expand_as(scores) & q.ne(0).any(-1, keepdim=True) & k.ne(0).any(-1).unsqueeze(-2)
        kk = min(module.topk, scores.shape[-1])
        for key, value in tie_stats(scores[0], mask[0], kk).items():
            for pooled in (site, "all " + self.suite):
                self.text[pooled][key] += value
        self.text_positions[site][0, :scores.shape[-1]] += winners[0].sum((0, 1)).numpy()
        jitter = torch.rand(scores.shape, generator=self.generator) * JITTER
        self.text_positions[site][1, :scores.shape[-1]] += topk_mask(scores, mask, kk, jitter)[0].sum((0, 1)).numpy()
        self.sample_winners[site].append((winners[0] > 0, mask[0].sum(-1) > kk))

    def record_head(self, routing, scores, allowed, weights):
        site, attention = self.names[routing], self.attention[routing]
        batch, heads, nq, nk = scores.shape
        q = attention._probe_query.reshape(batch, nq, heads, -1).transpose(1, 2)
        k = attention._probe_key.reshape(batch, nk, heads, -1).transpose(1, 2)
        valid = allowed[:, None] if allowed.ndim == 3 else allowed
        valid = valid.expand_as(scores)
        groups = self.groups(valid[0, 0, 0])
        fair = np.array([g.sum().item() for g in groups])
        # Shiftmax mass per group, rows renormalized to 1.
        w = weights[0] / weights[0].sum(-1, keepdim=True).clamp_min(1e-12)
        mass = np.array([w[..., g].sum().item() for g in groups])
        # Counterfactual WTA_8 with the release rule.
        mask = valid & q.ne(0).any(-1, keepdim=True) & k.ne(0).any(-1).unsqueeze(-2)
        ties = tie_stats(scores[0], mask[0], 8)
        index = topk_mask(scores[0], mask[0], 8)
        share = np.zeros((2, 4))
        share[0] = [index[..., g].sum().item() for g in groups]
        for _ in range(self.head_draws):
            jitter = torch.rand(scores[0].shape, generator=self.generator) * JITTER
            chosen = topk_mask(scores[0], mask[0], 8, jitter)
            share[1] += np.array([chosen[..., g].sum().item() for g in groups]) / self.head_draws
        live_key = k[0].ne(0).any(-1)  # [H, Nk]
        silent = np.array([[(~live_key[:, g]).sum().item() for g in groups], [live_key[:, g].numel() for g in groups]])
        for pooled in (site, "all " + self.suite):
            self.head_fair[pooled] += fair
            self.head_mass[pooled] += mass
            self.head[pooled]["mass_rows"] += heads * nq
            for key, value in ties.items():
                self.head[pooled][key] += value
            self.head_share[pooled] += share
            self.head_silent[pooled] += silent
        self.sample_winners[site].append((index, mask[0].sum(-1) > 8))

    @staticmethod
    def groups(valid_keys):
        n = valid_keys.numel()
        state, vision = 8, 392
        text = torch.zeros(n, dtype=torch.bool)
        text[vision:n - state] = valid_keys[vision:n - state]
        return [torch.arange(n) < 196, (torch.arange(n) >= 196) & (torch.arange(n) < vision), text,
                torch.arange(n) >= n - state]

    def end_sample(self):
        for site, calls in self.sample_winners.items():
            for (a, live_a), (b, live_b) in zip(calls[:-1], calls[1:]):
                rows = live_a & live_b  # rows that select at both steps
                inter = (a & b).sum(-1)[rows].float()
                union = (a | b).sum(-1)[rows].float().clamp_min(1)
                self.overlap[site][0] += (inter / union).sum().item()
                self.overlap[site][1] += rows.sum().item()
            live = torch.stack([live for _, live in calls]).all(0)
            same = torch.stack([(w == calls[0][0]).all(-1) for w, _ in calls]).all(0)
            self.overlap[site][2] += (same & live).sum().item()
            self.overlap[site][3] += live.sum().item()
        self.sample_winners.clear()


# ---------------------------------------------------------------------------------------------- driver

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
    probe = Probe(model, args.head_draws)
    open_loop = defaultdict(lambda: defaultdict(list))
    start, count = time.time(), 0
    for sample in frames(args, stats, chunk):
        pixels = processor([sample["primary"], sample["wrist"]])["pixel_values"].view(1, 2, 3, 224, 224).float()
        state = torch.from_numpy(sample["state"])[None]
        target, mask = sample["target"], sample["mask"]
        probe.suite = sample["suite"]
        with torch.inference_mode():
            probe.mode, probe.recording = "index", True
            reference = model([sample["instruction"]], {"pixel_values": pixels}, state)[0].numpy()
            probe.end_sample()
            randomized = []
            probe.mode = "random"
            for seed in range(args.random_seeds):
                probe.generator.manual_seed(10_000 * count + seed)
                randomized.append(model([sample["instruction"]], {"pixel_values": pixels}, state)[0].numpy())
            probe.mode = "index"
        l1 = lambda a: float((np.abs(a - target).mean(-1) * mask).sum() / mask.sum())  # noqa: E731
        record = open_loop[sample["suite"]]
        record["l1_index"].append(l1(reference))
        record["l1_random"].append(np.mean([l1(a) for a in randomized]))
        record["shift"].append(np.mean([np.abs(a - reference)[:, :6].mean() for a in randomized]))
        record["gripper_flip"].append(np.mean([(np.sign(a[:, 6]) != np.sign(reference[:, 6])).mean()
                                               for a in randomized]))
        count += 1
        if count % 50 == 0:
            print(f"[probe] {count} frames, {(time.time() - start) / count:.2f} s/frame", flush=True)

    summary = dict(checkpoint=args.checkpoint, frames=count, text={}, head={}, open_loop={})
    for site, s in sorted(probe.text.items()):
        positions = probe.text_positions.get(site, np.ones((2, 21)))
        jaccard, pairs, constant, all_live = probe.overlap.get(site, np.zeros(4))
        summary["text"][site] = dict(
            rows=s["rows"], selecting_rows=s["selecting"] / s["rows"], tied_of_selecting=s["tied"] / max(s["selecting"], 1),
            order_decided_winners=s["order_decided"] / s["winners"], mean_candidates=s["candidates"] / s["rows"],
            mean_distinct_scores=s["distinct"] / s["rows"], mean_tie_group=s["tie_group"] / max(s["tied"], 1),
            consecutive_jaccard=jaccard / max(pairs, 1), rows_constant_over_T=constant / max(all_live, 1),
            winner_position_index=(positions[0] / positions[0].sum()).round(4).tolist(),
            winner_position_random=(positions[1] / positions[1].sum()).round(4).tolist())
    for site, s in sorted(probe.head.items()):
        mass = probe.head_mass[site] / s["mass_rows"]
        share = probe.head_share[site] / probe.head_share[site].sum(1, keepdims=True)
        silent = probe.head_silent[site][0] / probe.head_silent[site][1]
        summary["head"][site] = dict(
            fair_share=dict(zip(GROUPS, (probe.head_fair[site] / probe.head_fair[site].sum()).round(4).tolist())),
            shiftmax_mass=dict(zip(GROUPS, mass.round(4).tolist())),
            wta8_selecting_rows=s["selecting"] / max(s["rows"], 1), wta8_tied_of_selecting=s["tied"] / max(s["selecting"], 1),
            wta8_order_decided_winners=s["order_decided"] / max(s["winners"], 1),
            wta8_mean_candidates=s["candidates"] / max(s["rows"], 1), wta8_mean_distinct=s["distinct"] / max(s["rows"], 1),
            wta8_share_index=dict(zip(GROUPS, share[0].round(4).tolist())),
            wta8_share_random=dict(zip(GROUPS, share[1].round(4).tolist())),
            silent_key_fraction=dict(zip(GROUPS, silent.round(4).tolist())))
        if site in probe.overlap:
            jaccard, pairs, constant, all_live = probe.overlap[site]
            summary["head"][site].update(wta8_consecutive_jaccard=jaccard / max(pairs, 1),
                                         wta8_rows_constant_over_T=constant / max(all_live, 1))
    for suite, record in sorted(open_loop.items()):
        summary["open_loop"][suite] = {key: float(np.mean(value)) for key, value in record.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
