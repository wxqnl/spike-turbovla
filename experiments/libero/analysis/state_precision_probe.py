"""How much does the trained v1 base lose when its robot state is coded like the release's spiking state encoder?

The release `StateEncoder` sends each normalized state coordinate through ONE PLIF neuron (ternary events, hard
reset; decay and threshold are scalars shared by all coordinates) for T = 4 steps. Whatever the network does
afterwards, it can recover at most E[z | spike train] per coordinate. We fit the shared decay/threshold that
minimizes that error on the LIBERO training states (an optimistic bound for the release), replace the exact
state of the trained v1 base (continuous state path) by this best-case decoding, and measure open-loop action
error against the demonstrations on the same frames as `ckpt_routing_probe.py`.

Variants: exact state; PLIF code on all / position / rotation / gripper coordinates; Gaussian noise with the
same per-coordinate RMSE; the v1 population code (32 thresholds x T = 4 over [-4, 4] std) as the fix reference.

    python state_precision_probe.py <ckpt_routing_probe arguments> --out state_probe.json
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
from ckpt_routing_probe import build_model, frames  # noqa: E402

T = 4
SUBSETS = {"position": [0, 1, 2], "rotation": [3, 4, 5], "gripper": [6, 7]}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--checkpoint", "--v1-root", "--bert", "--data-root", "--work", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--episodes-per-suite", type=int, default=25)
    p.add_argument("--frames-per-episode", type=int, default=20)
    p.add_argument("--fit-frames", type=int, default=100_000)
    p.add_argument("--threads", type=int, default=24)
    return p.parse_args()


def plif_codes(z, decay, threshold):
    """Ternary PLIF spike trains of constant inputs z [..., 8], encoded as base-3 integers."""
    u, code = np.zeros_like(z), np.zeros(z.shape, np.int64)
    for t in range(T):
        v = u + decay * (z - u)
        s = (v >= threshold).astype(np.int64) - (v <= -threshold).astype(np.int64)
        u = v * (1 - np.abs(s))
        code += (s + 1) * 3 ** t
    return code


def fit_plif(z):
    """Shared (decay, threshold) minimizing the conditional-mean decoding error, plus per-coordinate decoders."""
    best = None
    for decay in (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999):
        for threshold in np.geomspace(0.002, 3.0, 100):
            code = plif_codes(z, decay, threshold)
            table = np.zeros((z.shape[1], 3 ** T))
            for c in range(z.shape[1]):
                count = np.bincount(code[:, c], minlength=3 ** T)
                table[c] = np.bincount(code[:, c], weights=z[:, c], minlength=3 ** T) / np.maximum(count, 1)
            decoded = table[np.arange(z.shape[1]), code]
            mse = ((decoded - z) ** 2).mean(0)
            if best is None or mse.mean() < best[0].mean():
                best = (mse, decay, float(threshold), table)
    mse, decay, threshold, table = best
    used = [len(np.unique(plif_codes(z, decay, threshold)[:, c])) for c in range(z.shape[1])]
    return dict(decay=decay, threshold=threshold, table=table, rmse=np.sqrt(mse), trains_used=used)


def training_states(data_root, mean, std):
    import pyarrow.parquet as pq
    rows = []
    for suite in sorted(Path(data_root).glob("libero_*_no_noops_1.0.0_lerobot")):
        for path in sorted((suite / "data").glob("chunk-*/episode_*.parquet")):
            rows.append(np.asarray(pq.read_table(path, columns=["observation.state"]).column(0).to_pylist(),
                                   np.float32))
    return (np.concatenate(rows) - mean) / (std + 1e-6)


def main():
    args = parse()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    model, checkpoint = build_model(args)
    from turbovla.evaluation.policy import build_sdtv3_manual_processor
    processor = build_sdtv3_manual_processor()
    stats_path = Path(args.v1_root) / "experiments" / "libero" / "configs" / "libero_all4_stats.json"
    stats = json.loads(stats_path.read_text())[checkpoint["args"]["stats_key"]]
    section = stats["proprio"] if "proprio" in stats else stats["state"]
    s_mean, s_std = np.asarray(section["mean"], np.float32), np.asarray(section["std"], np.float32)

    z_all = training_states(args.data_root, s_mean, s_std)
    rng = np.random.default_rng(0)
    fit = fit_plif(z_all[rng.choice(len(z_all), min(args.fit_frames, len(z_all)), replace=False)])
    print(f"[state] {len(z_all)} training states; PLIF decay {fit['decay']}, threshold {fit['threshold']:.3f}, "
          f"trains used {fit['trains_used']}, RMSE (std) {np.round(fit['rmse'], 3).tolist()}", flush=True)
    population = 8.0 / (32 * T)  # v1 multithreshold: 32 x T interleaved boundaries over [-4, 4]

    def variants(z):
        decoded = fit["table"][np.arange(8), plif_codes(z[None], fit["decay"], fit["threshold"])[0]]
        out = {"exact": z, "PLIF all": decoded}
        for name, dims in SUBSETS.items():
            partial = z.copy()
            partial[dims] = decoded[dims]
            out[f"PLIF {name}"] = partial
        out["noise, PLIF RMSE"] = z + rng.standard_normal(8) * fit["rmse"]
        clipped = np.clip(z, -4 + 1e-6, 4 - 1e-6)
        out["population (v1)"] = (np.floor((clipped + 4) / population) + 0.5) * population - 4
        return out

    records = defaultdict(lambda: defaultdict(list))
    start, count = time.time(), 0
    for sample in frames(args, stats, model.config.action.horizon):
        pixels = processor([sample["primary"], sample["wrist"]])["pixel_values"].view(1, 2, 3, 224, 224).float()
        target, mask = sample["target"], sample["mask"]
        transition = bool((np.sign(target[mask > 0, 6]) != np.sign(target[0, 6])).any())
        with torch.inference_mode():
            condition, features = model.encode_condition([sample["instruction"]], {"pixel_values": pixels},
                                                         return_distillation_features=True)
            text_valid = features["text_valid"]
            padding = torch.cat((torch.zeros((1, condition.shape[1] - text_valid.shape[1]), dtype=torch.bool),
                                 ~text_valid), dim=1)
            actions = {name: model.action_head(condition, torch.from_numpy(z.astype(np.float32))[None],
                                               memory_key_padding_mask=padding)[0].numpy()
                       for name, z in variants(sample["state"]).items()}
            if count == 0:
                full = model([sample["instruction"]], {"pixel_values": pixels},
                             torch.from_numpy(sample["state"])[None])[0].numpy()
                assert np.allclose(full, actions["exact"], atol=1e-5), "head-only path must match the full forward"
        exact = actions["exact"]
        for name, a in actions.items():
            l1 = float((np.abs(a - target).mean(-1) * mask).sum() / mask.sum())
            for key in (sample["suite"], "all") + (("gripper-transition chunks",) if transition else ()):
                records[key][f"{name}|l1"].append(l1)
                records[key][f"{name}|shift"].append(float(np.abs(a - exact)[:, :6].mean()))
                records[key][f"{name}|flip"].append(float((np.sign(a[:, 6]) != np.sign(exact[:, 6])).mean()))
        count += 1
        if count % 100 == 0:
            print(f"[state] {count} frames, {(time.time() - start) / count:.2f} s/frame", flush=True)

    summary = dict(frames=count, plif=dict(decay=fit["decay"], threshold=fit["threshold"],
                                           trains_used=fit["trains_used"], rmse_std=fit["rmse"].round(4).tolist(),
                                           rmse_physical=(fit["rmse"] * s_std).round(5).tolist()),
                   population_step_std=population, results={})
    for key, record in records.items():
        names = sorted({k.split("|")[0] for k in record}, key=list(variants(np.zeros(8, np.float32))).index)
        exact_l1 = np.array(record["exact|l1"])
        summary["results"][key] = {"frames": len(exact_l1)}
        for name in names:
            l1 = np.array(record[f"{name}|l1"])
            delta = l1 - exact_l1
            summary["results"][key][name] = dict(
                l1=float(l1.mean()), delta=float(delta.mean()), delta_se=float(delta.std(ddof=1) / np.sqrt(len(delta))),
                relative=float(delta.mean() / exact_l1.mean()), shift=float(np.mean(record[f"{name}|shift"])),
                gripper_flip=float(np.mean(record[f"{name}|flip"])))
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
