"""Where does the spiking vision pathway lose the geometry a manipulation policy needs?

Linear (kernel ridge) probes, cross-validated by episode, decode robot geometry and policy targets from the visual
tokens of each view at every stage of the trained v1 base:
  pixels (28x28) -> SDT-V3 stage 1 / stage 2 / block3.2 / block3.5 / final residual map -> the projection's input
  spikes (ternary PLIF, mean over T) -> projected tokens -> fused tokens after vision-language interaction,
and of frozen ImageNet-pretrained ANN references of similar size: DINOv2 ViT-S/14 (self-supervised) and ResNet-18
(supervised). The base's SDT-V3 was fine-tuned on LIBERO; the references were not.

Targets: end-effector position / rotation, gripper opening (normalized state), episode progress, and the first
demo action (6 arm dims). Object geometry: the end-effector position at the demo's first grasp (object location)
and at the following release (placement location), on frames before that event, centered per task, so R^2 measures
how well the per-episode object placement (a few cm) is localized. Spatial maps are average-pooled to 14x14
(ResNet-18 layer 4 stays 7x7).

    python vision_pathway_probe.py <ckpt_routing_probe arguments> --dinov2-repo ~/.cache/torch/hub/<repo> \
        --dinov2-weights dinov2_vits14_pretrain.pth --resnet18-weights resnet18-f37072fd.pth --out vision.json
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ckpt_routing_probe import build_model, frames  # noqa: E402

TARGETS = {"eef position": slice(0, 3), "eef rotation": slice(3, 6), "gripper": slice(6, 8),
           "progress": slice(8, 9), "action": slice(9, 15)}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--checkpoint", "--v1-root", "--bert", "--data-root", "--work", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--dinov2-repo")
    p.add_argument("--dinov2-weights")
    p.add_argument("--resnet18-weights")
    p.add_argument("--episodes-per-suite", type=int, default=25)
    p.add_argument("--frames-per-episode", type=int, default=20)
    p.add_argument("--threads", type=int, default=24)
    return p.parse_args()


def grid(x, size=14):
    """[N, C, H, W] -> [N, size*size*C] after average pooling (no pooling if already that small)."""
    if x.shape[-1] > size:
        x = F.adaptive_avg_pool2d(x.float(), size)
    return x.flatten(1)


def references(args):
    refs = {}
    if args.dinov2_repo and args.dinov2_weights:
        dinov2 = torch.hub.load(args.dinov2_repo, "dinov2_vits14", source="local", pretrained=False)
        dinov2.load_state_dict(torch.load(args.dinov2_weights, map_location="cpu"), strict=True)
        refs["DINOv2 ViT-S/14"] = dinov2.eval()
    if args.resnet18_weights:
        import torchvision
        resnet = torchvision.models.resnet18()
        resnet.load_state_dict(torch.load(args.resnet18_weights, map_location="cpu"), strict=True)
        refs["ResNet-18"] = resnet.eval()
    return refs


def reference_features(refs, images):
    out = {}
    if "DINOv2 ViT-S/14" in refs:
        tokens = refs["DINOv2 ViT-S/14"].forward_features(images)["x_norm_patchtokens"]  # [N, 256, 384]
        side = int(tokens.shape[1] ** 0.5)
        out["ref DINOv2-S final"] = grid(tokens.transpose(1, 2).reshape(len(images), -1, side, side))
    if "ResNet-18" in refs:
        r = refs["ResNet-18"]
        x = r.maxpool(r.relu(r.bn1(r.conv1(images))))
        x = r.layer3(r.layer2(r.layer1(x)))
        out["ref ResNet-18 layer3"] = grid(x)
        out["ref ResNet-18 layer4"] = grid(r.layer4(x))
    return out


def ridge_r2(gram, targets, episodes):
    """Kernel ridge with a linear kernel; outer 2-fold and inner 2-fold CV split by episode. Returns R^2 per target."""
    unique = sorted(set(episodes))
    fold_of = {e: i % 2 for i, e in enumerate(unique)}
    folds = np.array([fold_of[e] for e in episodes])
    scores = []
    for fold in (0, 1):
        train, test = np.where(folds != fold)[0], np.where(folds == fold)[0]
        inner = np.array([(unique.index(episodes[i]) // 2) % 2 for i in train])  # halves of the train episodes
        best = select_lambda(gram[np.ix_(train, train)], targets[train], inner)
        predicted = fit_predict(gram, targets, train, test, best)
        residual = ((targets[test] - predicted) ** 2).sum(0)
        total = ((targets[test] - targets[test].mean(0)) ** 2).sum(0)
        scores.append(1 - residual / np.maximum(total, 1e-12))
    return np.mean(scores, 0)


def centered(gram, train, rows, cols):
    mean_row = gram[:, train].mean(1)
    mean_all = gram[np.ix_(train, train)].mean()
    return gram[np.ix_(rows, cols)] - mean_row[rows, None] - mean_row[None, cols] + mean_all


def fit_predict(gram, targets, train, test, lambdas):
    k_train = centered(gram, train, train, train)
    k_test = centered(gram, train, test, train)
    values, vectors = np.linalg.eigh(k_train)
    y = targets[train] - targets[train].mean(0)
    projected = vectors.T @ y
    out = np.empty((len(test), targets.shape[1]))
    for j, lam in enumerate(lambdas):
        alpha = vectors @ (projected[:, j] / (values + lam))
        out[:, j] = k_test @ alpha + targets[train, j].mean()
    return out


def select_lambda(gram, targets, split):
    scale = np.trace(gram) / len(gram)
    grid_values = scale * np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    errors = np.zeros((len(grid_values), targets.shape[1]))
    index = np.arange(len(gram))
    for part in (0, 1):
        train, test = index[split != part], index[split == part]
        k_train = centered(gram, train, train, train)
        k_test = centered(gram, train, test, train)
        values, vectors = np.linalg.eigh(k_train)
        y = targets[train] - targets[train].mean(0)
        projected = vectors.T @ y
        for g, lam in enumerate(grid_values):
            alpha = vectors @ (projected / (values + lam)[:, None])
            errors[g] += ((k_test @ alpha + targets[train].mean(0) - targets[test]) ** 2).sum(0)
    return grid_values[errors.argmin(0)]


def main():
    args = parse()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    model, checkpoint = build_model(args)
    refs = references(args)
    from turbovla.evaluation.policy import build_sdtv3_manual_processor
    processor = build_sdtv3_manual_processor()
    stats_path = Path(args.v1_root) / "experiments" / "libero" / "configs" / "libero_all4_stats.json"
    stats = json.loads(stats_path.read_text())[checkpoint["args"]["stats_key"]]

    backbone, captured = model.vision_encoder.backbone, {}
    taps = {"SDT-V3 stage 1": backbone.ConvBlock1_2[-1], "SDT-V3 stage 2": backbone.ConvBlock2_2[-1],
            "SDT-V3 block3.2": backbone.block3[2], "SDT-V3 block3.5": backbone.block3[5],
            "SDT-V3 final": backbone.block4[-1]}
    for name, module in taps.items():
        module.register_forward_hook(lambda _, __, out, name=name: captured.setdefault(name, []).append(out.detach()))
    model.vision_projection.spike_nodes["input"].register_forward_hook(
        lambda _, __, out: captured.setdefault("projection input spikes", []).append(out[0].detach()))

    features = defaultdict(list)  # (stage, view) -> list of [D] float16
    targets, episodes, objects, start = [], [], [], time.time()
    for count, sample in enumerate(frames(args, stats, model.config.action.horizon), 1):
        images = processor([sample["primary"], sample["wrist"]])["pixel_values"].float()  # [2, 3, 224, 224]
        captured.clear()
        with torch.inference_mode():
            condition, extra = model.encode_condition([sample["instruction"]], {"pixel_values": images[None]},
                                                      return_distillation_features=True)
            row = {"pixels": grid(images, 28)}
            for name in taps:
                # views are encoded separately: one call per view, in order [agent, wrist]
                row[name] = torch.cat([grid(x) for x in captured[name]])
            spikes = torch.stack(captured["projection input spikes"]).mean(0)  # [1, V, 196, C]
            row["projection input spikes"] = spikes[0].flatten(1)
            row["projected tokens"] = extra["visual"][0].reshape(2, 196, -1).flatten(1)
            row["fused tokens"] = condition[0, :392].reshape(2, 196, -1).flatten(1)
            row.update(reference_features(refs, images))
        for name, value in row.items():
            for view, label in enumerate(("agent", "wrist")):
                features[(name, label)].append(value[view].numpy().astype(np.float16))
        targets.append(np.concatenate([sample["state"], [sample["progress"]], sample["target"][0, :6]]))
        episodes.append(sample["episode"])
        objects.append([sample["instruction"]] + [
            event[1] if event is not None and sample["step"] < event[0] else None
            for event in (sample["grasp"], sample["place"])])
        if count % 100 == 0:
            print(f"[vision] {count} frames, {(time.time() - start) / count:.2f} s/frame", flush=True)

    targets = np.asarray(targets, np.float64)
    targets = (targets - targets.mean(0)) / (targets.std(0) + 1e-8)
    grams, summary = {}, dict(frames=len(episodes), dims={}, r2={})
    for (name, view), rows in features.items():
        x = torch.from_numpy(np.stack(rows).astype(np.float32))
        x = x / x.norm(dim=1).mean().clamp_min(1e-12)  # comparable kernel scale across stages
        grams[(name, view)] = (x @ x.T).double().numpy()
        summary["dims"][name] = int(x.shape[1])
    names = list(dict.fromkeys(name for name, _ in features))
    for name in names:
        for view in ("agent", "wrist", "both"):
            gram = grams[(name, "agent")] + grams[(name, "wrist")] if view == "both" else grams[(name, view)]
            r2 = ridge_r2(gram, targets, episodes)
            summary["r2"][f"{name} | {view}"] = {group: float(r2[cols].mean()) for group, cols in TARGETS.items()}
        for target, column in (("grasp location", 1), ("place location", 2)):
            rows = [i for i, o in enumerate(objects) if o[column] is not None]
            values = np.array([objects[i][column] for i in rows], np.float64)
            tasks = [objects[i][0] for i in rows]
            for task in set(tasks):
                members = [j for j, t in enumerate(tasks) if t == task]
                values[members] -= values[members].mean(0)
            values /= values.std(0) + 1e-8
            for view in ("agent", "wrist", "both"):
                gram = grams[(name, "agent")] + grams[(name, "wrist")] if view == "both" else grams[(name, view)]
                r2 = ridge_r2(gram[np.ix_(rows, rows)], values, [episodes[i] for i in rows])
                summary["r2"][f"{name} | {view}"][target] = float(r2.mean())
                summary["r2"][f"{name} | {view}"][target + " frames"] = len(rows)
        print(f"[vision] {name}: " + json.dumps(summary["r2"][f"{name} | both"]), flush=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
