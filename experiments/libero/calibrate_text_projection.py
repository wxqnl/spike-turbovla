"""Calibrate only the existing linear text interface, then probe the real frozen policy.

No language lookup is deployed: the saved model runs SmoothSpike online. All forty
training instruction templates are used; this does not test unseen-language generalization.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from experiments.libero.settings import CODE, BERT, DINO, SMOOTH, VISION, DATASETS
from turbovla.data.libero_lerobot import LiberoMixedLeRobotDataset, vla_collate_fn
from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.text_encoder import TurboVLATextEncoder
from turbovla.models.turbovla import build_turbovla


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def metrics(predicted, target, selected):
    p, t = predicted[selected], target[selected]
    return {"nmse": float((p-t).square().mean() / t.square().mean().clamp_min(1e-12)),
            "cosine": float(F.cosine_similarity(p, t, dim=-1).mean()),
            "mae": float((p-t).abs().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_checkpoint", required=True, help="Vision + ANN BERT policy with EMA weights")
    parser.add_argument("--smoothspike_path", default=SMOOTH)
    opts = parser.parse_args()
    destination = Path(opts.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    baseline = torch.load(opts.baseline_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    config = TurboVLAConfig.from_mapping(baseline["model_config"])
    args = SimpleNamespace(**baseline["args"])
    if (config.vision.encoder_type != "sdtv3_19m"
            or config.text.encoder_type != "bert"
            or config.sentence_mask_version != "legacy"):
        raise ValueError("This calibration recipe requires an SDT-V3 + ANN BERT baseline with legacy sentence masks")
    config.text.model_name_or_path = BERT
    config.vision.model_name_or_path = VISION
    config.vision.pretrained_checkpoint = VISION
    config.vision.model_source_path = str(CODE / "third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py")
    args.dataset_dir, args.dataset_dirs = DATASETS[0], ",".join(DATASETS)
    args.stats_path = str(CODE / "experiments/libero/configs/libero_all4_stats.json")
    args.dinov3_path = DINO
    model = build_turbovla(config).eval()
    model.load_state_dict(baseline["ema_model_state_dict"], strict=True)
    model.requires_grad_(False).to(device)
    # Correct only per-batch head indexing; this preserves B=1 reference behavior.
    model.config.text_head_mask_version = "corrected"
    for layer in model.vision_language_interaction.text_layers:
        layer.text_mask_version = "corrected"

    dataset = LiberoMixedLeRobotDataset(
        dataset_dir=args.dataset_dir, dataset_dirs=args.dataset_dirs,
        stats_path=args.stats_path, stats_key=args.stats_key,
        normalize_binary_gripper=args.normalize_binary_gripper,
        gripper_convention=args.lerobot_gripper_convention,
        LOCAL_DINOV3_PATH=args.dinov3_path, chunk_size=args.chunk_size,
        seed=args.seed, local_files_only=True, expected_image_size=args.expected_image_size,
        vision_encoder_type=args.vision_encoder_type, vision_image_size=args.vision_image_size,
        enable_dinov3_teacher=False)
    samples, frames, seen = [], [], set()
    for episode in dataset.episodes:
        key = (episode.suite_index, episode.fallback_instruction)
        if key in seen:
            continue
        seen.add(key)
        decoded = dataset._load_episode(episode)
        step = episode.expected_length // 2
        samples.append(dataset._build_step_sample(decoded, step))
        frames.append({"suite_index": episode.suite_index, "episode_index": episode.episode_index,
                       "step": step, "instruction": episode.fallback_instruction})
    assert len(samples) == 40
    images, instructions, states, targets, valid_actions = vla_collate_fn(samples)
    del dataset, samples
    gc.collect()
    states, targets = states.to(device), targets.to(device)
    valid_actions = valid_actions.to(device).bool().unsqueeze(-1).expand_as(targets)

    spike_config = copy.deepcopy(model.config.text)
    spike_config.encoder_type = "smoothspike_bert"
    spike_config.model_name_or_path = opts.smoothspike_path
    spike_config.timesteps = 4
    spike_config.attention_implementation = "eager"
    spike = TurboVLATextEncoder(spike_config, config.interaction.hidden_dim, "legacy").eval().to(device)
    spike.requires_grad_(False)
    spike.text_projection.load_state_dict(model.text_encoder.text_projection.state_dict(), strict=True)
    with torch.no_grad():
        teacher, padding, masks = model.text_encoder(instructions, device)
        hidden, valid, spike_masks = spike.encode_bert_hidden(instructions, device)
        assert torch.equal(valid, ~padding) and torch.equal(masks, spike_masks)
        raw = spike.text_projection(hidden)
        # The unchanged action decoder sees PAD rows too, so align all memory rows.
        x, y = hidden.reshape(-1, hidden.shape[-1]).double(), teacher.reshape(-1, teacher.shape[-1]).double()
        mx, my = x.mean(0), y.mean(0)
        xc, yc = x-mx, y-my
        gram = xc.T @ xc / len(x)
        regularization = 0.01 * gram.diag().mean()
        weight = torch.linalg.solve(gram + regularization * torch.eye(gram.shape[0], device=device, dtype=gram.dtype),
                                    xc.T @ yc / len(x))
        bias = my - mx @ weight
        spike.text_projection.weight.copy_(weight.T)
        spike.text_projection.bias.copy_(bias)
        calibrated = spike.text_projection(hidden)
        features = {"raw_valid": metrics(raw, teacher, valid),
                    "calibrated_valid": metrics(calibrated, teacher, valid),
                    "raw_padding": metrics(raw, teacher, padding),
                    "calibrated_padding": metrics(calibrated, teacher, padding)}
        emit("linear_calibration", regularization=float(regularization), **features)
        vision = torch.cat([model.encode_vision(images["pixel_values"][i:i+4].to(device)) for i in range(0,40,4)])
        def predict(text):
            outputs = []
            for i in range(0,40,4):
                v, t = model.vision_language_interaction(vision[i:i+4], text[i:i+4], padding[i:i+4], masks[i:i+4])
                outputs.append(model.action_head(torch.cat((v,t),dim=1), states[i:i+4]))
            return torch.cat(outputs)
        reference = predict(teacher)
        predictions = {"baseline_ann": reference, "unadapted_spike_swap": predict(raw),
                       "calibrated_spike_swap": predict(calibrated)}
        # Diagnostic only: isolate residual PAD mismatch without deploying teacher features.
        teacher_padding = torch.where(padding.unsqueeze(-1), teacher, calibrated)
        predictions["diagnostic_calibrated_with_ann_padding"] = predict(teacher_padding)
        report = {"verification_status": "ANALYZED", "baseline_checkpoint": str(opts.baseline_checkpoint),
                  "scope": "Frozen policy transfer, 40 training instruction templates and 40 real training frames; not a rollout score",
                  "changes": ["Frozen online SmoothSpike T4 replaces frozen BERT",
                              "Only existing 768-to-256 linear text projection fitted by ridge regression",
                              "Sentence/position/padding protocol unchanged; batched head indexing corrected"],
                  "frozen_modules": ["vision encoder", "vision-language interaction", "action decoder", "SmoothSpike backbone"],
                  "policy_optimizer_steps": 0, "ridge_relative_regularization": 0.01,
                  "ridge_strength": float(regularization), "frames": frames,
                  "unseen_language_generalization": "Not established; earlier 32/8 heldout probe remains separate",
                  "feature_metrics": features, "action_metrics": {}}
        for name, prediction in predictions.items():
            assert torch.isfinite(prediction).all()
            item = {"target_l1": float((prediction-targets).abs()[valid_actions].mean()),
                    "ann_policy_mae": float((prediction-reference).abs().mean()),
                    "ann_policy_max": float((prediction-reference).abs().max()),
                    "per_instruction_target_l1": (prediction-targets).abs().mean((1,2)).cpu().tolist(),
                    "per_instruction_ann_mae": (prediction-reference).abs().mean((1,2)).cpu().tolist()}
            report["action_metrics"][name] = item
            emit("real_frame_probe", variant=name, **{k:v for k,v in item.items() if not isinstance(v,list)})
        # Save ordinary inference weights. No text feature cache is part of the model.
        model.text_encoder = spike
        model.config.text = spike_config
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        output_args = vars(args).copy()
        output_args.update(text_encoder_type="smoothspike_bert", text_model_path=spike_config.model_name_or_path,
                           text_timesteps=4, text_sentence_mask_version="legacy", text_head_mask_version="corrected")
        checkpoint = destination / "spike_text_projection_calibrated.pth"
        torch.save({"model_state_dict": state, "model_config": model.config.to_dict(),
                    "args": output_args, "global_step": 0, "source_global_step": baseline.get("global_step"),
                    "checkpoint_kind": "ema_policy_with_calibrated_online_spike_text",
                    "calibration_metadata": report}, checkpoint)
        report["checkpoint"] = str(checkpoint)
        report["checkpoint_weights"] = "raw"
        torch.save({"frames": frames, "predictions": {k:v.cpu() for k,v in predictions.items()},
                    "targets": targets.cpu()}, destination / "action_probe.pt")
        (destination / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        emit("completed", checkpoint=str(checkpoint), result=str(destination/"result.json"))


if __name__ == "__main__":
    main()
