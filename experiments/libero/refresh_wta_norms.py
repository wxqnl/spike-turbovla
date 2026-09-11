"""Export A / AA / B EMA using only the recorded LIBERO training data."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from turbovla.data.libero_lerobot import LiberoMixedLeRobotDataset, vla_collate_fn
from turbovla.evaluation.policy import configure_transformers_offline
from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.turbovla import build_turbovla
from turbovla.models.components.wta_operators import verify_running_norms
from turbovla.training.trainer import move_samples_to_device, set_seed
from turbovla.training.wta_norms import training_stat_refresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    options = parser.parse_args()
    if min(options.batches, options.batch_size) < 1 or options.num_workers < 0:
        raise ValueError("Positive batches/batch_size and nonnegative workers required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run EMA-stat export on one GPU, not under torchrun")
    source = Path(options.checkpoint).resolve()
    output = Path(options.output).resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output == source or output.exists() or temporary.exists():
        raise FileExistsError("EMA export must use a fresh output; training checkpoints are never overwritten")
    configure_transformers_offline()
    torch.set_num_threads(4)
    checkpoint = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    config = TurboVLAConfig.from_mapping(checkpoint["model_config"])
    args = SimpleNamespace(**checkpoint["args"])
    if (config.interaction.backbone_type not in {"spike_wta", "spike_text_only"}
            or args.dataset_format != "lerobot"
            or getattr(args, "dataset_split", "train") != "train"):
        raise ValueError("Expected an A / AA / B checkpoint with recorded LeRobot training data")
    numerics = checkpoint.get("numerics", {})
    if (numerics.get("float32_matmul_precision") != "highest"
            or numerics.get("cudnn_allow_tf32") is not False):
        raise ValueError("Expected the A / AA / B highest/fp32, TF32-disabled training policy")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    set_seed(args.seed)
    device = torch.device("cuda:0")
    model = build_turbovla(config)
    model.load_state_dict(checkpoint["ema_model_state_dict"], strict=True)
    model.to(device).float().eval().requires_grad_(False)
    dataset = LiberoMixedLeRobotDataset(
        dataset_dir=args.dataset_dir, dataset_dirs=args.dataset_dirs,
        stats_path=args.stats_path, stats_key=args.stats_key,
        normalize_binary_gripper=args.normalize_binary_gripper,
        gripper_convention=args.lerobot_gripper_convention,
        LOCAL_DINOV3_PATH=args.dinov3_path, chunk_size=args.chunk_size,
        rank=0, world_size=1, seed=args.seed, local_files_only=True,
        expected_image_size=args.expected_image_size,
        vision_encoder_type=args.vision_encoder_type, vision_image_size=args.vision_image_size,
        shuffle_buffer=args.shuffle_buffer,
        shuffle_steps_within_episode=args.shuffle_steps_within_episode,
        step_mix_buffer_size=args.step_mix_buffer_size, enable_dinov3_teacher=False)
    iterator = iter(DataLoader(
        dataset, batch_size=options.batch_size, num_workers=options.num_workers,
        collate_fn=vla_collate_fn, pin_memory=True, drop_last=True))
    with training_stat_refresh(model):
        for index in range(options.batches):
            images, instructions, states, _, _ = next(iterator)
            predictions = model(instructions, move_samples_to_device(images, device), states.to(device))
            if not torch.isfinite(predictions).all():
                raise RuntimeError("Non-finite output during EMA training-stat refresh")
            if index == 0 or (index + 1) % 16 == 0 or index + 1 == options.batches:
                print(json.dumps({"event": "training_stat_refresh", "batch": index + 1,
                                  "batches": options.batches}), flush=True)
    norms = verify_running_norms(model)
    metadata = {
        "version": 1, "kind": "masked_per_timestep_synchronized_batch_norm",
        "raw": {"status": "absent"},
        "ema": {"status": "refreshed_with_training_data", "source": "training_data",
                "source_checkpoint": str(source), "dataset_dirs": args.dataset_dirs,
                "seed": args.seed, "batches": options.batches, "batch_size": options.batch_size,
                "optimizer_steps": 0, "vision_bn_refreshed": True, "layers": norms},
    }
    export = {key: checkpoint[key] for key in
              ("model_config", "args", "numerics", "global_step", "ema_decay", "training_scope")}
    export.update(ema_model_state_dict={key: tensor.detach().cpu()
                                       for key, tensor in model.state_dict().items()},
                  wta_normalization=metadata, export_kind="ema_evaluation_only")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as stream:
        torch.save(export, stream)
    temporary.replace(output)
    print(json.dumps({"event": "ema_export_complete", "output": str(output),
                      "norm_layers": len(norms), "optimizer_steps": 0}), flush=True)


if __name__ == "__main__":
    main()
