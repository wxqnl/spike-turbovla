"""Shared A / AA / WTA launcher; run with torchrun, never starts via import.

Example: torchrun --standalone --nproc_per_node=3 experiments/libero/train_wta.py
  --run_dir ... --vision_checkpoint ... --text_projection_init ...
  --max_steps 70000 --batch_size 64 --grad_accum_steps 2 --approve_training
"""
import argparse
import json
import os
import sys
from pathlib import Path

from experiments.libero.run_cross_joint_stage import training_arguments
from experiments.libero.settings import CODE, SMOOTH


def wta_arguments(options):
    return [
        *training_arguments(CODE, Path(options.run_dir)),
        "--fusion_backbone_type", "spike_wta", "--cross_attention_type", "spike_wta",
        "--cross_timesteps", "4", "--fusion_droppath", "0",
        "--wta_text_topk", "3", "--wta_vision_topk", "8",
        "--wta_surrogate_temperature", "1", "--spike_norm_momentum", "0.1",
        "--spike_norm_eps", "1e-5", "--spike_norm_sync",
        "--text_encoder_type", "smoothspike_bert", "--text_model_path", SMOOTH,
        "--text_timesteps", "4", "--vision_encoder_type", "sdtv3_19m",
        "--vision_image_size", "224", "--vision_model_source_path",
        str(CODE / "third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py"),
        "--vision_model_path", options.vision_checkpoint,
        "--vision_pretrained_checkpoint", options.vision_checkpoint,
        "--text_projection_init", options.text_projection_init,
        "--checkpoint_prefix", "spike_wta", "--max_steps", str(options.max_steps),
        "--lr_schedule_steps", str(options.lr_schedule_steps),
        "--warmup_steps", str(options.warmup_steps), "--batch_size", str(options.batch_size),
        "--grad_accum_steps", str(options.grad_accum_steps),
        "--num_workers", str(options.num_workers), "--log_freq", str(options.log_freq),
        "--save_steps", str(options.save_steps), "--extra_save_steps",
        *map(str, options.extra_save_steps),
        "--save_final", "--resume_mode", options.resume_mode,
        "--approve_spike_fusion_training",
        *(["--vision_gradient_checkpointing"] if options.vision_gradient_checkpointing else []),
        "--vision_checkpoint_scope", options.vision_checkpoint_scope,
        *(["--projection_gradient_checkpointing"] if options.projection_gradient_checkpointing else []),
    ]


def main(argument_builder=wta_arguments, description=__doc__, required_world_size=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--vision_checkpoint", required=True)
    parser.add_argument("--text_projection_init", required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=3334)
    parser.add_argument("--lr_schedule_steps", type=int, default=70000)
    parser.add_argument("--extra_save_steps", type=int, nargs="*", default=[20, 1000])
    parser.add_argument("--resume_mode", choices=["none", "all"], default="none")
    parser.add_argument("--approve_training", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", action="store_true",
                        help="Print the actual recipe without importing the trainer or using a GPU")
    parser.add_argument("--world-size", type=int,
                        help="Process count for --dry-run only; live training uses torchrun WORLD_SIZE")
    parser.add_argument("--vision_gradient_checkpointing", action="store_true")
    parser.add_argument("--vision_checkpoint_scope", choices=["all", "early"], default="all")
    parser.add_argument("--projection_gradient_checkpointing", action="store_true")
    options = parser.parse_args()
    actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if options.world_size is not None and not options.dry_run:
        raise ValueError("--world-size is only a preview option; use torchrun for live training")
    world_size = options.world_size if options.world_size is not None else actual_world_size
    if world_size < 1:
        raise ValueError("World size must be positive")
    if required_world_size is not None and world_size != required_world_size:
        raise ValueError(f"This recipe requires {required_world_size} processes, got {world_size}")
    if not options.dry_run and not options.approve_training:
        raise RuntimeError("Review the architecture/configuration and explicitly pass --approve_training")
    if (min(options.max_steps, options.batch_size, options.grad_accum_steps,
            options.log_freq, options.save_steps) < 1 or options.max_steps > 70000
            or options.num_workers < 0):
        raise ValueError("Invalid WTA training budget or batch configuration")
    if options.warmup_steps < 0 or options.lr_schedule_steps <= options.warmup_steps:
        raise ValueError("Learning-rate schedule must extend beyond nonnegative warmup")
    if any(step < 1 for step in options.extra_save_steps):
        raise ValueError("Extra checkpoint steps must be positive")
    if options.resume_mode == "none" and any((Path(options.run_dir) / "checkpoints").glob("*.pth")):
        raise FileExistsError("Fresh variants require separate, unused run directories")
    arguments = argument_builder(options)
    if options.dry_run:
        global_batch = world_size * options.batch_size * options.grad_accum_steps
        print(json.dumps({
            "entry": argument_builder.__module__,
            "world_size": world_size,
            "batch_per_gpu": options.batch_size,
            "gradient_accumulation": options.grad_accum_steps,
            "global_batch": global_batch,
            "optimizer_steps": options.max_steps,
            "sample_exposures": global_batch * options.max_steps,
            "training_arguments": arguments,
            "note": "Preview only; no trainer import, model load, output creation, or GPU use.",
        }, indent=2))
        return
    from turbovla.training import train_mixed

    sys.argv = [sys.argv[0], *arguments]
    train_mixed.main()


if __name__ == "__main__":
    main()
