"""Launch a fresh LIBERO ablation, then evaluate its final EMA checkpoint."""
import argparse
import json
import os
import shlex

from experiments.libero.recipes import PRESETS, VISION_CHOICES, TEXT_CHOICES, CROSS_CHOICES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS),
                        help="Choose a preset; clears module overrides from the environment")
    parser.add_argument("--vision", choices=VISION_CHOICES)
    parser.add_argument("--text", choices=TEXT_CHOICES)
    parser.add_argument("--cross", choices=CROSS_CHOICES)
    parser.add_argument("--run-name", help="New output directory name; existing runs are never overwritten")
    parser.add_argument("--gpus", help="Three physical GPU indices, e.g. 4,5,6")
    parser.add_argument("--batch-size", type=int, help="Per-GPU microbatch (default: 128)")
    parser.add_argument("--grad-accum-steps", type=int, help="Accumulation steps (default: 1)")
    parser.add_argument("--max-steps", type=int, help="Optimizer updates (default: 70000)")
    parser.add_argument("--seed", type=int, help="Training seed (default: 42)")
    parser.add_argument("--eval-trials", type=int, help="Trials/task: 5 = 200 total, 50 = 2000 total")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved recipe/command only; no GPU use, file checks, or output creation")
    args = parser.parse_args()
    if args.preset is not None:
        for variable in ("SPIKE_VISION", "SPIKE_TEXT", "SPIKE_CROSS"):
            os.environ.pop(variable, None)
    for name, variable in (
        ("preset", "SPIKE_PRESET"), ("vision", "SPIKE_VISION"),
        ("text", "SPIKE_TEXT"), ("cross", "SPIKE_CROSS"), ("run_name", "SPIKE_RUN_NAME"),
        ("gpus", "SPIKE_CUDA_DEVICES"), ("batch_size", "SPIKE_BATCH_SIZE"),
        ("grad_accum_steps", "SPIKE_GRAD_ACCUM_STEPS"), ("max_steps", "SPIKE_MAX_STEPS"),
        ("seed", "SPIKE_SEED"), ("eval_trials", "SPIKE_EVAL_TRIALS"),
    ):
        value = getattr(args, name)
        if value is not None:
            os.environ[variable] = str(value)

    # Import only after configuration, so all child commands read the same recipe.
    from experiments.libero import settings as s
    from experiments.libero import run_cross_joint_stage as pipeline

    if args.dry_run:
        command = [s.PYTHON, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=3",
                   str(s.CODE / "experiments/libero/train.py"),
                   *pipeline.training_arguments(s.CODE, s.RUN)]
        print(json.dumps({
            "preset": s.PRESET, "variant": s.VARIANT, "modules": s.MODULES,
            "gpus": s.GPUS, "run": str(s.RUN),
            "calibrate_text": s.TEXT_TYPE == "smoothspike_bert",
            "batch_per_gpu": s.BATCH_SIZE, "gradient_accumulation": s.GRAD_ACCUM,
            "global_batch": s.GLOBAL_BATCH, "optimizer_steps": s.MAX_STEPS,
            "eval_episodes": 40 * s.EVAL_TRIALS,
            "float32_matmul_precision": "highest",
            "training_scope": "joint_except_frozen_text_backbone",
            "final_checkpoint": str(s.RUN / "checkpoints" / f"{s.CHECKPOINT_PREFIX}_{s.MAX_STEPS}.pth"),
            "training_command": shlex.join(command),
            "note": "Actual execution uses an isolated source snapshot; dry-run does not validate assets.",
        }, indent=2))
        return
    pipeline.main([])


if __name__ == "__main__":
    main()
