"""LIBERO training entry for spike-driven-vla and archived extensions.

Use --variant spike-driven-vla for the base architecture. Legacy names remain
accepted for checkpoint and command compatibility. Importing this module never
starts training or evaluation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.libero.settings import CODE
from experiments.libero.train_spike_text_only import text_only_arguments
from experiments.libero.train_wta import main as run_training

CONFIG_DIR = Path(__file__).with_name("configs")
SHARED = json.loads((CONFIG_DIR / "shared_training.json").read_text())
BASE = json.loads((CONFIG_DIR / "spike_driven_vla.json").read_text())
EXTENSIONS = json.loads((CONFIG_DIR / "extensions" / "action_head_variants.json").read_text())
RECIPES = {
    **SHARED,
    "variants": {BASE["name"]: BASE["recipe"], **EXTENSIONS["variants"]},
    "legacy_variant_aliases": {
        "spike_driven_lightaug": BASE["name"],
        "shiftmax_ann_state": "spike_driven",
    },
}


def configure_parser(parser):
    names = (*RECIPES["variants"], *RECIPES.get("legacy_variant_aliases", {}))
    parser.add_argument("--variant", choices=names, required=True,
                        help="Select spike-driven-vla (base) or an extension recipe")


def action_head_arguments(options):
    variant = RECIPES.get("legacy_variant_aliases", {}).get(options.variant, options.variant)
    recipe = RECIPES["variants"][variant]
    common = RECIPES["common_training"]
    actual = (options.effective_world_size, options.batch_size,
              options.grad_accum_steps, options.max_steps)
    expected = (recipe["world_size"], recipe["batch_per_gpu"],
                common["gradient_accumulation"], recipe["max_steps"])
    if actual != expected:
        raise ValueError(
            f"{options.variant} requires world/batch/accum/steps={expected}, got {actual}. "
            "Use a separately named recipe for a different budget or rank layout."
        )
    if options.resume_mode != "none":
        raise ValueError("These published recipes start fresh; do not load another experiment's policy")
    run = Path(options.run_dir).expanduser().resolve()
    if run.is_relative_to(CODE):
        raise ValueError("Training outputs must be outside the source checkout")
    if not options.dry_run and run.exists():
        raise FileExistsError(f"Use a new run directory; existing runs are not overwritten: {run}")

    chosen = copy.copy(options)
    chosen.run_dir = str(run)
    chosen.warmup_steps = recipe["warmup_steps"]
    chosen.lr_schedule_steps = recipe["max_steps"]
    chosen.save_steps = common["save_steps"]
    chosen.extra_save_steps = recipe.get("extra_save_steps", common["extra_save_steps"])
    chosen.log_freq = common["log_freq"]
    chosen.num_workers = common["num_workers_per_rank"]
    chosen.vision_gradient_checkpointing = recipe["vision_gradient_checkpointing"]
    chosen.vision_checkpoint_scope = "all"
    chosen.projection_gradient_checkpointing = False
    arguments = [
        *text_only_arguments(chosen),
        *common["trainer_arguments"],
        "--action_head_type", recipe["head_type"],
        "--num_state_tokens", str(recipe["num_state_tokens"]),
        "--checkpoint_prefix", recipe["checkpoint_prefix"],
        "--lr_warmup_samples", str(recipe["lr_warmup_samples"]),
        "--spike_state_population", str(recipe["state_population"]),
    ]
    if recipe["head_type"] != "ann":
        arguments += [
            "--spike_state_encoding", recipe["state_encoding"],
            "--spike_state_population_range", str(recipe["state_population_range"]),
        ]
    arguments += ["--text_attention_routing", recipe.get("text_attention_routing", "wta")]
    if recipe.get("continuous_state", False):
        arguments.append("--continuous_state")
    if recipe.get("libero_light_augmentation", False):
        arguments.append("--libero_light_augmentation")
    return arguments


if __name__ == "__main__":
    run_training(argument_builder=action_head_arguments, description=__doc__,
                 configure_parser=configure_parser)
