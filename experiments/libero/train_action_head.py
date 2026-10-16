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
VISION = json.loads((CONFIG_DIR / "extensions" / "vision_attribution.json").read_text())
IFR = json.loads((CONFIG_DIR / "extensions" / "ifr.json").read_text())
RECIPES = {
    **SHARED,
    "variants": {BASE["name"]: BASE["recipe"], **EXTENSIONS["variants"], **VISION["variants"], **IFR["variants"]},
    "legacy_variant_aliases": {
        "spike_driven_lightaug": BASE["name"],
        "shiftmax_ann_state": "spike_driven",
    },
}


def configure_parser(parser):
    names = (*RECIPES["variants"], *RECIPES.get("legacy_variant_aliases", {}))
    parser.add_argument("--variant", choices=names, required=True,
                        help="Select spike-driven-vla (base) or an extension recipe")
    parser.add_argument("--vision_backbone",
                        help="Local Hugging Face ViT directory for recipes with vision_encoder=hf_vit")


def action_head_arguments(options):
    variant = RECIPES.get("legacy_variant_aliases", {}).get(options.variant, options.variant)
    recipe = RECIPES["variants"][variant]
    common = RECIPES["common_training"]
    actual = (options.effective_world_size, options.batch_size,
              options.grad_accum_steps, options.max_steps)
    expected = (recipe["world_size"], recipe["batch_per_gpu"],
                recipe.get("gradient_accumulation", common["gradient_accumulation"]), recipe["max_steps"])
    if actual != expected:
        raise ValueError(
            f"{options.variant} requires world/batch/accum/steps={expected}, got {actual}. "
            "Use a separately named recipe for a different budget or rank layout."
        )
    run = Path(options.run_dir).expanduser().resolve()
    if run.is_relative_to(CODE):
        raise ValueError("Training outputs must be outside the source checkout")
    if options.resume_mode == "all":
        # Only continues this run from its own latest checkpoint (same recipe, same directory), e.g. after a paused
        # queue; never loads another experiment's policy.
        if not any((run / "checkpoints").glob(f"{recipe['checkpoint_prefix']}_*.pth")):
            raise FileNotFoundError(f"No {recipe['checkpoint_prefix']} checkpoint to continue in {run}")
    elif options.resume_mode != "none":
        raise ValueError("These published recipes start fresh; do not load another experiment's policy")
    elif not options.dry_run and run.exists():
        raise FileExistsError(f"Use a new run directory; existing runs are not overwritten: {run}")

    chosen = copy.copy(options)
    chosen.run_dir = str(run)
    chosen.warmup_steps = recipe["warmup_steps"]
    chosen.lr_schedule_steps = recipe["max_steps"]
    chosen.save_steps = recipe.get("save_steps", common["save_steps"])
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
    if recipe.get("ifr_integrate", False):
        arguments.append("--ifr_integrate")
    if recipe.get("ifr_shift", 0):
        arguments += ["--ifr_shift", str(recipe["ifr_shift"])]
    if recipe.get("ifr_split_state", False):
        arguments.append("--ifr_split_state")
    if recipe.get("ifr_locate", False):
        arguments.append("--ifr_locate")
    if recipe.get("ifr_spatial", 0):
        arguments += ["--ifr_spatial", str(recipe["ifr_spatial"])]
    if recipe.get("ifr_norm_bits", 0):
        arguments += ["--ifr_norm_bits", str(recipe["ifr_norm_bits"])]
    if recipe.get("ifr_sign_field", False):
        arguments.append("--ifr_sign_field")
    if recipe.get("ifr_context_levels", 0):
        arguments += ["--ifr_context_levels", str(recipe["ifr_context_levels"])]
    if "other_lr" in recipe:  # overrides the shared 5e-5 / 1e-5 (argparse keeps the last occurrence)
        arguments += ["--lr", str(recipe["other_lr"]), "--head_lr", str(recipe["other_lr"]),
                      "--vision_encoder_lr", str(recipe["vision_lr"]), "--dinov3_lr", str(recipe["vision_lr"])]
    if recipe.get("position_embedding", "view") != "view":
        arguments += ["--position_embedding", recipe["position_embedding"]]
    arguments += vision_arguments(options, recipe)
    if recipe.get("distillation", True) is False:
        # The paper recipe trains with the masked action L1 only.
        arguments += ["--distill_feature_weight", "0", "--distill_action_weight", "0", "--distill_text_weight", "0"]
    return arguments


def vision_arguments(options, recipe):
    """Swap the SDT-V3 slot for an ANN ViT of the recipe's family; everything after the encoder is unchanged."""
    if recipe.get("vision_encoder", "sdtv3_19m") == "sdtv3_19m":
        if options.vision_backbone:
            raise ValueError(f"{options.variant} uses SDT-V3; do not pass --vision_backbone")
        return []
    if not options.vision_backbone:
        raise ValueError(f"{options.variant} requires --vision_backbone (local Hugging Face directory)")
    backbone = Path(options.vision_backbone).expanduser().resolve()
    config = json.loads((backbone / "config.json").read_text())
    expected = recipe["vision_check"]
    found = {"model_type": config.get("model_type"), "hidden_size": config.get("hidden_size"),
             "patch_size": config.get("patch_size")}
    if (found["model_type"] not in expected["model_type"] or found["hidden_size"] != expected["hidden_size"]
            or found["patch_size"] != expected["patch_size"]):
        raise ValueError(f"{options.variant} expects {expected}, but {backbone} is {found}")
    if not (backbone / "preprocessor_config.json").is_file():
        raise FileNotFoundError(f"{backbone} lacks preprocessor_config.json (needed for evaluation)")
    return ["--vision_encoder_type", "dinov3", "--vision_model_path", str(backbone),
            "--vision_image_size", str(recipe["vision_image_size"])]


if __name__ == "__main__":
    run_training(argument_builder=action_head_arguments, description=__doc__,
                 configure_parser=configure_parser)
