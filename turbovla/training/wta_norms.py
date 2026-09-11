"""Training-only running statistics and strict WTA checkpoint compatibility."""
from contextlib import contextmanager

import torch
from torch import nn

from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.components.wta_operators import running_norms, verify_running_norms


def check_wta_resume_config(config, saved):
    previous = TurboVLAConfig.from_mapping(saved)
    if (previous.interaction.backbone_type not in {"spike_wta", "spike_text_only"}
            or previous.interaction.backbone_type != config.interaction.backbone_type):
        raise ValueError("Resume must keep the same full-WTA or text-only architecture")
    # Paths and gradient checkpointing may move/change without changing math.
    # Architecture, neuron/routing/normalization settings and mask layout may not.
    current, old = config.to_dict(), previous.to_dict()
    for mapping in (current, old):
        mapping["interaction"].pop("cross_gradient_checkpointing", None)
        mapping["interaction"].pop("projection_gradient_checkpointing", None)
        mapping["vision"].pop("gradient_checkpointing", None)
        mapping["vision"].pop("checkpoint_scope", None)
        for section in ("text", "vision"):
            for field in ("model_name_or_path", "pretrained_checkpoint", "model_source_path", "local_files_only"):
                mapping[section].pop(field, None)
    if current != old:
        differences = [section for section in current if current[section] != old[section]]
        raise ValueError(f"WTA resume changes model semantics in: {differences}")


def normalization_metadata(ema=False):
    return {
        "version": 1,
        "kind": "masked_per_timestep_synchronized_batch_norm",
        "raw": {"status": "training_running_stats", "source": "training_data"},
        "ema": {"status": "requires_training_data_refresh" if ema else "absent"},
    }


def verify_wta_evaluation(model, checkpoint, weights):
    metadata = checkpoint.get("wta_normalization", {})
    expected = "refreshed_with_training_data" if weights == "ema" else "training_running_stats"
    entry = metadata.get(weights, {})
    if metadata.get("version") != 1 or entry.get("status") != expected:
        raise ValueError(
            f"WTA {weights} normalization is not ready for evaluation. "
            "Export EMA weights with experiments/libero/refresh_wta_norms.py using training data."
        )
    return verify_running_norms(model)


@contextmanager
def training_stat_refresh(model):
    """Refresh WTA and trainable vision BN with fixed EMA parameters.

    Dropout and the frozen SmoothSpike backbone stay in eval mode. Only BN
    accumulators change. The caller must use the recorded training dataset.
    """
    norms = running_norms(model)
    if not norms:
        raise ValueError("Stat refresh requires a WTA model")
    vision_bn = [module for module in model.vision_encoder.modules()
                 if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats]
    modes = {module: module.training for module in model.modules()}
    momenta = {module: module.momentum for module in vision_bn}
    model.eval()
    for _, module in norms:
        module.reset_running_stats()
        module.calibrating = True
    for module in vision_bn:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    try:
        with torch.no_grad():
            yield
        verify_running_norms(model)
        for module in vision_bn:
            if (module.num_batches_tracked.item() == 0
                    or not torch.isfinite(module.running_mean).all()
                    or not torch.isfinite(module.running_var).all()):
                raise RuntimeError("Vision BN refresh did not complete")
    finally:
        for _, module in norms:
            module.calibrating = False
        for module, momentum in momenta.items():
            module.momentum = momentum
        for module, mode in modes.items():
            module.training = mode
