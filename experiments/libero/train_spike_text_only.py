"""Variant A: spike projections and text blocks, with the original SDSA Cross.

Use the same explicit budget/asset arguments as train_wta.py, with a separate
run directory. Cross remains the original binary SDSA/PLIF implementation,
including its own temporal readout, LayerNorm, residual style and DropPath.
The two signed-PLIF projections read out T=4 before entering that unchanged
Cross. Older ANN-projection text-only checkpoints remain a separate config.
"""
from experiments.libero.train_wta import main as run_training, wta_arguments


def text_only_arguments(options):
    return [
        *wta_arguments(options),
        "--fusion_backbone_type", "spike_text_only",
        "--cross_attention_type", "spike_sdsa",
        "--fusion_droppath", "0.1",
        "--text_only_spike_projections",
        "--checkpoint_prefix", "spike_text_projected",
    ]


if __name__ == "__main__":
    run_training(argument_builder=text_only_arguments, description=__doc__)
