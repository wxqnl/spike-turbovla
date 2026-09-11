"""AA: A's projections/fusion plus a spike-driven action head.

Fresh common initialization, frozen online SmoothSpike, all remaining modules
jointly trained. Default deployment follows B: 8 x 128, 18,750 updates.
"""
import math
from experiments.libero.train_spike_text_only import text_only_arguments
from experiments.libero.train_wta import main as run_training


def aa_arguments(options):
    # The user's selected eight-GPU recipe preserves A/B's sample exposure.
    global_batch = 8 * options.batch_size * options.grad_accum_steps
    if global_batch * options.max_steps != 19_200_000:
        raise ValueError("AA requires exactly 19,200,000 training sample exposures")
    return [
        *text_only_arguments(options),
        "--action_head_type", "spike_sdsa",
        "--checkpoint_prefix", "spike_aa",
        "--lr_schedule_steps", str(options.max_steps),
        "--warmup_steps", str(math.ceil(3334 * 384 / global_batch)),
        "--extra_save_steps", "20", str(max(1, 1000 * 384 // global_batch)),
    ]


if __name__ == "__main__":
    run_training(argument_builder=aa_arguments, description=__doc__, required_world_size=8)
