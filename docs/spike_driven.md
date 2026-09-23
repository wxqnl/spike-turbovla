# Spike-Driven VLA

This is the promoted action-head architecture in this repository. Its public name is **Spike-Driven VLA**; the supported LIBERO training recipe is `--variant spike_driven_lightaug`. The earlier unaugmented recipe remains available as `--variant spike_driven` for comparison. The original internal action-head type string is retained for checkpoint compatibility.

## Architecture

The fixed upstream is SDT-V3 vision, SmoothSpike text at T=4, spike projections and `spike_text_only` fusion with text Top-3 WTA and SDSA cross-attention. Each of the eight robot-state coordinates enters its own Linear/GELU encoder. The action decoder uses three layers, eight heads, hidden width 256, an eight-token state representation and a 12-action horizon. Action self/cross attention keeps continuous query and memory inputs, projected signed-spike Q/K, continuous V, Shiftmax and spiking context. The FFN and readout retain their spiking paths. See [model assembly](../turbovla/models/turbovla.py), [state/action head](../turbovla/models/full_state_action_head.py) and [Shiftmax](../turbovla/models/shiftmax_action_head.py).

Some state, value, residual and output paths remain continuous. “Spike-Driven” names the architecture family; it does **not** claim a fully event-driven implementation or measured energy savings.

## Selected completed benchmarks

| Benchmark | Training checkpoint | Result |
| --- | --- | ---: |
| LIBERO four suites, 40 tasks × 50 trials | 40k, global batch 256, light RGB augmentation, own refreshed EMA | **1841/2000 = 92.05%** |
| Public Meta-World MT50, single corner2 view, 50 tasks × 50 trials | cumulative 40k, global batch 256, own refreshed EMA | **1897/2500 = 75.88%** |

LIBERO breakdown: Spatial 452/500 (90.4%), Object 493/500 (98.6%), Goal 486/500 (97.2%), Long 410/500 (82.0%). The public MT50 run uses policy seed 7, the fixed native-reset manifest, open-loop 12 and a 500-native-step limit. These are different benchmarks and their percentages must not be compared as a single score. The LIBERO ANN-head reference elsewhere in this repository reached 92.35%, so 92.05% is **not** an all-architecture LIBERO record. Both displayed scores are from completed, successful runs of this architecture; they are not projections from training loss. [Machine-readable selected results](results/spike_driven_benchmarks_20260923.json).

## LIBERO recipe and code

The selected LIBERO run used 8 × 32, accumulation 1, global batch 256, 40,000 optimizer steps and 10,240,000 sample exposures. Warmup spans 1,280,000 samples / 5,000 steps. Vision LR is 1e-5 and other trainable groups use 5e-5 after warmup; seed 42, FP32/highest and both TF32 switches off. The loss is masked action L1 + 0.1 visual feature KD + 0.1 action KD, with no text KD. Only the text backbone is frozen. Vision activation checkpointing is off for this recipe; upstream cross and action-head checkpointing retain their recorded behavior.

During **training only**, every real RGB frame is translated by independently sampled integer x/y offsets in [-4,4] with edge padding, then brightness and contrast are independently scaled in [0.9,1.1]. The one augmented RGB frame feeds both teacher and student preprocessing branches. Labels, evaluation images, and the later EMA statistic refresh are unaugmented. Implementation: [augmentation](../turbovla/data/light_augmentation.py), [dataset integration](../turbovla/data/libero_lerobot.py), [parser](../turbovla/training/train_mixed.py).

The portable entry is [train_action_head.py](../experiments/libero/train_action_head.py) with the [fixed recipe](../experiments/libero/configs/action_head_variants.json). After setting up the dependencies, data and public initialization described in the [LIBERO setup guide](../experiments/libero/README.md), preview the expanded arguments before starting a new run:

```bash
python -m experiments.libero.train_action_head \
  --variant spike_driven_lightaug --world-size 8 --dry-run \
  --run_dir "$SPIKE_RUN_ROOT/spike_driven_lightaug_40k" \
  --vision_checkpoint "$SPIKE_VISION_CHECKPOINT" \
  --text_projection_init "$SPIKE_TEXT_INIT_DIR/text_projection.pth" \
  --max_steps 40000 --batch_size 32 --grad_accum_steps 1
```

The entry trains a fresh policy and does not automatically refresh EMA or launch evaluation. The reported LIBERO score used its own final EMA, refreshed on unaugmented training data for 128 × 16 samples with zero optimizer updates, then seed-7 FP32/highest four-suite evaluation. No trained policy, dataset, private path, GPU scheduler or credential is published here.

For public MT50, the same action-head design was adapted to one 256-pixel corner2 image, four state dimensions and four continuous actions, with an independent fixed ANN teacher and public-data statistics. The 40k result comes from that separate run and fixed public 2,500-episode protocol. **The Meta-World training/evaluation adapter is not part of this LIBERO-focused release entry**, so the MT50 benchmark is a documented result, not a claim that the LIBERO command reproduces it. The two benchmark source summaries and exact selected-run metadata are named in the machine-readable result file.
