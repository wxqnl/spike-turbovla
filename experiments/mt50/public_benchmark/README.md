# Meta-World MT50 public benchmark

This directory contains the single-view `corner2` evaluation used for the 70k training-data EMA model. The fixed protocol is 50 native tasks × 50 seeded episodes, 500 native steps maximum, and 12-step open-loop action chunks.

- `evaluate.py`: four-worker controller, fixed episode manifest, completion checks.
- `adapter.py`: image/state preprocessing, action inverse normalization, native MT1 environment.
- `rollout.py`: one complete episode without retrying a valid failure.
- `episode_store.py`: immutable episode receipts and safe resume.
- `policy.py`: strict EMA/model/numerics binding.
- `task_mapping.json`: native task IDs and the training instructions.
- `model_source/`: model source copied from the evaluated training snapshot, separate from other project variants.

The original completed 70k run, EMA/checkpoint, and `dataset/complete.json` are external assets; no weights or dataset files are included. Set `MT50_RUN_ROOT`, `MT50_EMA`, `MT50_SOURCE_CHECKPOINT`, `MT50_DATA_COMPLETE`, and `MT50_MODEL_SOURCE` (this directory's `model_source`). Set `MT50_TRAINING_SOURCE` to the source path recorded in the EMA metadata if different from `MT50_MODEL_SOURCE`. Set `MT50_GPU_MAP` to a JSON map of four physical GPU indices to their current UUIDs. `MT50_EVAL_OUT` optionally selects a fresh output directory.

Run `python experiments/mt50/public_benchmark/evaluate.py --approve-evaluation` from the repository root. A failed run can resume only with `--resume`; committed episodes, including unsuccessful ones, are never rerun.
