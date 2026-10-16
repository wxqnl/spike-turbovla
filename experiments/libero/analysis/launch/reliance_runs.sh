#!/usr/bin/env bash
# State vs vision reliance (reliance_policy.py) for the spiking base and TurboVLA, one GPU each.
# Usage: OUT=<out_dir> reliance_runs.sh name=checkpoint:weights:gpu [...]
set -uo pipefail
source "${SPIKE_ENV:-$(dirname "$0")/env.sh}"
cd "$SPIKE_CODE_ROOT"
out=${OUT:?set OUT to the output directory}
script=${SCRIPT:-reliance_policy.py}  # SCRIPT=token_position_probe.py OUT=... for the token position probe
mkdir -p "$out"
specs=("$@")
[ ${#specs[@]} -gt 0 ] || { echo "usage: OUT=<dir> $0 name=checkpoint:weights:gpu [...]"; exit 1; }
for spec in "${specs[@]}"; do
  name=${spec%%=*} rest=${spec#*=}
  IFS=: read -r ckpt weights gpu <<< "$rest"
  CUDA_VISIBLE_DEVICES=$gpu LIBERO_CONFIG_PATH=$W/benchmarks/LIBERO/.libero MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  OMP_NUM_THREADS=4 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python "$SPIKE_CODE_ROOT/experiments/libero/analysis/$script" --data_root "$W/datasets/libero" --out "$out/$name.json" \
    --ckpt_path "$ckpt" --checkpoint_weights "$weights" --task_suite_name libero_spatial \
    --libero_root "$W/benchmarks/LIBERO" --dinov3_path "$SPIKE_DINOV3_B" \
    --bert_path "$W/pretrained/TurboVLA/backbones/bert-base-uncased" \
    --stats_path "$SPIKE_CODE_ROOT/experiments/libero/configs/libero_all4_stats.json" \
    --stats_key libero_all4_no_noops --normalize_binary_gripper false --seed 7 --precision fp32 \
    --float32_matmul_precision highest --control_mode relative --chunk_size 12 --num_open_loop_steps 12 \
    --num_steps_wait 10 --env_img_res 256 --log_path "$out/$name.log" > "$out/$name.stdout.log" 2>&1 &
done
wait
echo "$(date -Is) $script done"
