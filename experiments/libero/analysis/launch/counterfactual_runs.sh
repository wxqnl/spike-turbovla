#!/usr/bin/env bash
# Plate-shift counterfactuals (counterfactual_rollout.py) on LIBERO-Spatial.
# Usage: counterfactual_runs.sh <out_dir> <trials per task> <parallel> <name=checkpoint:raw|ema> [...]
#   TASKS="5" counterfactual_runs.sh ...   # subset of tasks
#   SCRIPT=head_probe.py counterfactual_runs.sh ...   # check 1b: open-loop probes with action-head internals
set -uo pipefail
source "${SPIKE_ENV:-$(dirname "$0")/env.sh}"
cd "$SPIKE_CODE_ROOT"
out=$1 trials=$2 parallel=$3
shift 3
mkdir -p "$out"

job() {  # name checkpoint weights task gpu
  local name=$1 ckpt=$2 weights=$3 task=$4 gpu=$5
  local stem=$out/$name/task$(printf %02d "$task")
  mkdir -p "$out/$name"
  [ -f "$stem.pkl" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu LIBERO_CONFIG_PATH=$W/benchmarks/LIBERO/.libero \
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4 \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python "$SPIKE_CODE_ROOT/experiments/libero/analysis/${SCRIPT:-counterfactual_rollout.py}" --record_path "$stem.pkl" \
    --ckpt_path "$ckpt" --checkpoint_weights "$weights" \
    --task_ids "$task" --task_suite_name libero_spatial --num_trials_per_task "$trials" \
    --libero_root "$W/benchmarks/LIBERO" --dinov3_path "$SPIKE_DINOV3_B" \
    --bert_path "$W/pretrained/TurboVLA/backbones/bert-base-uncased" \
    --stats_path "$SPIKE_CODE_ROOT/experiments/libero/configs/libero_all4_stats.json" \
    --stats_key libero_all4_no_noops --normalize_binary_gripper false \
    --seed 7 --precision fp32 --float32_matmul_precision highest \
    --control_mode relative --chunk_size 12 --num_open_loop_steps 12 --num_steps_wait 10 \
    --env_img_res 256 --mujoco_gl osmesa --pyopengl_platform osmesa --save_video false \
    --result_json_path "$stem.json" --log_path "$stem.log" > "$stem.stdout.log" 2>&1
}
export -f job
export out W SPIKE_CODE_ROOT SPIKE_DINOV3_B trials SCRIPT

IFS=, read -r -a gpus <<< "$SPIKE_GPUS"
n=0
for spec in "$@"; do
  name=${spec%%=*} rest=${spec#*=}
  ckpt=${rest%:*} weights=${rest##*:}
  for task in ${TASKS:-$(seq 0 9)}; do
    echo "$name $ckpt $weights $task ${gpus[$((n % ${#gpus[@]}))]}"
    n=$((n + 1))
  done
done | xargs -P "$parallel" -L 1 bash -c 'job "$@"' _
echo "$(date -Is) done $out"
