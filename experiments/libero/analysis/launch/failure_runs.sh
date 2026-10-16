#!/usr/bin/env bash
# LIBERO evaluation with per-step simulator records (failure_rollout.py), official protocol.
# One process per task runs its 50 episodes in order, as in the official runs: episode order is part of the protocol
# (each hard reset redraws the scene model from the per-task RNG). env.sh raises the open-file limit, which the
# mesh files leaked by every hard reset would otherwise exhaust at the 21st episode.
# Usage: failure_runs.sh <out_dir> <parallel> <suite,...> <name=checkpoint:raw|ema> [...]
#   CHUNKS="0:2" TASKS="0 3" failure_runs.sh ...   # quick check: initial states 0-1 of tasks 0 and 3
set -uo pipefail
source "${SPIKE_ENV:-$(dirname "$0")/env.sh}"
cd "$SPIKE_CODE_ROOT"
out=$1 parallel=$2 suites=$3
shift 3
chunks=${CHUNKS:-"0:50"}
mkdir -p "$out"

job() {  # name checkpoint weights suite task start count gpu
  local name=$1 ckpt=$2 weights=$3 suite=$4 task=$5 start=$6 count=$7 gpu=$8
  local dir=$out/$name/$suite stem
  stem=$dir/task$(printf %02d "$task")_e$(printf %02d "$start")
  mkdir -p "$dir"
  [ -f "$stem.pkl" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu LIBERO_CONFIG_PATH=$W/benchmarks/LIBERO/.libero \
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4 \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python "$SPIKE_CODE_ROOT/experiments/libero/analysis/failure_rollout.py" --record_path "$stem.pkl" --episode_start "$start" \
    --ckpt_path "$ckpt" --checkpoint_weights "$weights" \
    --task_ids "$task" --task_suite_name "$suite" --num_trials_per_task "$count" \
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
export out W SPIKE_CODE_ROOT SPIKE_DINOV3_B

IFS=, read -r -a gpus <<< "$SPIKE_GPUS"
n=0
for spec in "$@"; do
  name=${spec%%=*} rest=${spec#*=}
  ckpt=${rest%:*} weights=${rest##*:}
  for suite in ${suites//,/ }; do for task in ${TASKS:-$(seq 0 9)}; do for chunk in $chunks; do
    echo "$name $ckpt $weights $suite $task ${chunk%%:*} ${chunk##*:} ${gpus[$((n % ${#gpus[@]}))]}"
    n=$((n + 1))
  done; done; done
done | xargs -P "$parallel" -L 1 bash -c 'job "$@"' _

python - "$out" <<'EOF'
import json, sys
from collections import defaultdict
from pathlib import Path
out = Path(sys.argv[1]); total = defaultdict(lambda: [0, 0])
for f in sorted(out.glob("*/*/task*_e*.json")):
    r = json.loads(f.read_text())
    key = (f.parts[-3], f.parts[-2])
    total[key][0] += r["total_successes"]; total[key][1] += r["total_episodes"]
missing = [str(f) for f in out.glob("*/*/task*_e*.stdout.log") if not f.with_suffix("").with_suffix(".json").exists()]
summary = {f"{m}/{s}": {"successes": v[0], "episodes": v[1]} for (m, s), v in sorted(total.items())}
for k, v in summary.items():
    print(f"{k:40s} {v['successes']:4d}/{v['episodes']:4d}")
print("chunks without result:", len(missing))
(out / "summary.json").write_text(json.dumps({"suites": summary, "chunks_without_result": missing}, indent=1))
EOF
echo "$(date -Is) done $out"
