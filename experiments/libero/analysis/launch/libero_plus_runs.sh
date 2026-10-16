#!/usr/bin/env bash
# LIBERO-plus (arXiv 2510.13626) with per-step simulator records (libero_plus_rollout.py), the benchmark's
# protocol: every perturbed task runs once from its own initial state. Policy settings as in failure_runs.sh.
# Task ids come from the benchmark's task_classification.json (id - 1 is the suite index), filtered by category;
# each suite's selected tasks are split into chunks of CHUNK tasks, one process per chunk, tasks in id order.
# The checkout lives beside LIBERO ($W/benchmarks/LIBERO-plus, its own .libero config); its extra packages (wand,
# scikit-image) sit in $W/benchmarks/libero_plus_pydeps, on PYTHONPATH for these processes only.
# Usage: libero_plus_runs.sh <out_dir> <parallel> <category|all> <name=checkpoint:raw|ema> [...]
#   CHUNK=24 SUITES=libero_spatial,libero_goal libero_plus_runs.sh out 16 "Objects Layout" f7=...pth:ema
# A rerun skips chunks with a result json and resumes the others from their records. LIBERO-plus builds a new scene
# per task and leaks file handles: 22 tasks per process exhausted 65536, so this raises the limit where it can
# (root); elsewhere keep CHUNK at 12 or less.
# Where ImageMagick is not a system library, a conda prefix at $W/benchmarks/libero_plus_magick provides MagickWand
# for wand through MAGICK_HOME. Its libgobject needs the prefix's newer glib, but the process loads the system glib
# first, so that one library is preloaded.
set -uo pipefail
source "${SPIKE_ENV:-$(dirname "$0")/env.sh}"
ulimit -n 1048576 2>/dev/null || true
if [ -d "$W/benchmarks/libero_plus_magick" ]; then
  export MAGICK_HOME=$W/benchmarks/libero_plus_magick
  export LD_PRELOAD=$MAGICK_HOME/lib/libglib-2.0.so.0${LD_PRELOAD:+:$LD_PRELOAD}
fi
cd "$SPIKE_CODE_ROOT"
out=$1 parallel=$2 category=$3
shift 3
P=$W/benchmarks/LIBERO-plus
mkdir -p "$out"

job() {  # name checkpoint weights suite chunk_index task_ids gpu
  local name=$1 ckpt=$2 weights=$3 suite=$4 index=$5 ids=$6 gpu=$7
  local stem
  stem=$out/$name/$suite/chunk$(printf %03d "$index")
  mkdir -p "$(dirname "$stem")"
  [ -f "$stem.json" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu LIBERO_CONFIG_PATH=$P/.libero \
  PYTHONPATH=$W/benchmarks/libero_plus_pydeps${PYTHONPATH:+:$PYTHONPATH} \
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4 \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python "$SPIKE_CODE_ROOT/experiments/libero/analysis/libero_plus_rollout.py" --record_path "$stem.pkl" \
    --ckpt_path "$ckpt" --checkpoint_weights "$weights" \
    --task_ids "$ids" --task_suite_name "$suite" --num_trials_per_task 1 \
    --libero_root "$P" --dinov3_path "$SPIKE_DINOV3_B" \
    --bert_path "$W/pretrained/TurboVLA/backbones/bert-base-uncased" \
    --stats_path "$SPIKE_CODE_ROOT/experiments/libero/configs/libero_all4_stats.json" \
    --stats_key libero_all4_no_noops --normalize_binary_gripper false \
    --seed 7 --precision fp32 --float32_matmul_precision highest \
    --control_mode relative --chunk_size 12 --num_open_loop_steps 12 --num_steps_wait 10 \
    --env_img_res 256 --mujoco_gl osmesa --pyopengl_platform osmesa --save_video false \
    --result_json_path "$stem.json" --log_path "$stem.log" > "$stem.stdout.log" 2>&1
}
export -f job
export out W P SPIKE_CODE_ROOT SPIKE_DINOV3_B

chunks=$(python - "$P/libero/libero/benchmark/task_classification.json" "$category" \
  "${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}" "${CHUNK:-24}" <<'EOF'
import json
import sys
table, category, suites, size = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3].split(","), int(sys.argv[4])
for suite in suites:
    ids = [row["id"] - 1 for row in table[suite] if category == "all" or row["category"] == category]
    for start in range(0, len(ids), size):
        print(suite, start // size, ",".join(map(str, ids[start:start + size])))
EOF
)
# Models alternate chunk by chunk, so partial results already pair the models on the same tasks.
IFS=, read -r -a gpus <<< "$SPIKE_GPUS"
n=0
while read -r suite index ids; do
  for spec in "$@"; do
    name=${spec%%=*} rest=${spec#*=}
    ckpt=${rest%:*} weights=${rest##*:}
    echo "$name $ckpt $weights $suite $index $ids ${gpus[$((n % ${#gpus[@]}))]}"
    n=$((n + 1))
  done
done <<< "$chunks" | xargs -P "$parallel" -L 1 bash -c 'job "$@"' _

python "$SPIKE_CODE_ROOT/experiments/libero/analysis/libero_plus_summary.py" "$out" "$P/libero/libero/benchmark/task_classification.json"
