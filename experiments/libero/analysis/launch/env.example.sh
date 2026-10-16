# Environment for the launchers in this directory. Copy to env.sh beside them (or point SPIKE_ENV at a copy) and set
# SPIKE_WORK_ROOT. Expected layout under it: code/ (this repository), benchmarks/LIBERO (with its .libero config),
# benchmarks/LIBERO-plus, datasets/libero, pretrained/TurboVLA/backbones/{bert-base-uncased,dinov3-vitb16}.
W=${SPIKE_WORK_ROOT:?set SPIKE_WORK_ROOT}
export SPIKE_WORK_ROOT=$W
export SPIKE_CODE_ROOT=${SPIKE_CODE_ROOT:-$W/code}
export SPIKE_DINOV3_B=${SPIKE_DINOV3_B:-$W/pretrained/TurboVLA/backbones/dinov3-vitb16}
export PYTHONPATH=$SPIKE_CODE_ROOT:$SPIKE_CODE_ROOT/third_party/vla_adapter:$SPIKE_CODE_ROOT/third_party/starvla_runtime
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset NVIDIA_TF32_OVERRIDE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
# Every LIBERO hard reset leaks ~50 open mesh files; at the default limit of 1024 the 21st reset fails and LIBERO's
# reset loop retries forever. 50 episodes per process need ~2600.
ulimit -n 65536
# GPUs the launchers spread their processes over.
export SPIKE_GPUS=${SPIKE_GPUS:-0,1,2,3}
