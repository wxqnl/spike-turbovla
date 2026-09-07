"""Paths for the reproducible three-GPU LIBERO recipe.

Override SPIKE_* variables before starting a command. No host-specific path is
required, and importing this module does not allocate a GPU or launch a job.
"""
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("SPIKE_WORK_ROOT", CODE.parent)).expanduser().resolve()
ROOT = Path(os.environ.get("SPIKE_RUN_ROOT", WORK / "experiments/spike-turbovla/libero")).expanduser().resolve()
if ROOT.is_relative_to(CODE):
    raise ValueError("SPIKE_RUN_ROOT must be outside the code directory to keep source snapshots non-recursive")
RUN_NAME = os.environ.get("SPIKE_RUN_NAME", "cross_joint_seed42_bs128_3gpu_70k")
if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {".", ".."}:
    raise ValueError("SPIKE_RUN_NAME must be one directory name")
CONTROL = ROOT / (RUN_NAME + "_control")
RUN = ROOT / RUN_NAME
CALIBRATION = ROOT / (RUN_NAME + "_text_init")
PREFLIGHT = ROOT / (RUN_NAME + "_preflight")
EVALUATION = ROOT / (RUN_NAME + "_eval_200")
PYTHON = sys.executable

def asset(variable, default):
    return str(Path(os.environ.get(variable, default)).expanduser().resolve())

BERT = asset("SPIKE_BERT_PATH", WORK / "pretrained/TurboVLA/backbones/bert-base-uncased")
DINO = asset("SPIKE_DINO_PATH", WORK / "pretrained/TurboVLA/backbones/dinov3-vitb16")
SMOOTH = asset("SPIKE_TEXT_PATH", WORK / "pretrained/SmoothSpike/smoothspike-bert-base-fused")
VISION = asset("SPIKE_VISION_CHECKPOINT", WORK / "pretrained/SDT-V3/V3_19.0M_1x4.pth")
TEACHER = asset("SPIKE_TEACHER_CHECKPOINT", WORK / "pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth")
INIT = asset("SPIKE_GROUNDING_CHECKPOINT", WORK / "pretrained/GroundingDINO/groundingdino_swint_ogc.pth")
LIBERO_ROOT = asset("SPIKE_LIBERO_ROOT", WORK / "benchmarks/LIBERO")
LIBERO_CONFIG = asset("LIBERO_CONFIG_PATH", Path(LIBERO_ROOT) / ".libero")
DATA = Path(asset("SPIKE_DATA_ROOT", WORK / "datasets/libero"))
DATASETS = [str(DATA / (suite + "_no_noops_1.0.0_lerobot"))
            for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial")]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
GPU_IDS = tuple(int(value.strip()) for value in os.environ.get("SPIKE_CUDA_DEVICES", "4,5,6").split(","))
if len(GPU_IDS) != 3 or len(set(GPU_IDS)) != 3 or min(GPU_IDS) < 0:
    raise ValueError("The fixed global-batch-384 recipe requires three distinct physical GPU indices")
GPUS = ",".join(map(str, GPU_IDS))
