"""Environment-backed paths and recipe shared by the pipeline and its children."""
import os
import sys
from pathlib import Path

from experiments.libero.recipes import PRESETS, VISION_CHOICES, TEXT_CHOICES, CROSS_CHOICES


def positive_int(variable, default):
    value = int(os.environ.get(variable, default))
    if value < 1:
        raise ValueError(f"{variable} must be positive")
    return value


PRESET = os.environ.get("SPIKE_PRESET", "vision_text_cross")
if PRESET not in PRESETS:
    raise ValueError(f"SPIKE_PRESET must be one of {tuple(PRESETS)}")
VISION_TYPE = os.environ.get("SPIKE_VISION", PRESETS[PRESET][0])
TEXT_TYPE = os.environ.get("SPIKE_TEXT", PRESETS[PRESET][1])
CROSS_TYPE = os.environ.get("SPIKE_CROSS", PRESETS[PRESET][2])
for name, value, choices in (
    ("SPIKE_VISION", VISION_TYPE, VISION_CHOICES),
    ("SPIKE_TEXT", TEXT_TYPE, TEXT_CHOICES),
    ("SPIKE_CROSS", CROSS_TYPE, CROSS_CHOICES),
):
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}")
MODULES = {"vision": VISION_TYPE, "text": TEXT_TYPE, "cross": CROSS_TYPE}
VARIANT = "_".join(name for name, enabled in (
    ("vision", VISION_TYPE == "sdtv3_19m"),
    ("text", TEXT_TYPE == "smoothspike_bert"),
    ("cross", CROSS_TYPE == "spike_sdsa"),
) if enabled) or "ann"
BATCH_SIZE = positive_int("SPIKE_BATCH_SIZE", 128)
GRAD_ACCUM = positive_int("SPIKE_GRAD_ACCUM_STEPS", 1)
MAX_STEPS = positive_int("SPIKE_MAX_STEPS", 70000)
SEED = int(os.environ.get("SPIKE_SEED", "42"))
EVAL_TRIALS = positive_int("SPIKE_EVAL_TRIALS", 5)
CHECKPOINT_PREFIX = "spike_turbovla"

CODE = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("SPIKE_WORK_ROOT", CODE.parent)).expanduser().resolve()
ROOT = Path(os.environ.get("SPIKE_RUN_ROOT", WORK / "experiments/spike-turbovla/libero")).expanduser().resolve()
if ROOT.is_relative_to(CODE):
    raise ValueError("SPIKE_RUN_ROOT must be outside the code directory to keep source snapshots non-recursive")
RUN_NAME = os.environ.get(
    "SPIKE_RUN_NAME", f"{VARIANT}_seed{SEED}_bs{BATCH_SIZE}_acc{GRAD_ACCUM}_3gpu_{MAX_STEPS}steps")
if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {".", ".."}:
    raise ValueError("SPIKE_RUN_NAME must be one directory name")
CONTROL = ROOT / (RUN_NAME + "_control")
RUN = ROOT / RUN_NAME
CALIBRATION = ROOT / (RUN_NAME + "_text_init")
PREFLIGHT = ROOT / (RUN_NAME + "_preflight")
EVALUATION = ROOT / (RUN_NAME + f"_eval_{40 * EVAL_TRIALS}")
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
    raise ValueError("The pipeline requires three distinct physical GPU indices")
GPUS = ",".join(map(str, GPU_IDS))
GLOBAL_BATCH = BATCH_SIZE * GRAD_ACCUM * len(GPU_IDS)
