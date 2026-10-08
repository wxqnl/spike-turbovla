"""Strict final-EMA policy for the fixed public MT50 70k benchmark."""
import json
import os
import random
from pathlib import Path

import numpy as np

from adapter import PublicObservationAdapter

ROOT = Path(os.environ["MT50_RUN_ROOT"]).resolve()
EXT = ROOT / "continuation_70k_anneal_after_60k"
EMA = Path(os.environ["MT50_EMA"]).resolve()
SOURCE_CHECKPOINT = Path(os.environ["MT50_SOURCE_CHECKPOINT"]).resolve()
MODEL_SOURCE = Path(os.environ["MT50_MODEL_SOURCE"]).resolve()
TRAINING_SOURCE = os.environ.get("MT50_TRAINING_SOURCE", str(MODEL_SOURCE))
ALLOWED_UUIDS = set(json.loads(os.environ["MT50_GPU_MAP"]).values())


class FinalEMA70kPolicy:
    def __init__(self, seed=7):
        import torch

        if os.environ.get("CUDA_VISIBLE_DEVICES") not in ALLOWED_UUIDS:
            raise RuntimeError("Worker must expose exactly one configured GPU UUID")
        if (EXT / "overall.exit_code").read_text().strip() != "0":
            raise RuntimeError("Own 70k training/EMA success receipt is missing")
        for name in ("train", "refresh"):
            receipt = __import__("json").loads((EXT / "training" / f"{name}.success.json").read_text())
            if receipt["exit_code"] != 0 or Path(receipt["exit_file"]).read_text().strip() != "0":
                raise RuntimeError(f"Own 70k {name} receipt is not successful")
        if not EMA.is_file() or not SOURCE_CHECKPOINT.is_file():
            raise FileNotFoundError("Own 70k final EMA assets are missing")
        import turbovla
        source = MODEL_SOURCE
        if not Path(turbovla.__file__).resolve().is_relative_to(source.resolve()):
            raise RuntimeError("Incorrect experiment source package in evaluator")
        from turbovla.models.configuration import TurboVLAConfig
        from turbovla.models.turbovla import build_turbovla

        torch.set_num_threads(2)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_device(0)
        self.adapter = PublicObservationAdapter()
        ck = torch.load(EMA, map_location="cpu", mmap=True, weights_only=False)
        expected = {
            "global_step": 70000,
            "samples_seen": 17920000,
            "dataset_repo": "lerobot/metaworld_mt50",
            "dataset_revision": "v3.0",
            "source_snapshot": TRAINING_SOURCE,
        }
        if any(ck.get(key) != value for key, value in expected.items()):
            raise ValueError("The final 70k EMA provenance changed")
        if ck.get("numerics") != {
            "precision": "fp32",
            "float32_matmul_precision": "highest",
            "cuda_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }:
            raise ValueError("The final 70k numerical policy changed")
        if ck.get("state_stats") != self.adapter.info["state_stats"] or ck.get("action_stats") != self.adapter.info["action_stats"]:
            raise ValueError("The final 70k normalization metadata changed")
        ema = ck["wta_normalization"]["ema"]
        if (ck.get("export_kind") != "ema_evaluation_only"
                or ema["status"] != "refreshed_with_training_data"
                or ema["source"] != "training_data"
                or ema["source_checkpoint"] != str(SOURCE_CHECKPOINT)
                or ema["batches"] != 128 or ema["batch_size"] != 16
                or ema["optimizer_steps"] != 0):
            raise ValueError("The final 70k EMA refresh provenance changed")
        config = TurboVLAConfig.from_mapping(ck["model_config"])
        if (config.action.state_dim != 4 or config.action.action_dim != 4
                or config.action.horizon != 12 or config.vision.num_views != 1
                or config.vision.image_size != 256 or config.vision.compute_precision != "fp32"):
            raise ValueError("Wrong public MT50 model dimensions or precision")
        self.model = build_turbovla(config)
        self.model.load_state_dict(ck["model_state_dict"], strict=True)
        self.model = self.model.float().cuda().eval().requires_grad_(False)
        from turbovla.models.components.wta_operators import verify_running_norms
        verify_running_norms(self.model)
        self.ready = {
            "variant": "spike_driven_vla_ema70k",
            "source": TRAINING_SOURCE,
            "ema": str(EMA),
            "global_step": 70000,
            "samples_seen": 17920000,
            "dataset_repo": "lerobot/metaworld_mt50",
            "dataset_revision": "v3.0",
            "strict_load": True,
            "precision": "fp32",
            "matmul": "highest",
            "cuda_tf32": False,
            "cudnn_tf32": False,
            "policy_seed": seed,
            "ema_batches": 128,
            "ema_batch_size": 16,
            "ema_optimizer_steps": 0,
        }
        del ck

    def predict(self, image, raw_obs, task_id):
        import torch

        inputs = self.adapter.tensors(image, raw_obs, task_id, "cuda")
        with torch.inference_mode():
            chunk = self.model(*inputs)[0].detach().cpu().numpy()
        if chunk.shape != (12, 4) or not np.isfinite(chunk).all():
            raise RuntimeError("Invalid full-model action chunk")
        return chunk
