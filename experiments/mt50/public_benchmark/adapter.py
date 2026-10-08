"""Public MT50 observation/action and real environment adapter.

No policy loading, device selection, rollout budget, or controller side effects.
Reference: LeRobot main envs/metaworld.py, inspected 2026-09-21.
Keep this separate from the running training source.
"""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA_COMPLETE = Path(os.environ["MT50_DATA_COMPLETE"]).resolve()


class PublicObservationAdapter:
    def __init__(self):
        self.info = json.loads(DATA_COMPLETE.read_text())
        mapping = json.loads((HERE / "task_mapping.json").read_text())
        self.tasks = {row["native_task_id"]: row for row in mapping["tasks"]}
        if set(self.tasks) != set(range(50)):
            raise ValueError("Expected all 50 native task IDs")
        # Match mt_data's NumPy promotion with JSON-list statistics, followed
        # by its final float32 cast.
        self.mean = np.asarray(self.info["state_stats"]["mean"], dtype=np.float64)
        self.std = np.asarray(self.info["state_stats"]["std"], dtype=np.float64)
        self.minimum = np.asarray(self.info["action_stats"]["min"], dtype=np.float32)
        self.maximum = np.asarray(self.info["action_stats"]["max"], dtype=np.float32)
        self.span = self.maximum - self.minimum
        if any(x.shape != (4,) for x in (self.mean, self.std, self.minimum, self.maximum)):
            raise ValueError("Expected this experiment's four-dimensional statistics")
        if not np.all(self.std > 0) or not np.all(self.span > 0):
            raise ValueError("Invalid training normalization")
        for row in self.tasks.values():
            if row["instruction"] != self.info["texts"][str(row["text_index"])]:
                raise ValueError("Training language mapping changed")

    def observation(self, image, raw_obs, native_task_id):
        image = np.asarray(image)
        state = np.asarray(raw_obs[:4], dtype=np.float32)
        if image.shape != (480, 480, 3) or image.dtype != np.uint8:
            raise ValueError("Expected wrapper-oriented native 480 RGB uint8 image")
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("Nonfinite or malformed native state")
        resized = np.asarray(Image.fromarray(image).resize((256, 256), Image.Resampling.BILINEAR))
        pixels = np.ascontiguousarray(resized.transpose(2, 0, 1)[None])
        state = ((state - self.mean) / self.std).astype(np.float32)
        return self.tasks[native_task_id]["instruction"], pixels, state

    def tensors(self, image, raw_obs, native_task_id, device):
        import torch
        instruction, pixels, state = self.observation(image, raw_obs, native_task_id)
        x = torch.from_numpy(pixels).unsqueeze(0).to(device).float() / 255.
        mean = x.new_tensor((.485, .456, .406)).view(1, 1, 3, 1, 1)
        std = x.new_tensor((.229, .224, .225)).view(1, 1, 3, 1, 1)
        return [instruction], (x - mean) / std, torch.from_numpy(state).unsqueeze(0).to(device)

    def native_action(self, normalized_action):
        value = np.asarray(normalized_action, dtype=np.float32)
        if value.shape != (4,) or not np.isfinite(value).all():
            raise ValueError("Nonfinite or malformed predicted action")
        if np.max(np.abs(value)) > 1.00001:
            raise ValueError("Prediction violates the trained tanh output range")
        # Restore the pre-clipping expert command range. Native MetaWorld applies
        # its own control limits; do not clip or binarize the gripper here.
        return ((value + 1.) * .5 * self.span + self.minimum).astype(np.float32)


class PublicMT50Environment:
    def __init__(self, task, benchmark=None):
        import metaworld
        self.native_task_id = int(task["native_task_id"])
        self.name = task["env_name"]
        benchmark = benchmark or metaworld.MT1(self.name, seed=42)
        self.env = benchmark.train_classes[self.name](
            render_mode="rgb_array", camera_name="corner2", width=480, height=480)
        self.env.set_task(benchmark.train_tasks[0])
        self.env.model.cam_pos[2] = [.75, .075, .7]
        self.env.reset()
        self.env._freeze_rand_vec = False
        self.env.seeded_rand_vec = True
        if self.env.frame_skip != 5 or not np.isclose(self.env.dt, 1 / 80):
            self.close()
            raise ValueError("Unexpected native physical timestep; do not compensate with repeats")
        self.steps = 0

    def reset(self, seed):
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("An explicit nonnegative episode seed is required")
        self.env.seed(seed)
        raw, info = self.env.reset(seed=seed)
        self.steps = 0
        return raw, info

    def image(self):
        image = self.env.render()
        if image is None or image.shape != (480, 480, 3):
            raise RuntimeError("Native rendering failed")
        if not np.isfinite(image).all() or image.std() <= 1:
            raise RuntimeError("Black or invalid native image; never evaluate on it")
        return np.flip(image, (0, 1)).copy()

    def step(self, native_action):
        if self.steps >= 500:
            raise RuntimeError("Episode exceeds the native 500-step horizon")
        raw, reward, done, truncated, info = self.env.step(native_action)
        self.steps += 1
        info = dict(info)
        success = bool(info["success"])
        info["is_success"] = success
        # The official wrapper's post-terminal auto-reset is unnecessary here:
        # the rollout loop explicitly resets before the next episode.
        return raw, reward, bool(done or success), bool(truncated or self.steps == 500), info

    def close(self):
        self.env.close()
