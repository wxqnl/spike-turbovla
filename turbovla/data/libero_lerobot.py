"""Native LeRobot v2.1 input pipeline for the four LIBERO no-noop suites."""

from __future__ import annotations

import json
import os
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import get_worker_info
from transformers import AutoImageProcessor

try:
    import av
except ImportError:
    av = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from .libero_rlds import LiberoRLDSDataset, vla_collate_fn
from .suite_stats import _select_stats, _vector


def _split_dataset_dirs(
    dataset_dirs: str | os.PathLike[str] | list[str] | tuple[str, ...],
) -> list[Path]:
    if isinstance(dataset_dirs, (list, tuple)):
        parts = [str(item).strip() for item in dataset_dirs]
    else:
        normalized = str(dataset_dirs).replace("\n", ",").replace(";", ",")
        parts = [item.strip() for item in normalized.split(",")]
    paths = [Path(item).expanduser() for item in parts if item]
    if not paths:
        raise ValueError("dataset_dirs must contain at least one LeRobot dataset directory")
    return paths


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def _parse_split(spec: str) -> tuple[int, int]:
    fields = spec.split(":")
    if len(fields) != 2 or not all(field.strip().isdigit() for field in fields):
        raise ValueError(f"Unsupported LeRobot split specification: {spec!r}")
    start, stop = (int(field) for field in fields)
    if start < 0 or stop <= start:
        raise ValueError(f"Invalid LeRobot split range: {spec!r}")
    return start, stop


def _decode_video_frames(path: Path) -> np.ndarray:
    if av is None:
        raise ImportError("PyAV is required for LeRobot video decoding; install av>=12")
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot video not found: {path}")

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        width, height = stream.codec_context.width, stream.codec_context.height
        # DataLoader workers already provide parallelism across videos.
        stream.codec_context.thread_count = 1
        try:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        except av.error.FFmpegError as exc:
            # Some AV1 streams fail with the libdav1d bundled with PyAV while
            # system FFmpeg decodes every frame. Restart the entire stream;
            # never mix a partial decode with the fallback or drop an episode.
            container.close()
            decoded = _decode_video_frames_ffmpeg(path, width=width, height=height)
            warnings.warn(
                f"PyAV could not decode {path}: {exc}; decoded the complete "
                f"video with system FFmpeg ({len(decoded)} frames).",
                RuntimeWarning,
                stacklevel=2,
            )
            return decoded
    finally:
        container.close()
    if not frames:
        raise ValueError(f"LeRobot video contains no decodable frames: {path}")
    return np.stack(frames, axis=0)


def _decode_video_frames_ffmpeg(path: Path, *, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions {width}x{height}: {path}")
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-xerror", "-err_detect", "explode", "-threads", "1",
        "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
        # Preserve every decoded frame; do not duplicate/drop frames to fit fps.
        "-vsync", "0", "-pix_fmt", "rgb24", "-f", "rawvideo",
        "-threads", "1", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True, timeout=120)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"PyAV failed to decode {path} and the FFmpeg fallback is not installed"
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg also failed to decode {path}: {details}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"FFmpeg decoding exceeded 120 seconds: {path}") from exc
    # With loglevel=error, stderr indicates a decode error even if a particular
    # FFmpeg build returns zero. Do not accept concealed/corrupt frames.
    if result.stderr.strip():
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg reported an error decoding {path}: {details}")
    bytes_per_frame = height * width * 3
    if not result.stdout or len(result.stdout) % bytes_per_frame:
        raise ValueError(
            f"Incomplete RGB frame data from FFmpeg for {path}: "
            f"{len(result.stdout)} bytes, expected a multiple of {bytes_per_frame}"
        )
    # The caller checks the frame count against the aligned Parquet table.
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


@dataclass(frozen=True)
class _SuiteSpec:
    root: Path
    data_pattern: str
    video_pattern: str
    episodes_per_chunk: int
    primary_video_key: str
    wrist_video_key: str
    tasks: dict[int, str]


@dataclass(frozen=True)
class _EpisodeSpec:
    suite_index: int
    episode_index: int
    expected_length: int
    fallback_instruction: str


class LiberoMixedLeRobotDataset(LiberoRLDSDataset):
    """Stream aligned Parquet actions/states and the two LeRobot camera videos."""

    def __init__(
        self,
        dataset_dir,
        *args,
        dataset_dirs: str | os.PathLike[str] | list[str] | tuple[str, ...],
        stats_path: str | os.PathLike[str],
        stats_key: str | None = None,
        normalize_binary_gripper: str = "auto",
        gripper_convention: str,
        LOCAL_DINOV3_PATH: str,
        rank: int = 0,
        world_size: int = 1,
        chunk_size: int = 8,
        split: str = "train",
        shuffle_buffer: int = 512,
        shuffle_steps_within_episode: bool = False,
        step_mix_buffer_size: int = 0,
        seed: int = 42,
        local_files_only: bool = True,
        expected_image_size: int = 256,
        vision_encoder_type: str = "dinov3",
        vision_image_size: int | None = None,
        enable_dinov3_teacher: bool = False,
        light_augmentation: bool = False,
        **kwargs,
    ):
        del dataset_dir, args, kwargs
        if av is None or pq is None:
            missing = [name for name, module in (("av", av), ("pyarrow", pq)) if module is None]
            raise ImportError(f"LeRobot input requires: {', '.join(missing)}")
        if rank < 0 or world_size <= 0 or rank >= world_size:
            raise ValueError(f"Invalid distributed shard rank={rank}, world_size={world_size}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.light_augmentation = bool(light_augmentation)
        if self.light_augmentation and split != "train":
            raise ValueError("Light augmentation is training-only")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.chunk_size = int(chunk_size)
        self.split = split
        self.shuffle_buffer = int(shuffle_buffer)
        self.shuffle_steps_within_episode = bool(shuffle_steps_within_episode)
        self.step_mix_buffer_size = int(step_mix_buffer_size)
        self.seed = int(seed)
        self.expected_image_size = int(expected_image_size)
        self.vision_encoder_type = str(vision_encoder_type)
        self.vision_image_size = int(vision_image_size or expected_image_size)
        self.enable_dinov3_teacher = bool(enable_dinov3_teacher)
        if self.vision_encoder_type not in {"dinov3", "sdtv3_19m"}:
            raise ValueError(f"unsupported vision encoder type: {self.vision_encoder_type}")
        self.dataset_dirs = _split_dataset_dirs(dataset_dirs)

        self.dino_processor = AutoImageProcessor.from_pretrained(
            LOCAL_DINOV3_PATH,
            local_files_only=bool(local_files_only),
        )
        self._disable_spatial_resize(self.dino_processor)

        payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        stats = _select_stats(payload, stats_key)
        state_section = "proprio" if "proprio" in stats else "state"
        self.proprio_mean = _vector(stats, state_section, "mean")
        self.proprio_std = _vector(stats, state_section, "std")
        self.action_min = _vector(stats, "action", "min")
        self.action_max = _vector(stats, "action", "max")
        if gripper_convention not in {"binary_open", "signed_close"}:
            raise ValueError("LeRobot gripper_convention must be binary_open or signed_close")
        if normalize_binary_gripper not in {"auto", "false", "0", "no"}:
            raise ValueError(
                "LeRobot uses an explicit input gripper convention; do not also enable "
                "the legacy --normalize_binary_gripper transform"
            )
        if float(self.action_min[6]) != -1.0 or float(self.action_max[6]) != 1.0:
            raise ValueError("LeRobot targets use signed LIBERO grippers; stats must specify [-1, +1]")
        self.gripper_convention = gripper_convention

        self.suites: list[_SuiteSpec] = []
        self.episodes: list[_EpisodeSpec] = []
        for root in self.dataset_dirs:
            self._register_suite(root)
        if not self.episodes:
            raise ValueError(f"No episodes found for split {split!r}")

        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "mixed_lerobot="
                f"dirs={[str(path) for path in self.dataset_dirs]}, split={self.split}, "
                f"episodes={len(self.episodes)}, stats={stats_path}:{stats_key}, "
                f"input_gripper_convention={self.gripper_convention}, "
                "target_gripper_convention=signed_close (-1=open, +1=close)",
                flush=True,
            )

    @staticmethod
    def _disable_spatial_resize(processor):
        if hasattr(processor, "do_resize"):
            processor.do_resize = False
        if hasattr(processor, "do_center_crop"):
            processor.do_center_crop = False

    def _register_suite(self, root: Path) -> None:
        info_path = root / "meta" / "info.json"
        modality_path = root / "meta" / "modality.json"
        tasks_path = root / "meta" / "tasks.jsonl"
        episodes_path = root / "meta" / "episodes.jsonl"
        for required in (info_path, modality_path, tasks_path, episodes_path):
            if not required.is_file():
                raise FileNotFoundError(f"Incomplete LeRobot dataset; missing {required}")

        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("codebase_version") != "v2.1":
            raise ValueError(
                f"Expected LeRobot v2.1 at {root}, got {info.get('codebase_version')!r}"
            )
        splits = info.get("splits", {})
        if self.split not in splits:
            raise KeyError(f"Split {self.split!r} not found at {root}; available={sorted(splits)}")
        split_start, split_stop = _parse_split(str(splits[self.split]))

        modality = json.loads(modality_path.read_text(encoding="utf-8"))
        video_meta = modality.get("video", {})
        try:
            primary_key = str(video_meta["primary_image"]["original_key"])
            wrist_key = str(video_meta["wrist_image"]["original_key"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"LeRobot dataset must define primary_image and wrist_image in {modality_path}") from exc

        tasks = {
            int(record["task_index"]): str(record["task"])
            for record in _read_jsonl(tasks_path)
        }
        suite = _SuiteSpec(
            root=root,
            data_pattern=str(info["data_path"]),
            video_pattern=str(info["video_path"]),
            episodes_per_chunk=int(info.get("chunks_size", info.get("chunk_size", 1000))),
            primary_video_key=primary_key,
            wrist_video_key=wrist_key,
            tasks=tasks,
        )
        suite_index = len(self.suites)
        self.suites.append(suite)

        registered = 0
        for record in _read_jsonl(episodes_path):
            episode_index = int(record["episode_index"])
            if not split_start <= episode_index < split_stop:
                continue
            instructions = record.get("tasks") or []
            fallback = str(instructions[0]) if instructions else ""
            self.episodes.append(
                _EpisodeSpec(
                    suite_index=suite_index,
                    episode_index=episode_index,
                    expected_length=int(record["length"]),
                    fallback_instruction=fallback,
                )
            )
            registered += 1
        expected = split_stop - split_start
        if registered != expected:
            raise ValueError(f"Split {self.split!r} at {root} declares {expected} episodes, found {registered}")

    @staticmethod
    def _format_episode_path(suite: _SuiteSpec, pattern: str, episode_index: int, **kwargs) -> Path:
        episode_chunk = episode_index // suite.episodes_per_chunk
        relative = pattern.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            **kwargs,
        )
        return suite.root / relative

    def _load_episode(self, episode: _EpisodeSpec) -> dict[str, np.ndarray | list[str]]:
        suite = self.suites[episode.suite_index]
        parquet_path = self._format_episode_path(suite, suite.data_pattern, episode.episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"LeRobot episode table not found: {parquet_path}")
        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "action", "frame_index", "task_index"],
        )
        states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        frame_indices = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
        task_indices = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)

        gripper = actions[:, 6]
        allowed = (0.0, 1.0) if self.gripper_convention == "binary_open" else (-1.0, 1.0)
        if not np.isin(gripper, allowed).all():
            raise ValueError(
                f"Gripper values in {parquet_path} do not match {self.gripper_convention}: "
                f"expected {allowed}, found {np.unique(gripper)[:8].tolist()}"
            )
        if self.gripper_convention == "binary_open":
            # RLDS-derived LeRobot: 1=open, 0=close. The teacher and LIBERO
            # evaluator use -1=open, +1=close. Convert exactly once per episode.
            actions[:, 6] = 1.0 - 2.0 * gripper

        length = int(table.num_rows)
        if length != episode.expected_length:
            raise ValueError(
                f"Episode {episode.episode_index} at {suite.root} has {length} rows; "
                f"metadata declares {episode.expected_length}"
            )
        order = np.argsort(frame_indices, kind="stable")
        if not np.array_equal(frame_indices[order], np.arange(length)):
            raise ValueError(f"Non-contiguous frame_index in {parquet_path}")
        states = states[order]
        actions = actions[order]
        task_indices = task_indices[order]

        primary_path = self._format_episode_path(
            suite,
            suite.video_pattern,
            episode.episode_index,
            video_key=suite.primary_video_key,
        )
        wrist_path = self._format_episode_path(
            suite,
            suite.video_pattern,
            episode.episode_index,
            video_key=suite.wrist_video_key,
        )
        primary_frames = _decode_video_frames(primary_path)
        wrist_frames = _decode_video_frames(wrist_path)
        if len(primary_frames) != length or len(wrist_frames) != length:
            raise ValueError(
                f"Video/table length mismatch for episode {episode.episode_index}: "
                f"table={length}, primary={len(primary_frames)}, wrist={len(wrist_frames)}"
            )

        instructions = []
        for task_index in task_indices:
            instruction = suite.tasks.get(int(task_index), episode.fallback_instruction)
            if not instruction:
                raise KeyError(f"Unknown task_index={task_index} in {parquet_path}")
            instructions.append(instruction)
        return {
            "states": states,
            "actions": actions,
            "primary_frames": primary_frames,
            "wrist_frames": wrist_frames,
            "instructions": instructions,
        }

    def _process_image(self, image: np.ndarray) -> dict[str, torch.Tensor]:
        # Augment once before the teacher and student normalization branches.
        if self.light_augmentation:
            from .light_augmentation import light_augment_rgb
            image = light_augment_rgb(image)
        return self._process_image_pair(image)

    def _build_step_sample(self, episode_data: dict, step: int):
        states = episode_data["states"]
        actions = episode_data["actions"]
        episode_length = len(actions)

        state = torch.as_tensor(states[step], dtype=torch.float32)
        state = (state - self.proprio_mean) / (self.proprio_std + 1e-6)

        stop = min(step + self.chunk_size, episode_length)
        action_chunk_np = actions[step:stop]
        valid_length = len(action_chunk_np)
        if valid_length < self.chunk_size:
            padding = np.repeat(action_chunk_np[-1:], self.chunk_size - valid_length, axis=0)
            action_chunk_np = np.concatenate([action_chunk_np, padding], axis=0)
        action_chunk = torch.as_tensor(action_chunk_np.copy(), dtype=torch.float32)
        action_chunk[:, :6] = (
            2.0
            * (action_chunk[:, :6] - self.action_min[:6])
            / (self.action_max[:6] - self.action_min[:6] + 1e-6)
            - 1.0
        ).clamp(-1.0, 1.0)

        mask = torch.zeros(self.chunk_size, dtype=torch.float32)
        mask[:valid_length] = 1.0
        images = (
            self._process_image(episode_data["primary_frames"][step]),
            self._process_image(episode_data["wrist_frames"][step]),
        )
        return images, episode_data["instructions"][step], state, action_chunk, mask

    @staticmethod
    def _buffer_shuffle(items: list[_EpisodeSpec], size: int, rng: np.random.Generator) -> Iterator[_EpisodeSpec]:
        if size <= 1:
            yield from items
            return
        buffer: list[_EpisodeSpec] = []
        for item in items:
            if len(buffer) < size:
                buffer.append(item)
                continue
            index = int(rng.integers(0, len(buffer)))
            yield buffer[index]
            buffer[index] = item
        rng.shuffle(buffer)
        yield from buffer

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        shard_count = self.world_size * num_workers
        shard_index = self.rank + self.world_size * worker_id
        local_episodes = self.episodes[shard_index::shard_count]
        if not local_episodes:
            raise RuntimeError(
                f"Empty LeRobot shard {shard_index}/{shard_count}; reduce ranks/workers or add data"
            )

        base_seed = self.seed + 1009 * self.rank + 9176 * worker_id
        epoch = 0
        while True:
            rng = np.random.default_rng(base_seed + epoch)
            step_buffer = []
            episode_iterator = self._buffer_shuffle(local_episodes, self.shuffle_buffer, rng)
            for episode in episode_iterator:
                episode_data = self._load_episode(episode)
                step_indices = np.arange(episode.expected_length)
                if self.shuffle_steps_within_episode:
                    rng.shuffle(step_indices)
                for step in step_indices:
                    sample = self._build_step_sample(episode_data, int(step))
                    if self.step_mix_buffer_size > 0:
                        step_buffer.append(sample)
                        if len(step_buffer) >= self.step_mix_buffer_size:
                            output_index = int(rng.integers(0, len(step_buffer)))
                            yield step_buffer.pop(output_index)
                    else:
                        yield sample

            while step_buffer:
                output_index = int(rng.integers(0, len(step_buffer)))
                yield step_buffer.pop(output_index)
            epoch += 1


__all__ = ["LiberoMixedLeRobotDataset", "vla_collate_fn"]
