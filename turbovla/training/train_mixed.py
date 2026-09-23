#!/usr/bin/env python3
"""Mixed-suite LIBERO trainer with RLDS and native LeRobot input backends."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..data.libero_lerobot import LiberoMixedLeRobotDataset
from ..data.mixed_suite import LiberoMixedRLDSDataset
from . import pi05, trainer


def parse_args_with_mixed_suite_stats():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dinov3_precision",
        type=str,
        default="bf16_autocast",
        choices=["fp32", "bf16", "bf16_autocast"],
    )
    parser.add_argument("--libero_light_augmentation", action="store_true")
    parser.add_argument("--dataset_dirs", type=str)
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="auto",
        choices=["auto", "rlds", "lerobot"],
        help="Input format. auto detects TFDS dataset_info.json or LeRobot meta/info.json.",
    )
    parser.add_argument("--stats_path", type=str)
    parser.add_argument("--stats_key", type=str)
    parser.add_argument(
        "--lerobot_gripper_convention",
        choices=["binary_open", "signed_close"],
        help="Required for LeRobot: binary_open is 1=open/0=close; signed_close is -1=open/+1=close.",
    )
    parser.add_argument(
        "--normalize_binary_gripper",
        type=str,
        default="auto",
        choices=["auto", "true", "false", "1", "0", "yes", "no"],
    )
    known, remaining = parser.parse_known_args()
    if any(argument in {"-h", "--help"} for argument in remaining):
        print("Mixed-suite input options:")
        parser.print_help()
        print("\nModel and optimization options:")

    original_argv = sys.argv
    sys.argv = [original_argv[0], *remaining]
    try:
        args = pi05._ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv

    required = {
        "--dataset_dirs": known.dataset_dirs,
        "--stats_path": known.stats_path,
        "--stats_key": known.stats_key,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    args.libero_light_augmentation = known.libero_light_augmentation
    args.dinov3_precision = known.dinov3_precision
    args.dataset_dirs = known.dataset_dirs
    args.dataset_format = known.dataset_format
    args.stats_path = known.stats_path
    args.stats_key = known.stats_key
    args.normalize_binary_gripper = known.normalize_binary_gripper
    args.lerobot_gripper_convention = known.lerobot_gripper_convention
    return args


def _dataset_paths(dataset_dirs: str) -> list[Path]:
    normalized = dataset_dirs.replace("\n", ",").replace(";", ",")
    paths = [Path(item.strip()).expanduser() for item in normalized.split(",") if item.strip()]
    if not paths:
        raise ValueError("--dataset_dirs must contain at least one path")
    return paths


def _resolve_dataset_format(dataset_dirs: str, requested: str) -> str:
    paths = _dataset_paths(dataset_dirs)
    if requested != "auto":
        resolved = requested
    else:
        kinds = set()
        for path in paths:
            if (path / "meta" / "info.json").is_file():
                kinds.add("lerobot")
            elif (path / "dataset_info.json").is_file():
                kinds.add("rlds")
            else:
                raise FileNotFoundError(
                    f"Cannot detect dataset format at {path}: expected meta/info.json or dataset_info.json"
                )
        if len(kinds) != 1:
            raise ValueError(f"All --dataset_dirs must use one format, found {sorted(kinds)}")
        resolved = kinds.pop()

    marker = "meta/info.json" if resolved == "lerobot" else "dataset_info.json"
    missing = [str(path) for path in paths if not (path / marker).is_file()]
    if missing:
        raise FileNotFoundError(f"--dataset_format={resolved} but {marker} is missing at: {missing}")
    return resolved


class MixedSuiteTrainingDataset:
    def __new__(cls, *args, **kwargs):
        active_args = trainer._ACTIVE_TRAIN_ARGS
        dataset_format = _resolve_dataset_format(active_args.dataset_dirs, active_args.dataset_format)
        dataset_cls = LiberoMixedLeRobotDataset if dataset_format == "lerobot" else LiberoMixedRLDSDataset
        if dataset_format == "lerobot":
            if active_args.lerobot_gripper_convention is None:
                raise ValueError("LeRobot input requires an explicit --lerobot_gripper_convention")
            kwargs["gripper_convention"] = active_args.lerobot_gripper_convention
            kwargs["light_augmentation"] = active_args.libero_light_augmentation
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"dataset_format={dataset_format}", flush=True)
        return dataset_cls(
            *args,
            dataset_dirs=active_args.dataset_dirs,
            stats_path=active_args.stats_path,
            stats_key=active_args.stats_key,
            normalize_binary_gripper=active_args.normalize_binary_gripper,
            **kwargs,
        )


trainer._ACTIVE_TRAIN_ARGS = None


def parse_args_and_record():
    args = parse_args_with_mixed_suite_stats()
    trainer._ACTIVE_TRAIN_ARGS = args
    return args


trainer.parse_args = parse_args_and_record
trainer.LiberoRLDSDataset = MixedSuiteTrainingDataset


def main():
    trainer.train_model()


if __name__ == "__main__":
    main()
