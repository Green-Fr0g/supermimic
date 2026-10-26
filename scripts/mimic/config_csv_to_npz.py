"""Read a motion config yaml and convert its CSV paths via ``csv_to_npz.py``.

``csv_to_npz.py`` is config-free, so this launcher acts as the
config-driven front end:

1. read the robot and motion metadata from a yaml config
2. resolve every CSV path relative to that config file
3. build one ``csv_to_npz.py`` command per CSV and run it

Example:
    python scripts/mimic/config_csv_to_npz.py \
        --config motion_data/config/g1.yaml \
        --output_dir /tmp/motions

    python scripts/mimic/config_csv_to_npz.py \
        --config motion_data/config/g1.yaml \
        --output_dir /tmp/motions \
        --overwrite --headless
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MotionConfig:
    """Robot and motion metadata read from a converter config."""

    robot_name: str
    root_link_name: str
    root_quat_order: str
    input_fps: int
    csv_joint_names: tuple[str, ...]
    csv_paths: tuple[Path, ...]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Read a motion config yaml and convert its CSV paths via scripts/mimic/csv_to_npz.py."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Motion config yaml containing a motion_data mapping.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the generated NPZ files will be written.",
    )
    parser.add_argument("--output_fps", type=int, default=50, help="Output NPZ frame rate.")
    parser.add_argument(
        "--physics",
        choices=("physx", "isaacsim_physx", "newton_mjwarp", "newton_vbd", "ovphysx"),
        default="newton_mjwarp",
        help="Physics backend passed to the Isaac Lab 3.0 converter.",
    )
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional 1-based inclusive frame range passed to the single-file converter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate NPZ files even when the output already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned commands without running conversion.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run csv_to_npz.py.",
    )
    return parser.parse_known_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml structure in: {path}")
    return data


def _resolve_path(base_dir: Path, maybe_path: str) -> Path:
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _require_str(motion_cfg: dict[str, Any], key: str, config_path: Path) -> str:
    value = motion_cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"motion_data.{key} must be a non-empty string in: {config_path}")
    return value


def _require_str_list(motion_cfg: dict[str, Any], key: str, config_path: Path) -> list[str]:
    value = motion_cfg.get(key)
    if not isinstance(value, list) or not value:
        raise TypeError(f"motion_data.{key} must be a non-empty list of strings in: {config_path}")
    entries: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"motion_data.{key} must contain only non-empty strings in: {config_path}")
        entries.append(item)
    return entries


def load_motion_config(config_path: Path) -> MotionConfig:
    """Validate a converter config and return its motion metadata."""
    config = _load_yaml(config_path)
    motion_cfg = config.get("motion_data")
    if not isinstance(motion_cfg, dict):
        raise TypeError(f"'motion_data' must be a mapping in: {config_path}")

    root_quat_order = str(motion_cfg.get("root_quat_order", "xyzw"))
    if root_quat_order not in ("wxyz", "xyzw"):
        raise ValueError(f"motion_data.root_quat_order must be 'wxyz' or 'xyzw' in: {config_path}")

    input_fps = int(motion_cfg.get("csv_fps", 30))
    if input_fps <= 0:
        raise ValueError(f"motion_data.csv_fps must be positive in: {config_path}")

    return MotionConfig(
        robot_name=_require_str(motion_cfg, "robot_name", config_path),
        root_link_name=_require_str(motion_cfg, "root_link_name", config_path),
        root_quat_order=root_quat_order,
        input_fps=input_fps,
        csv_joint_names=tuple(_require_str_list(motion_cfg, "csv_joint_names", config_path)),
        csv_paths=tuple(
            _resolve_path(config_path.parent, entry)
            for entry in _require_str_list(motion_cfg, "csv_paths", config_path)
        ),
    )


def _unique_output_path(output_dir: Path, csv_path: Path, used_names: set[str]) -> Path:
    """Pick a stable, collision-free NPZ name for one CSV file."""
    stem = csv_path.stem
    candidate = f"{stem}.npz"
    if candidate not in used_names:
        used_names.add(candidate)
        return output_dir / candidate

    parent_stem = csv_path.parent.name or "motion"
    candidate = f"{parent_stem}__{stem}.npz"
    if candidate not in used_names:
        used_names.add(candidate)
        return output_dir / candidate

    index = 2
    while True:
        candidate = f"{parent_stem}__{stem}_{index}.npz"
        if candidate not in used_names:
            used_names.add(candidate)
            return output_dir / candidate
        index += 1


def build_command(
    python_executable: str,
    converter_path: Path,
    csv_path: Path,
    motion_cfg: MotionConfig,
    output_path: Path,
    output_fps: int,
    physics: str,
    frame_range: list[int] | None,
    passthrough_args: list[str],
) -> list[str]:
    """Build the ``csv_to_npz.py`` command for a single CSV file."""
    command = [
        python_executable,
        str(converter_path),
        "--input_file",
        str(csv_path),
        "--input_fps",
        str(motion_cfg.input_fps),
        "--output_name",
        str(output_path),
        "--output_fps",
        str(output_fps),
        "--physics",
        physics,
        "--robot_name",
        motion_cfg.robot_name,
        "--root_link_name",
        motion_cfg.root_link_name,
        "--csv_joint_names",
        *motion_cfg.csv_joint_names,
        "--root_quat_order",
        motion_cfg.root_quat_order,
    ]
    if frame_range is not None:
        command.extend(["--frame_range", str(frame_range[0]), str(frame_range[1])])
    command.extend(passthrough_args)
    return command


def main() -> None:
    args, passthrough_args = parse_args()

    script_dir = Path(__file__).resolve().parent
    converter_path = script_dir / "csv_to_npz.py"
    if not converter_path.is_file():
        raise FileNotFoundError(f"Single-file converter not found: {converter_path}")

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    motion_cfg = load_motion_config(config_path)
    if args.output_fps <= 0:
        raise ValueError("--output_fps must be positive")
    if args.frame_range is not None and (args.frame_range[0] < 1 or args.frame_range[1] < args.frame_range[0]):
        raise ValueError("--frame_range must be a 1-based inclusive range with START <= END")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Config: {config_path}")
    print(f"[INFO] Robot: {motion_cfg.robot_name} | root link: {motion_cfg.root_link_name}")
    print(f"[INFO] Found {len(motion_cfg.csv_paths)} csv job(s).")
    print(f"[INFO] Output directory: {output_dir}")
    if passthrough_args:
        print(f"[INFO] Passing extra args to csv_to_npz.py: {' '.join(passthrough_args)}")

    converted = 0
    skipped = 0
    failed = 0
    used_names: set[str] = set()

    for csv_path in motion_cfg.csv_paths:
        output_path = _unique_output_path(output_dir, csv_path, used_names)

        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] {output_path} already exists.")
            skipped += 1
            continue

        command = build_command(
            python_executable=args.python,
            converter_path=converter_path,
            csv_path=csv_path,
            motion_cfg=motion_cfg,
            output_path=output_path,
            output_fps=args.output_fps,
            physics=args.physics,
            frame_range=args.frame_range,
            passthrough_args=passthrough_args,
        )
        print(f"[RUN] {csv_path} -> {output_path}")
        print(f"      {' '.join(command)}")

        if args.dry_run:
            if not csv_path.is_file():
                print(f"[WARN] CSV not found: {csv_path}")
            converted += 1
            continue

        if not csv_path.is_file():
            print(f"[FAIL] CSV not found: {csv_path}")
            failed += 1
            continue

        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"[FAIL] Conversion failed with exit code {result.returncode}: {csv_path}")
            failed += 1
            continue
        converted += 1

    print(f"[DONE] converted={converted}, skipped={skipped}, failed={failed}, total={len(motion_cfg.csv_paths)}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
