from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.robot.maniskill.defaults import (
    ARTIFACT_ROOT,
    CHECKPOINT_POLICY,
    DEFAULT_GPU_INDEX_ENV_KEY,
    FULL_EPISODE_COUNT_PER_TASK,
    RAW_FRAME_LAYOUT_CONTRACT,
    SEED_POLICY,
    TASK_IDS,
)


CHECKPOINT_ENV_KEY = "OPENVLA_MANISKILL_CHECKPOINT"
MIN_FREE_DISK_BYTES_ENV_KEY = "OPENVLA_MANISKILL_MIN_FREE_DISK_BYTES"
DEFAULT_MIN_FREE_DISK_BYTES = 40 * 1024**3


def fail(tag: str, message: str, code: int = 1) -> NoReturn:
    print(f"{tag}: {message}")
    raise SystemExit(code)


def _is_reference_checkpoint(checkpoint_value: str) -> bool:
    fallback_reference = str(CHECKPOINT_POLICY["fallback_reference"])
    return checkpoint_value == fallback_reference


def _resolve_checkpoint_target() -> str:
    checkpoint_value = os.environ.get(CHECKPOINT_ENV_KEY, "").strip()
    if checkpoint_value:
        return checkpoint_value
    return str(CHECKPOINT_POLICY["fallback_reference"])


def _validate_checkpoint_layout(checkpoint_target: str) -> None:
    checkpoint_path = Path(checkpoint_target)
    if not checkpoint_path.exists():
        if _is_reference_checkpoint(checkpoint_target):
            return
        fail("CHECKPOINT_MISSING", f"Path does not exist: `{checkpoint_path}`")

    run_dir: Path
    if checkpoint_path.is_file():
        if checkpoint_path.suffix != ".pt" or checkpoint_path.parent.name != "checkpoints":
            fail(
                "CHECKPOINT_MISSING",
                f"Expected a `.pt` file under a `checkpoints/` directory but got `{checkpoint_path}`.",
            )
        run_dir = checkpoint_path.parents[1]
    elif checkpoint_path.is_dir():
        run_dir = checkpoint_path
        checkpoint_path = run_dir / "checkpoints" / "latest-checkpoint.pt"
        if not checkpoint_path.exists():
            fail(
                "CHECKPOINT_MISSING",
                f"Missing `checkpoints/latest-checkpoint.pt` under `{run_dir}`.",
            )
    else:
        fail("CHECKPOINT_MISSING", f"Unsupported checkpoint path: `{checkpoint_path}`")

    config_json = run_dir / "config.json"
    dataset_statistics_json = run_dir / "dataset_statistics.json"
    if not config_json.exists():
        fail("CHECKPOINT_MISSING", f"Missing `config.json` for `{run_dir}`")
    if not dataset_statistics_json.exists():
        fail("CHECKPOINT_MISSING", f"Missing `dataset_statistics.json` for `{run_dir}`")


def _validate_gpu_policy() -> None:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        fail("GPU_POLICY_UNAVAILABLE", f"Unable to import `torch` for CUDA check: {exc}")

    if not torch.cuda.is_available():
        required_gpu_index = os.environ.get(DEFAULT_GPU_INDEX_ENV_KEY, "").strip()
        required_label = required_gpu_index if required_gpu_index else "auto/visible"
        fail(
            "GPU_POLICY_UNAVAILABLE",
            f"CUDA is unavailable; required GPU index is {required_label}.",
        )

    gpu_count = torch.cuda.device_count()
    required_gpu_index_raw = os.environ.get(DEFAULT_GPU_INDEX_ENV_KEY, "").strip()
    required_gpu_index = None
    if required_gpu_index_raw:
        try:
            required_gpu_index = int(required_gpu_index_raw)
        except ValueError:
            fail(
                "GPU_POLICY_UNAVAILABLE",
                f"`{DEFAULT_GPU_INDEX_ENV_KEY}` must be an integer physical GPU index, got `{required_gpu_index_raw}`.",
            )

    visible_devices_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices_raw is not None:
        visible_devices = [entry.strip() for entry in visible_devices_raw.split(",") if entry.strip()]
        if required_gpu_index is not None and str(required_gpu_index) not in visible_devices:
            fail(
                "GPU_POLICY_UNAVAILABLE",
                f"CUDA_VISIBLE_DEVICES does not include required physical GPU {required_gpu_index}: `{visible_devices_raw}`.",
            )
        if gpu_count < 1:
            fail(
                "GPU_POLICY_UNAVAILABLE",
                "CUDA_VISIBLE_DEVICES is set but no CUDA device is visible to Python.",
            )
        return

    if required_gpu_index is not None and gpu_count <= required_gpu_index:
        fail(
            "GPU_POLICY_UNAVAILABLE",
            f"No CUDA_VISIBLE_DEVICES mapping found and required physical GPU {required_gpu_index} is missing; only {gpu_count} GPU(s) detected.",
        )

    if gpu_count < 1:
        fail("GPU_POLICY_UNAVAILABLE", "No CUDA devices detected.")




def _validate_maniskill_import() -> None:
    try:
        _ = importlib.import_module("mani_skill")
    except Exception as exc:
        fail("MANISKILL_DEPENDENCY_MISSING", f"Unable to import `mani_skill`: {exc}")


def _validate_renderer_dependencies() -> None:
    try:
        _ = importlib.import_module("imageio")
        _ = importlib.import_module("imageio_ffmpeg")
    except Exception as exc:
        fail(
            "RENDERER_DEPENDENCY_MISSING",
            f"Missing video/render dependency for `imageio[ffmpeg]`: {exc}",
        )


def _disk_target_and_budget() -> tuple[Path, int]:
    frames_root = Path(RAW_FRAME_LAYOUT_CONTRACT["root"])
    artifact_root = Path(ARTIFACT_ROOT)

    default_seed_count = len(SEED_POLICY.get("full", [])) or 1
    full_rollout_count = len(TASK_IDS) * default_seed_count * FULL_EPISODE_COUNT_PER_TASK
    rough_floor_bytes = max(DEFAULT_MIN_FREE_DISK_BYTES, full_rollout_count * 128 * 1024**2)

    min_free_disk_bytes_str = os.environ.get(MIN_FREE_DISK_BYTES_ENV_KEY, "").strip()
    min_free_disk_bytes = rough_floor_bytes
    if min_free_disk_bytes_str:
        try:
            min_free_disk_bytes = int(min_free_disk_bytes_str)
        except ValueError:
            fail(
                "DISK_BUDGET_UNAVAILABLE",
                f"`{MIN_FREE_DISK_BYTES_ENV_KEY}` must be an integer byte count.",
            )

    if RAW_FRAME_LAYOUT_CONTRACT.get("retention") != "retain_all_raw_frames":
        min_free_disk_bytes = max(min_free_disk_bytes, DEFAULT_MIN_FREE_DISK_BYTES)

    target_path = frames_root if frames_root.parts else artifact_root
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)
    return target_path, min_free_disk_bytes


def _validate_disk_budget() -> None:
    target_path, min_free_disk_bytes = _disk_target_and_budget()
    usage = shutil.disk_usage(target_path)
    if usage.free < min_free_disk_bytes:
        fail(
            "DISK_BUDGET_UNAVAILABLE",
            f"Free space at `{target_path}` is {usage.free} bytes; requires at least {min_free_disk_bytes} bytes for `{RAW_FRAME_LAYOUT_CONTRACT.get('retention')}`.",
        )


def main() -> None:
    checkpoint_target = _resolve_checkpoint_target()
    _validate_checkpoint_layout(checkpoint_target)
    _validate_gpu_policy()
    _validate_maniskill_import()
    _validate_renderer_dependencies()
    _validate_disk_budget()
    print("SETUP_OK")


if __name__ == "__main__":
    main()
