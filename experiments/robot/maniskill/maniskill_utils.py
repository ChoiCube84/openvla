from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MANISKILL_OBS_MODE = "rgb"
MANISKILL_CONTROL_MODE = "pd_ee_delta_pose"
MANISKILL_EXPECTED_ACTION_DIM = 7


def create_maniskill_env(task_id: str, render_mode: str = "rgb_array", obs_mode: str = MANISKILL_OBS_MODE, control_mode: str = MANISKILL_CONTROL_MODE):
    import mani_skill  # noqa: F401
    import gymnasium as gym

    env = gym.make(
        task_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
    )

    validate_maniskill_action_space(env.action_space, expected_dim=MANISKILL_EXPECTED_ACTION_DIM)
    return env


def validate_maniskill_action_space(action_space: Any, expected_dim: int = MANISKILL_EXPECTED_ACTION_DIM) -> None:
    if not hasattr(action_space, "shape"):
        raise ValueError("ACTION_SHAPE_MISMATCH: action space has no shape attribute.")
    shape = tuple(action_space.shape)
    if len(shape) != 1 or shape[0] != expected_dim:
        raise ValueError(
            f"ACTION_SHAPE_MISMATCH: expected 1D action space with dim={expected_dim}, got shape={shape}."
        )


def _safe_get_by_path(container: Any, path: Sequence[str]) -> Any:
    current = container
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass

    return None


def _iter_array_candidates(prefix: str, payload: Any) -> Iterable[tuple[str, np.ndarray]]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_array_candidates(next_prefix, value)
        return

    array = _to_numpy(payload)
    if array is not None:
        yield prefix, array


def _normalize_rgb_array(array: np.ndarray, source: str) -> np.ndarray:
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(
                f"OBS_SCHEMA_MISMATCH: image source `{source}` expected leading batch=1 for 4D tensor, "
                f"got shape={array.shape}."
            )
        array = array[0]

    if array.ndim != 3:
        raise ValueError(f"OBS_SCHEMA_MISMATCH: image source `{source}` must be HWC-like, got shape={array.shape}.")

    if array.shape[-1] not in (3, 4):
        raise ValueError(
            f"OBS_SCHEMA_MISMATCH: image source `{source}` expected channel dimension 3 or 4, got {array.shape[-1]}."
        )

    if array.shape[-1] == 4:
        array = array[..., :3]

    if np.issubdtype(array.dtype, np.floating):
        if np.nanmax(array) <= 1.0:
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = array.astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return array


def extract_image_observation(observation: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    preferred_paths = [
        ("sensor_data.base_camera.rgb", ("sensor_data", "base_camera", "rgb")),
        ("sensor_data.hand_camera.rgb", ("sensor_data", "hand_camera", "rgb")),
        ("sensor_data.camera.rgb", ("sensor_data", "camera", "rgb")),
        ("rgb", ("rgb",)),
        ("image", ("image",)),
    ]

    for name, path in preferred_paths:
        value = _safe_get_by_path(observation, path)
        array = _to_numpy(value)
        if array is not None:
            return _normalize_rgb_array(array, name), name

    candidates = []
    for name, array in _iter_array_candidates("", observation):
        lower_name = name.lower()
        if ("rgb" not in lower_name) and ("image" not in lower_name):
            continue
        if array.ndim not in (3, 4):
            continue
        if array.shape[-1] not in (3, 4):
            continue
        candidates.append((name, array))

    if not candidates:
        top_level_keys = ", ".join(sorted(str(key) for key in observation.keys()))
        raise ValueError(
            "OBS_SCHEMA_MISMATCH: no RGB image source found. "
            f"Expected keys like sensor_data.*.rgb / rgb / image, top-level keys=[{top_level_keys}]"
        )

    source, array = candidates[0]
    return _normalize_rgb_array(array, source), source


def adapt_action_for_maniskill(action: np.ndarray, action_space: Any) -> np.ndarray:
    if not hasattr(action_space, "shape"):
        raise ValueError("ACTION_SHAPE_MISMATCH: ManiSkill action space has no `shape` attribute.")

    expected_shape = tuple(action_space.shape)
    action = np.asarray(action, dtype=np.float32)

    if action.shape != expected_shape:
        raise ValueError(
            f"ACTION_SHAPE_MISMATCH: expected action shape {expected_shape}, got {tuple(action.shape)}."
        )

    if not np.all(np.isfinite(action)):
        raise ValueError("ACTION_SHAPE_MISMATCH: action contains non-finite values.")

    if hasattr(action_space, "low") and hasattr(action_space, "high"):
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        if low.shape == action.shape and high.shape == action.shape:
            action = np.clip(action, low, high)

    return action


@dataclass(frozen=True)
class StepOutcome:
    success: bool
    terminal: bool
    terminal_reason: str


def interpret_step_outcome(terminated: bool, truncated: bool, info: Mapping[str, Any] | None) -> StepOutcome:
    info = info or {}
    success_keys = ("success", "is_success", "episode_success")
    success = any(bool(info.get(key, False)) for key in success_keys)

    terminal = bool(terminated) or bool(truncated) or success
    if success:
        reason = "success_info"
    elif terminated and truncated:
        reason = "terminated_and_truncated"
    elif terminated:
        reason = "terminated"
    elif truncated:
        reason = "truncated"
    else:
        reason = "running"

    return StepOutcome(success=success, terminal=terminal, terminal_reason=reason)
