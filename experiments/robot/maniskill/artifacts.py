from __future__ import annotations

import json
import time
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from experiments.robot.maniskill.defaults import (
    ARTIFACT_ROOT,
    EXEMPLAR_LIMITS,
    SUMMARY_SCHEMA_KEYS,
)


EPISODES_JSONL = "episodes.jsonl"
SUMMARY_JSON = "summary.json"
MANIFEST_JSON = "manifest.json"
COMPARISON_SUMMARY_JSON = "comparison_summary.json"
FRAMES_DIR = "frames"
VIDEOS_DIR = "videos"
COMPARISON_ARTIFACT_ROOT = "rollouts/maniskill_comparisons"
COMPARE_SUMMARY_SCHEMA_KEYS = [
    "compare_id",
    "comparison_status",
    "children",
    "artifact_paths",
]


def _to_path(path_like: str | Path) -> Path:
    return Path(path_like)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _safe_task_id(task_id: str) -> str:
    return task_id.replace("/", "_").replace(" ", "_")


def _episode_dir_name(episode_index: int) -> str:
    return f"episode_{episode_index:04d}"


def _seed_dir_name(seed: int) -> str:
    return f"seed_{seed}"


def _episode_sort_key(episode: Mapping[str, Any]) -> int:
    raw_episode_index = episode.get("episode_index")
    if raw_episode_index is None:
        return 10**12
    try:
        return int(raw_episode_index)
    except Exception:
        return 10**12


def _require_summary_keys(summary_payload: Mapping[str, Any]) -> None:
    missing = [key for key in SUMMARY_SCHEMA_KEYS if key not in summary_payload]
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Summary payload missing required keys: {missing_keys}")


def _require_compare_summary_keys(summary_payload: Mapping[str, Any]) -> None:
    missing = [key for key in COMPARE_SUMMARY_SCHEMA_KEYS if key not in summary_payload]
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Comparison summary payload missing required keys: {missing_keys}")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("w") as f:
        json.dump(dict(payload), f, indent=2, default=_json_default)


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(dict(row), default=_json_default))
            f.write("\n")


def create_run_layout(run_id: str, artifact_root: str | Path = ARTIFACT_ROOT) -> Path:
    run_dir = _to_path(artifact_root) / run_id
    (run_dir / FRAMES_DIR).mkdir(parents=True, exist_ok=True)
    (run_dir / VIDEOS_DIR).mkdir(parents=True, exist_ok=True)

    (run_dir / EPISODES_JSONL).touch(exist_ok=True)
    _write_json(
        run_dir / MANIFEST_JSON,
        {
            "run_id": run_id,
            "created_at_unix": time.time(),
            "artifact_root": str(_to_path(artifact_root)),
            "artifact_paths": build_artifact_paths(run_dir),
        },
    )
    return run_dir


def build_artifact_paths(run_dir: str | Path) -> Dict[str, str]:
    resolved_run_dir = _to_path(run_dir)
    return {
        "run_dir": str(resolved_run_dir),
        "summary": str(resolved_run_dir / SUMMARY_JSON),
        "manifest": str(resolved_run_dir / MANIFEST_JSON),
        "episodes": str(resolved_run_dir / EPISODES_JSONL),
        "frames": str(resolved_run_dir / FRAMES_DIR),
        "videos": str(resolved_run_dir / VIDEOS_DIR),
    }


def create_comparison_layout(
    compare_id: str,
    artifact_root: str | Path = COMPARISON_ARTIFACT_ROOT,
) -> Path:
    compare_dir = _to_path(artifact_root) / compare_id
    (compare_dir / "openvla").mkdir(parents=True, exist_ok=True)
    (compare_dir / "pi0").mkdir(parents=True, exist_ok=True)
    return compare_dir


def build_comparison_artifact_paths(compare_dir: str | Path) -> Dict[str, str]:
    resolved_compare_dir = _to_path(compare_dir)
    return {
        "compare_dir": str(resolved_compare_dir),
        "comparison_summary": str(resolved_compare_dir / COMPARISON_SUMMARY_JSON),
        "openvla_root": str(resolved_compare_dir / "openvla"),
        "pi0_root": str(resolved_compare_dir / "pi0"),
    }


def get_frame_dir(
    run_dir: str | Path,
    task_id: str,
    seed: int,
    episode_index: int,
    create: bool = True,
) -> Path:
    task_dir = _to_path(run_dir) / FRAMES_DIR / _safe_task_id(task_id)
    frame_dir = task_dir / _seed_dir_name(seed) / _episode_dir_name(episode_index)
    if create:
        frame_dir.mkdir(parents=True, exist_ok=True)
    return frame_dir


def get_video_dir(run_dir: str | Path, task_id: str, create: bool = True) -> Path:
    video_dir = _to_path(run_dir) / VIDEOS_DIR / _safe_task_id(task_id)
    if create:
        video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir


def get_video_path(run_dir: str | Path, task_id: str, episode_index: int, success: bool) -> Path:
    video_dir = get_video_dir(run_dir, task_id, create=True)
    file_name = (
        f"task={_safe_task_id(task_id)}--episode={episode_index:04d}--"
        f"success={int(bool(success))}.mp4"
    )
    return video_dir / file_name


def write_summary(run_dir: str | Path, summary_payload: Mapping[str, Any]) -> Path:
    _require_summary_keys(summary_payload)
    summary = dict(summary_payload)
    artifact_paths = dict(summary.get("artifact_paths", {}))
    artifact_paths.update(build_artifact_paths(run_dir))
    summary["artifact_paths"] = artifact_paths

    summary_path = _to_path(run_dir) / SUMMARY_JSON
    _write_json(summary_path, summary)
    return summary_path


def write_manifest(run_dir: str | Path, manifest_payload: Mapping[str, Any]) -> Path:
    manifest = dict(manifest_payload)
    manifest.setdefault("artifact_paths", build_artifact_paths(run_dir))
    manifest_path = _to_path(run_dir) / MANIFEST_JSON
    _write_json(manifest_path, manifest)
    return manifest_path


def write_comparison_summary(compare_dir: str | Path, summary_payload: Mapping[str, Any]) -> Path:
    _require_compare_summary_keys(summary_payload)
    summary = dict(summary_payload)
    artifact_paths = dict(summary.get("artifact_paths", {}))
    artifact_paths.update(build_comparison_artifact_paths(compare_dir))
    summary["artifact_paths"] = artifact_paths

    summary_path = _to_path(compare_dir) / COMPARISON_SUMMARY_JSON
    _write_json(summary_path, summary)
    return summary_path


def build_episode_metadata(
    task_id: str,
    episode_index: int,
    success: bool,
    seed: int,
    checkpoint_id: str,
    checkpoint_path: str,
    frame_dir: str | Path,
    timing: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "episode_index": int(episode_index),
        "success": bool(success),
        "seed": int(seed),
        "checkpoint": {
            "id": checkpoint_id,
            "path": checkpoint_path,
        },
        "timing": dict(timing),
        "frame_dir": str(frame_dir),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def append_episode_record(run_dir: str | Path, episode_payload: Mapping[str, Any]) -> Path:
    records_path = _to_path(run_dir) / EPISODES_JSONL
    _append_jsonl(records_path, [episode_payload])
    return records_path


def append_episode_records(run_dir: str | Path, episode_payloads: Iterable[Mapping[str, Any]]) -> Path:
    records_path = _to_path(run_dir) / EPISODES_JSONL
    _append_jsonl(records_path, episode_payloads)
    return records_path


def select_exemplars(
    episodes: Sequence[Mapping[str, Any]],
    success_limit: int = EXEMPLAR_LIMITS["success"],
    failure_limit: int = EXEMPLAR_LIMITS["failure"],
) -> Dict[str, List[Mapping[str, Any]]]:
    selected_success: List[Mapping[str, Any]] = []
    selected_failure: List[Mapping[str, Any]] = []

    for episode in episodes:
        is_success = bool(episode.get("success", False))
        if is_success and len(selected_success) < success_limit:
            selected_success.append(episode)
        if (not is_success) and len(selected_failure) < failure_limit:
            selected_failure.append(episode)
        if len(selected_success) >= success_limit and len(selected_failure) >= failure_limit:
            break

    return {
        "success": selected_success,
        "failure": selected_failure,
    }


def select_exemplars_by_task(
    episode_records: Sequence[Mapping[str, Any]],
    task_ids: Sequence[str] | None = None,
    success_limit: int = EXEMPLAR_LIMITS["success"],
    failure_limit: int = EXEMPLAR_LIMITS["failure"],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {task_id: [] for task_id in (task_ids or [])}
    for record in episode_records:
        task_id = str(record.get("task_id", ""))
        grouped.setdefault(task_id, []).append(record)

    per_task: Dict[str, Dict[str, Any]] = {}
    for task_id, episodes in grouped.items():
        ordered_episodes = sorted(episodes, key=_episode_sort_key)
        selected = select_exemplars(ordered_episodes, success_limit=success_limit, failure_limit=failure_limit)
        success_available = sum(1 for episode in ordered_episodes if bool(episode.get("success", False)))
        failure_available = len(ordered_episodes) - success_available

        per_task[task_id] = {
            "success": selected["success"],
            "failure": selected["failure"],
            "class_counts": {
                "success_available": success_available,
                "failure_available": failure_available,
                "success_selected": len(selected["success"]),
                "failure_selected": len(selected["failure"]),
                "absent_success_count": 1 if success_available == 0 else 0,
                "absent_failure_count": 1 if failure_available == 0 else 0,
            },
        }

    return per_task


def build_exemplar_manifest(
    episode_records: Sequence[Mapping[str, Any]],
    task_ids: Sequence[str] | None = None,
    success_limit: int = EXEMPLAR_LIMITS["success"],
    failure_limit: int = EXEMPLAR_LIMITS["failure"],
) -> Dict[str, Any]:
    per_task = select_exemplars_by_task(
        episode_records,
        task_ids=task_ids,
        success_limit=success_limit,
        failure_limit=failure_limit,
    )
    return {
        "selection_policy": {
            "success_limit": success_limit,
            "failure_limit": failure_limit,
            "ordering": "episode_order",
            "deterministic": True,
            "notes": "first successful and failed episodes in order, up to per-class limits",
        },
        "per_task": per_task,
    }


def save_video_from_frames(frame_paths: Sequence[str | Path], output_path: str | Path, fps: int = 30) -> Path:
    imageio = importlib.import_module("imageio")

    output = _to_path(output_path)
    _ensure_parent(output)
    writer = imageio.get_writer(output, fps=fps)
    try:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    finally:
        writer.close()
    return output
