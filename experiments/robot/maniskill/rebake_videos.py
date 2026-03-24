from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.robot.maniskill.artifacts import (
    build_exemplar_manifest,
    get_video_path,
    save_video_from_frames,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"INVALID_JSON_OBJECT: expected object at {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _resolve_frame_dir(frame_dir_value: str, run_dir: Path) -> Path:
    candidate = Path(frame_dir_value)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return run_dir / candidate


def _collect_frame_paths(frame_dir: Path) -> list[Path]:
    frames = sorted(frame_dir.glob("frame_*.png"))
    if frames:
        return frames
    return sorted(frame_dir.glob("*.png"))


def _episode_sort_key(record: dict[str, Any]) -> tuple[str, int, int]:
    task_id = str(record.get("task_id", ""))
    episode_index = int(record.get("episode_index", 10**9))
    seed = int(record.get("seed", 10**9))
    return (task_id, episode_index, seed)


def _load_default_rebake_episodes(run_dir: Path, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    exemplar_manifest = manifest.get("exemplars")
    if not isinstance(exemplar_manifest, dict):
        exemplar_manifest = build_exemplar_manifest(episodes)

    per_task = exemplar_manifest.get("per_task")
    if not isinstance(per_task, dict):
        per_task = {}

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for task_id in sorted(per_task.keys()):
        task_entry = per_task.get(task_id, {})
        if not isinstance(task_entry, dict):
            continue
        for cls_name in ("success", "failure"):
            rows = task_entry.get(cls_name, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                key = (
                    str(row.get("task_id", "")),
                    int(row.get("episode_index", -1)),
                    int(row.get("seed", -1)),
                )
                if key in seen:
                    continue
                seen.add(key)
                selected.append(row)

    if selected:
        return sorted(selected, key=_episode_sort_key)

    return sorted(episodes, key=_episode_sort_key)


def _load_targeted_episode(episodes: list[dict[str, Any]], task_id: str, episode_index: int) -> list[dict[str, Any]]:
    matches = [
        row
        for row in episodes
        if str(row.get("task_id", "")) == task_id and int(row.get("episode_index", -1)) == int(episode_index)
    ]
    return sorted(matches, key=_episode_sort_key)[:1]


def _rebake_episodes(run_dir: Path, episodes: Iterable[dict[str, Any]], fps: int, overwrite: bool) -> int:
    rebaked_count = 0
    skipped_count = 0
    for episode in episodes:
        task_id = str(episode.get("task_id", ""))
        episode_index = int(episode.get("episode_index", -1))
        success = bool(episode.get("success", False))
        frame_dir_value = str(episode.get("frame_dir", ""))

        frame_dir = _resolve_frame_dir(frame_dir_value, run_dir=run_dir)
        if not frame_dir.is_dir():
            print(f"FRAME_DIR_MISSING:{frame_dir}")
            return 1

        frame_paths = _collect_frame_paths(frame_dir)
        if not frame_paths:
            print(f"FRAME_DIR_MISSING:{frame_dir}")
            return 1

        output_path = get_video_path(run_dir, task_id=task_id, episode_index=episode_index, success=success)
        if output_path.exists() and not overwrite:
            print(f"VIDEO_EXISTS_SKIP:{output_path}")
            skipped_count += 1
            continue

        save_video_from_frames(frame_paths, output_path, fps=fps)
        print(
            f"REBAKED task_id={task_id} episode_index={episode_index} "
            f"success={int(success)} frame_count={len(frame_paths)} video_path={output_path}"
        )
        rebaked_count += 1

    print(f"REBAKE_DONE: count={rebaked_count} skipped={skipped_count}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild ManiSkill rollout MP4s from saved frames and metadata.")
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--task_id", type=str, default=None)
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"RUN_DIR_MISSING:{run_dir}")
        return 1

    episodes_path = run_dir / "episodes.jsonl"
    if not episodes_path.is_file():
        print(f"FRAME_DIR_MISSING:{episodes_path}")
        return 1

    episodes = _read_jsonl(episodes_path)

    if (args.task_id is None) != (args.episode_index is None):
        raise ValueError("INVALID_TARGET: provide both --task_id and --episode_index together")

    if args.task_id is not None and args.episode_index is not None:
        selected = _load_targeted_episode(episodes, task_id=args.task_id, episode_index=args.episode_index)
    else:
        selected = _load_default_rebake_episodes(run_dir=run_dir, episodes=episodes)

    if not selected:
        print("REBAKE_DONE: count=0")
        return 0

    return _rebake_episodes(run_dir=run_dir, episodes=selected, fps=args.fps, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
