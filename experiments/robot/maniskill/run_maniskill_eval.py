import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import draccus
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.robot.maniskill.artifacts import (
    append_episode_record,
    build_artifact_paths,
    build_episode_metadata,
    build_exemplar_manifest,
    create_run_layout,
    get_frame_dir,
    get_video_path,
    save_video_from_frames,
    write_manifest,
    write_summary,
)
from experiments.robot.maniskill.checkpoint_guard import (
    CheckpointValidationError,
    resolve_checkpoint_target,
    validate_checkpoint_reference,
)
from experiments.robot.maniskill.backends import get_backend
from experiments.robot.maniskill.defaults import (
    ARTIFACT_ROOT,
    ASSUMPTION_LEDGER_PATH_TEMPLATE,
    ASSUMPTIONS,
    CHECKPOINT_POLICY,
    EPISODE_COUNT_DEFAULTS,
    TASK_IDS,
)
from experiments.robot.maniskill.maniskill_utils import (
    MANISKILL_CONTROL_MODE,
    MANISKILL_EXPECTED_ACTION_DIM,
    MANISKILL_OBS_MODE,
    adapt_action_for_maniskill,
    create_maniskill_env,
    extract_image_observation,
    interpret_step_outcome,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_backend_info,
    get_image_resize_size,
    get_model,
    get_processor,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class ManiSkillEvalConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    mode: str = "smoke"
    task_ids: str = ",".join(TASK_IDS)
    episodes_per_task: Optional[int] = None
    max_steps_per_episode: int = 200
    render_mode: str = "rgb_array"
    obs_mode: str = MANISKILL_OBS_MODE
    control_mode: str = MANISKILL_CONTROL_MODE
    save_videos: bool = False

    unnorm_key: str = ""

    pi0_policy_server_url: str = "http://127.0.0.1:8000"
    openpi_conda_env: str = "openpi"
    openpi_repo_root: str = ""
    openpi_checkpoint: Union[str, Path] = "gs://openpi-assets/checkpoints/pi05_libero"

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    artifact_root: str = ARTIFACT_ROOT
    seed: int = 7


JUELG_MANISKILL_UNNORM_KEY = "maniskill_human:7.0.0"


def _parse_task_ids(task_ids_csv: str) -> list[str]:
    requested = [task_id.strip() for task_id in task_ids_csv.split(",") if task_id.strip()]
    if not requested:
        requested = list(TASK_IDS)

    unsupported = [task_id for task_id in requested if task_id not in TASK_IDS]
    if unsupported:
        supported = ",".join(TASK_IDS)
        bad = ",".join(unsupported)
        raise ValueError(f"UNSUPPORTED_TASK: {bad}. Supported task ids: {supported}")

    ordered_unique: list[str] = []
    seen: set[str] = set()
    for task_id in requested:
        if task_id in seen:
            continue
        ordered_unique.append(task_id)
        seen.add(task_id)
    return ordered_unique


def _resolve_episode_count(cfg: ManiSkillEvalConfig) -> int:
    if cfg.episodes_per_task is not None:
        if cfg.episodes_per_task < 1:
            raise ValueError("INVALID_EPISODE_COUNT: episodes_per_task must be >= 1")
        return int(cfg.episodes_per_task)

    if cfg.mode not in EPISODE_COUNT_DEFAULTS:
        valid_modes = ",".join(sorted(EPISODE_COUNT_DEFAULTS.keys()))
        raise ValueError(f"INVALID_MODE: {cfg.mode}. Valid modes: {valid_modes}")
    return int(EPISODE_COUNT_DEFAULTS[cfg.mode])


def _resolve_checkpoint(cfg: ManiSkillEvalConfig) -> str:
    if cfg.model_family == "pi0":
        return str(cfg.openpi_checkpoint)
    return resolve_checkpoint_target(
        checkpoint_override=str(cfg.pretrained_checkpoint),
        fallback_reference=str(CHECKPOINT_POLICY["fallback_reference"]),
    )


def _resolve_maniskill_version() -> str:
    try:
        import mani_skill

        return str(getattr(mani_skill, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _write_assumption_ledger(run_id: str, run_dir: Path, assumptions: list[dict[str, Any]]) -> Path:
    expected_path = Path(ASSUMPTION_LEDGER_PATH_TEMPLATE.format(run_id=run_id))
    actual_path = run_dir / "assumptions.json"
    output_path = expected_path if expected_path == actual_path else actual_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(assumptions, f, indent=2)
    return output_path


def _resize_for_model(image: np.ndarray, resize_size: int) -> np.ndarray:
    pil_image = Image.fromarray(image).convert("RGB")
    resized = pil_image.resize((resize_size, resize_size))
    return np.asarray(resized)


def _save_frame(frame: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.asarray(frame)
    if frame.ndim == 4:
        if frame.shape[0] != 1:
            raise ValueError(f"OBS_SCHEMA_MISMATCH: render frame expected batch=1 when 4D, got shape={frame.shape}.")
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"OBS_SCHEMA_MISMATCH: render frame must be HWC with 3/4 channels, got shape={frame.shape}.")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        if np.nanmax(frame) <= 1.0:
            frame = np.clip(frame, 0.0, 1.0) * 255.0
        frame = frame.astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    Image.fromarray(frame).convert("RGB").save(output_path)


def _resolve_task_unnorm_key(cfg: ManiSkillEvalConfig, model: Any, task_id: str) -> str:
    if cfg.model_family != "openvla":
        return cfg.unnorm_key

    norm_stats = getattr(model, "norm_stats", None)
    if not isinstance(norm_stats, dict) or not norm_stats:
        raise ValueError(f"INVALID_UNNORM_KEY: model norm_stats unavailable for ManiSkill task `{task_id}`.")

    task_id_lower = task_id.lower()
    task_id_snake = task_id_lower.replace("-", "_")
    task_stem = task_id.rsplit("-v", 1)[0]
    task_stem_lower = task_stem.lower()
    task_stem_snake = task_stem_lower.replace("-", "_")

    if cfg.unnorm_key.strip():
        selected = cfg.unnorm_key.strip()
        if selected not in norm_stats:
            available = ",".join(sorted(norm_stats.keys()))
            raise ValueError(
                f"INVALID_UNNORM_KEY: configured key `{selected}` missing for task `{task_id}`. available=[{available}]"
            )

        if selected == JUELG_MANISKILL_UNNORM_KEY:
            return selected

        if task_stem_lower not in selected.lower() and task_id_lower not in selected.lower():
            raise ValueError(
                f"UNNORM_KEY_TASK_MISMATCH: configured key `{selected}` is not task-specific for `{task_id}`."
            )
        return selected

    if JUELG_MANISKILL_UNNORM_KEY in norm_stats:
        return JUELG_MANISKILL_UNNORM_KEY

    candidates: list[str] = []
    for base in (
        task_id,
        task_id_lower,
        task_id_snake,
        task_stem,
        task_stem_lower,
        task_stem_snake,
        f"maniskill_{task_stem_lower}",
        f"maniskill_{task_stem_snake}",
        f"maniskill/{task_id}",
        f"maniskill/{task_stem}",
    ):
        if base not in candidates:
            candidates.append(base)
        no_noops = f"{base}_no_noops"
        if no_noops not in candidates:
            candidates.append(no_noops)

    for key in candidates:
        if key in norm_stats:
            return key

    candidate_list = ",".join(candidates)
    available = ",".join(sorted(norm_stats.keys()))
    raise ValueError(
        f"INVALID_UNNORM_KEY: no ManiSkill key match for task `{task_id}`. "
        f"candidates=[{candidate_list}] available=[{available}]"
    )


@draccus.wrap()
def eval_maniskill(cfg: ManiSkillEvalConfig) -> None:
    try:
        selected_task_ids = _parse_task_ids(cfg.task_ids)
        episodes_per_task = _resolve_episode_count(cfg)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1)

    if cfg.render_mode != "rgb_array":
        raise ValueError("INVALID_RENDER_MODE: ManiSkill benchmark runner requires render_mode='rgb_array'.")

    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("INVALID_QUANTIZATION: cannot enable both load_in_8bit and load_in_4bit.")

    cfg.pretrained_checkpoint = _resolve_checkpoint(cfg)
    set_seed_everywhere(cfg.seed)

    try:
        backend_info = get_backend_info(cfg)
        backend = get_backend(cfg.model_family)
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(1)

    try:
        if cfg.model_family == "openvla":
            checkpoint_validation = validate_checkpoint_reference(
                str(cfg.pretrained_checkpoint),
                source_type="auto",
                require_dataset_statistics=True,
            )
            print(
                "CHECKPOINT_VALID: "
                f"source_type={checkpoint_validation.source_type} "
                f"checkpoint_reference={checkpoint_validation.checkpoint_reference}"
            )
    except CheckpointValidationError as exc:
        print(f"{exc.tag}: {exc.message}")
        raise SystemExit(1)

    model = get_model(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    run_id = f"EVAL-maniskill-{cfg.mode}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id = f"{run_id}--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, f"{run_id}.txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    run_dir = create_run_layout(run_id=run_id, artifact_root=cfg.artifact_root)
    artifact_paths = build_artifact_paths(run_dir)

    assumptions = [dict(item) for item in ASSUMPTIONS]
    assumptions.append(
        {
            "key": "render_mode_locked",
            "label": "assumption",
            "value": "Runner uses ManiSkill headless RGB rendering with render_mode='rgb_array'.",
        }
    )
    assumptions.append(
        {
            "key": "checkpoint_resolution_policy",
            "label": "assumption",
            "value": "Checkpoint uses explicit user config when provided; otherwise defaults fallback reference from CHECKPOINT_POLICY.",
        }
    )
    assumption_ledger_path = _write_assumption_ledger(run_id, run_dir, assumptions)

    total_episodes = 0
    total_successes = 0
    episode_records: list[dict[str, Any]] = []
    per_task_success_rate: dict[str, float] = {}

    try:
        print(f"mode={cfg.mode} tasks={selected_task_ids} episodes_per_task={episodes_per_task} max_steps={cfg.max_steps_per_episode}")
        log_file.write(
            f"mode={cfg.mode} tasks={selected_task_ids} episodes_per_task={episodes_per_task} max_steps={cfg.max_steps_per_episode}\n"
        )
        log_file.flush()

        for task_id in selected_task_ids:
            backend.reset_rollout_state(task_label=task_id, reason="task_reset")
            task_unnorm_key = _resolve_task_unnorm_key(cfg, model, task_id)
            cfg.unnorm_key = task_unnorm_key
            env = create_maniskill_env(
                task_id=task_id,
                render_mode=cfg.render_mode,
                obs_mode=cfg.obs_mode,
                control_mode=cfg.control_mode,
            )
            task_successes = 0

            try:
                action_space = env.action_space
                low = np.asarray(action_space.low) if hasattr(action_space, "low") else None
                high = np.asarray(action_space.high) if hasattr(action_space, "high") else None
                print(
                    f"task={task_id} action_space_shape={getattr(action_space, 'shape', None)} "
                    f"expected_action_dim={MANISKILL_EXPECTED_ACTION_DIM} "
                    f"obs_mode={cfg.obs_mode} control_mode={cfg.control_mode} "
                    f"unnorm_key={task_unnorm_key} "
                    f"low={low.tolist() if low is not None else None} high={high.tolist() if high is not None else None}"
                )
                log_file.write(
                    f"task={task_id} action_space_shape={getattr(action_space, 'shape', None)} "
                    f"expected_action_dim={MANISKILL_EXPECTED_ACTION_DIM} "
                    f"obs_mode={cfg.obs_mode} control_mode={cfg.control_mode} "
                    f"unnorm_key={task_unnorm_key} "
                    f"low={low.tolist() if low is not None else None} high={high.tolist() if high is not None else None}\n"
                )
                log_file.flush()

                for episode_index in range(episodes_per_task):
                    start_time = time.perf_counter()
                    episode_seed = cfg.seed + episode_index
                    backend.reset_rollout_state(task_label=task_id, reason="episode_reset")
                    obs, _ = env.reset(seed=episode_seed)
                    frame_dir = get_frame_dir(
                        run_dir,
                        task_id=task_id,
                        seed=episode_seed,
                        episode_index=episode_index,
                        create=True,
                    )

                    frame_count = 0
                    frame_paths: list[Path] = []
                    terminal_reason = "max_steps"
                    success = False
                    image_source = "unknown"

                    for step_idx in range(cfg.max_steps_per_episode):
                        image, image_source = extract_image_observation(obs)
                        if step_idx == 0:
                            print(f"task={task_id} episode={episode_index} image_source={image_source} image_shape={tuple(image.shape)}")
                            log_file.write(
                                f"task={task_id} episode={episode_index} image_source={image_source} image_shape={tuple(image.shape)}\n"
                            )

                        model_image = _resize_for_model(image, resize_size=resize_size)
                        observation = {"full_image": model_image}
                        action = get_action(cfg, model, observation, task_id, processor=processor)
                        action = normalize_gripper_action(action, binarize=True)
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)

                        env_action = adapt_action_for_maniskill(action, action_space)
                        obs, _, terminated, truncated, info = env.step(env_action)

                        rendered_frame = env.render()
                        if isinstance(rendered_frame, torch.Tensor):
                            rendered_frame = rendered_frame.detach().cpu().numpy()
                        else:
                            rendered_frame = np.asarray(rendered_frame)
                        frame_path = frame_dir / f"frame_{frame_count:06d}.png"
                        _save_frame(rendered_frame, frame_path)
                        frame_paths.append(frame_path)
                        frame_count += 1

                        outcome = interpret_step_outcome(terminated=terminated, truncated=truncated, info=info)
                        if outcome.terminal:
                            success = outcome.success
                            terminal_reason = outcome.terminal_reason
                            break

                    elapsed = time.perf_counter() - start_time
                    task_successes += int(success)
                    total_successes += int(success)
                    total_episodes += 1

                    timing = {
                        "episode_seconds": elapsed,
                        "steps_executed": frame_count,
                    }
                    video_path: str | None = None
                    if cfg.save_videos and frame_paths:
                        video_output = get_video_path(run_dir, task_id=task_id, episode_index=episode_index, success=success)
                        save_video_from_frames(frame_paths, video_output)
                        video_path = str(video_output)

                    metadata = build_episode_metadata(
                        task_id=task_id,
                        episode_index=episode_index,
                        success=success,
                        seed=episode_seed,
                        checkpoint_id=Path(str(cfg.pretrained_checkpoint)).name,
                        checkpoint_path=str(cfg.pretrained_checkpoint),
                        frame_dir=frame_dir,
                        timing=timing,
                        extra={
                            "terminal_reason": terminal_reason,
                            "image_source": image_source,
                            "mode": cfg.mode,
                            "video_path": video_path,
                        },
                    )
                    append_episode_record(run_dir, metadata)
                    episode_records.append(metadata)

                    print(
                        f"task={task_id} episode={episode_index} success={success} terminal_reason={terminal_reason} "
                        f"steps={frame_count} total_success={total_successes}/{total_episodes}"
                    )
                    log_file.write(
                        f"task={task_id} episode={episode_index} success={success} terminal_reason={terminal_reason} "
                        f"steps={frame_count} total_success={total_successes}/{total_episodes}\n"
                    )
                    log_file.flush()
            finally:
                env.close()

            per_task_success_rate[task_id] = float(task_successes) / float(episodes_per_task)

        average_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
        exemplar_manifest = build_exemplar_manifest(episode_records, task_ids=selected_task_ids)

        manifest_payload = {
            "run_id": run_id,
            "backend": backend_info,
            "tasks": selected_task_ids,
            "mode": cfg.mode,
            "config": asdict(cfg),
            "episode_count_per_task": episodes_per_task,
            "assumption_ledger_path": str(assumption_ledger_path),
            "local_log_filepath": local_log_filepath,
            "exemplars": exemplar_manifest,
        }
        write_manifest(run_dir, manifest_payload)

        summary_payload = {
            "tasks": selected_task_ids,
            "backend": backend_info,
            "per_task_success_rate": per_task_success_rate,
            "average_success_rate": average_success_rate,
            "checkpoint": str(cfg.pretrained_checkpoint),
            "maniskill_version": _resolve_maniskill_version(),
            "seed_config": {
                "seed": cfg.seed,
                "mode": cfg.mode,
            },
            "episode_count_per_task": episodes_per_task,
            "artifact_paths": artifact_paths,
            "assumptions": {
                "items": assumptions,
                "ledger_path": str(assumption_ledger_path),
            },
        }
        summary_path = write_summary(run_dir, summary_payload)

        print(f"average_success_rate={average_success_rate:.4f}")
        print(f"summary_path={summary_path}")
        print(f"manifest_path={artifact_paths['manifest']}")
        print(f"episodes_path={artifact_paths['episodes']}")
        log_file.write(f"average_success_rate={average_success_rate:.4f}\n")
        log_file.write(f"summary_path={summary_path}\n")
        log_file.write(f"manifest_path={artifact_paths['manifest']}\n")
        log_file.write(f"episodes_path={artifact_paths['episodes']}\n")
        log_file.flush()
    finally:
        log_file.close()


if __name__ == "__main__":
    entrypoint: Any = eval_maniskill
    entrypoint()
