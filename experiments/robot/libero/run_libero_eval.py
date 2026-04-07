"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.
"""

# pyright: reportMissingImports=false

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.robot.maniskill.artifacts import (
    append_episode_record,
    build_artifact_paths,
    build_episode_metadata,
    build_exemplar_manifest,
    create_run_layout,
    get_video_path,
    write_manifest,
)
from experiments.robot.workflow_logging import child_launch_metadata_from_env, emit_breadcrumb, failure_metadata_from_exception

def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    unnorm_key: str = ""
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    controller_log_path: str = ""
    artifact_root: str = "rollouts/libero"          # Artifact root for machine-readable outputs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def build_libero_config_from_workflow_request(workflow_request: dict[str, Any]) -> GenerateConfig:
    if bool(workflow_request.get("cancelled")):
        reason = str(workflow_request.get("cancellation_reason", "USER_CANCELLED"))
        raise ValueError(reason)

    LIBERO_WORKLOAD_KEYS = {"openvla_libero", "openvla_libero_ft", "openpi_libero"}
    selection = str(workflow_request.get("selection", "")).strip()
    if selection not in LIBERO_WORKLOAD_KEYS:
        supported = ", ".join(sorted(LIBERO_WORKLOAD_KEYS))
        raise ValueError(f"UNSUPPORTED_DIRECT_WORKLOAD: {selection or '<blank>'}. Supported values: {supported}")

    workload_details = workflow_request.get("workload_details", {})
    if not isinstance(workload_details, dict) or selection not in workload_details:
        raise ValueError(f"INVALID_WORKFLOW_REQUEST: missing workload details for `{selection}`.")

    workload_detail = workload_details[selection]
    if not isinstance(workload_detail, dict):
        raise ValueError(f"INVALID_WORKFLOW_REQUEST: workload detail for `{selection}` must be an object.")
    if str(workload_detail.get("benchmark", "")).strip() != "libero":
        raise ValueError(f"UNSUPPORTED_DIRECT_WORKLOAD: `{selection}` is not a LIBERO workload.")

    model_family = str(workload_detail.get("model_family", "")).strip()
    if model_family not in {"openvla", "pi0"}:
        raise ValueError(f"INVALID_WORKFLOW_REQUEST: unsupported LIBERO model family `{model_family}`.")

    checkpoint_map = workflow_request.get("checkpoint_map", {})
    checkpoint = str(checkpoint_map.get(selection, "")).strip() if isinstance(checkpoint_map, dict) else ""
    if not checkpoint:
        raise ValueError(f"INVALID_WORKFLOW_REQUEST: missing checkpoint mapping for `{selection}`.")

    task_suite_name = str(workflow_request.get("libero_task_suite_name", "")).strip()
    if not task_suite_name:
        raise ValueError("INVALID_WORKFLOW_REQUEST: missing LIBERO task suite.")

    raw_num_trials = workflow_request.get("libero_num_trials_per_task")
    raw_num_trials_text = str(raw_num_trials).strip() if raw_num_trials is not None else ""
    if not raw_num_trials_text:
        raise ValueError("INVALID_WORKFLOW_REQUEST: missing LIBERO trial count.")
    try:
        num_trials_per_task = int(raw_num_trials_text)
    except ValueError as exc:
        raise ValueError("INVALID_WORKFLOW_REQUEST: missing LIBERO trial count.") from exc

    artifact_root = str(workflow_request.get("artifact_root", GenerateConfig.artifact_root)).strip() or GenerateConfig.artifact_root
    return GenerateConfig(
        model_family=model_family,
        pretrained_checkpoint=checkpoint,
        task_suite_name=task_suite_name,
        num_trials_per_task=num_trials_per_task,
        artifact_root=artifact_root,
    )


def _eval_libero_impl(cfg: GenerateConfig) -> dict[str, Any]:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    from experiments.robot.robot_utils import DATE_TIME

    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    controller_log_path = str(cfg.controller_log_path).strip()
    if controller_log_path:
        local_log_path = Path(controller_log_path)
        local_log_path.parent.mkdir(parents=True, exist_ok=True)
        local_log_filepath = str(local_log_path)
    else:
        os.makedirs(cfg.local_log_dir, exist_ok=True)
        local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_mode = "a" if controller_log_path and Path(local_log_filepath).exists() else "w"
    log_file = open(local_log_filepath, log_mode)
    print(f"Logging to local log file: {local_log_filepath}")

    run_dir = create_run_layout(run_id=run_id, artifact_root=cfg.artifact_root)
    artifact_paths = build_artifact_paths(run_dir)

    try:
        launch_metadata = child_launch_metadata_from_env(
            default_launch_path="direct_libero_runner",
            defaults={
                "benchmark": "libero",
                "model_family": cfg.model_family,
            },
        )
        emit_breadcrumb(launch_metadata, log_file=log_file)

        import wandb
        from libero.libero import benchmark

        from experiments.robot.libero.libero_utils import (
            get_libero_dummy_action,
            get_libero_env,
            get_libero_image,
            quat2axisangle,
            save_rollout_video,
        )
        from experiments.robot.openvla_utils import get_processor
        from experiments.robot.robot_utils import (
            get_action,
            get_image_resize_size,
            get_model,
            invert_gripper_action,
            normalize_gripper_action,
            set_seed_everywhere,
        )

        set_seed_everywhere(cfg.seed)
        cfg.unnorm_key = cfg.task_suite_name

        model = get_model(cfg)

        if cfg.model_family == "openvla":
            if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
                cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
            assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        processor = None
        if cfg.model_family == "openvla":
            processor = get_processor(cfg)

        if cfg.use_wandb:
            wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                name=run_id,
            )

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        print(f"Task suite: {cfg.task_suite_name}")
        log_file.write(f"Task suite: {cfg.task_suite_name}\n")

        resize_size = get_image_resize_size(cfg)
        total_episodes, total_successes = 0, 0
        task_labels: list[str] = []
        task_descriptions: dict[str, str] = {}
        per_task_success_rate: dict[str, float] = {}
        episode_records: list[dict[str, Any]] = []

        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            task = task_suite.get_task(task_id)
            task_label = f"{cfg.task_suite_name}:{task_id}"
            task_labels.append(task_label)
            initial_states = task_suite.get_task_init_states(task_id)

            env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
            task_descriptions[task_label] = task_description

            task_episodes, task_successes = 0, 0
            try:
                for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                    print(f"\nTask: {task_description}")
                    log_file.write(f"\nTask: {task_description}\n")

                    env.reset()
                    obs = env.set_init_state(initial_states[episode_idx])

                    start_time = time.perf_counter()
                    t = 0
                    policy_steps = 0
                    replay_images = []
                    done = False
                    terminal_reason = "max_steps"
                    error_message = None
                    if cfg.task_suite_name == "libero_spatial":
                        max_steps = 220
                    elif cfg.task_suite_name == "libero_object":
                        max_steps = 280
                    elif cfg.task_suite_name == "libero_goal":
                        max_steps = 300
                    elif cfg.task_suite_name == "libero_10":
                        max_steps = 520
                    elif cfg.task_suite_name == "libero_90":
                        max_steps = 400
                    else:
                        raise ValueError(f"Unsupported LIBERO task suite: {cfg.task_suite_name}")

                    print(f"Starting episode {task_episodes + 1}...")
                    log_file.write(f"Starting episode {task_episodes + 1}...\n")
                    while t < max_steps + cfg.num_steps_wait:
                        try:
                            if t < cfg.num_steps_wait:
                                obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                                t += 1
                                continue

                            img = get_libero_image(obs, resize_size)
                            replay_images.append(img)

                            observation = {
                                "full_image": img,
                                "state": np.concatenate(
                                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                                ),
                            }

                            action = get_action(
                                cfg,
                                model,
                                observation,
                                task_description,
                                processor=processor,
                            )
                            action = normalize_gripper_action(action, binarize=True)
                            if cfg.model_family == "openvla":
                                action = invert_gripper_action(action)

                            obs, _, done, _ = env.step(action.tolist())
                            policy_steps += 1
                            if done:
                                task_successes += 1
                                total_successes += 1
                                terminal_reason = "success"
                                break
                            t += 1
                        except Exception as exc:
                            error_message = str(exc)
                            terminal_reason = "exception"
                            print(f"Caught exception: {exc}")
                            log_file.write(f"Caught exception: {exc}\n")
                            break

                    task_episodes += 1
                    total_episodes += 1
                    elapsed = time.perf_counter() - start_time

                    video_output_path = get_video_path(run_dir, task_id=task_label, episode_index=episode_idx, success=done)
                    video_path = save_rollout_video(
                        replay_images,
                        total_episodes,
                        success=done,
                        task_description=task_description,
                        log_file=log_file,
                        output_path=video_output_path,
                    )

                    metadata = build_episode_metadata(
                        task_id=task_label,
                        episode_index=episode_idx,
                        success=done,
                        seed=cfg.seed + episode_idx,
                        checkpoint_id=Path(str(cfg.pretrained_checkpoint)).name,
                        checkpoint_path=str(cfg.pretrained_checkpoint),
                        frame_dir="",
                        timing={
                            "episode_seconds": elapsed,
                            "steps_executed": policy_steps,
                        },
                        extra={
                            "task_suite_name": cfg.task_suite_name,
                            "task_description": task_description,
                            "video_path": str(video_path),
                            "terminal_reason": terminal_reason,
                            "error_message": error_message,
                        },
                    )
                    append_episode_record(run_dir, metadata)
                    episode_records.append(metadata)

                    print(f"Success: {done}")
                    print(f"# episodes completed so far: {total_episodes}")
                    print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                    log_file.write(f"Success: {done}\n")
                    log_file.write(f"# episodes completed so far: {total_episodes}\n")
                    log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
                    log_file.flush()
            finally:
                env.close()

            per_task_success_rate[task_label] = float(task_successes) / float(task_episodes)
            print(f"Current task success rate: {per_task_success_rate[task_label]}")
            print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
            log_file.write(f"Current task success rate: {per_task_success_rate[task_label]}\n")
            log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
            log_file.flush()
            if cfg.use_wandb:
                wandb.log(
                    {
                        f"success_rate/{task_label}": per_task_success_rate[task_label],
                        f"num_episodes/{task_label}": task_episodes,
                    }
                )

        average_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
        exemplar_manifest = build_exemplar_manifest(episode_records, task_ids=task_labels)
        manifest_payload = {
            "run_id": run_id,
            "benchmark": "libero",
            "model_family": cfg.model_family,
            "task_suite_name": cfg.task_suite_name,
            "tasks": task_labels,
            "task_descriptions": task_descriptions,
            "config": asdict(cfg),
            "num_trials_per_task": cfg.num_trials_per_task,
            "local_log_filepath": local_log_filepath,
            "artifact_paths": artifact_paths,
            "exemplars": exemplar_manifest,
        }
        manifest_path = write_manifest(run_dir, manifest_payload)

        summary_payload = {
            "run_id": run_id,
            "benchmark": "libero",
            "model_family": cfg.model_family,
            "task_suite_name": cfg.task_suite_name,
            "tasks": task_labels,
            "task_descriptions": task_descriptions,
            "per_task_success_rate": per_task_success_rate,
            "average_success_rate": average_success_rate,
            "checkpoint": str(cfg.pretrained_checkpoint),
            "seed_config": {
                "seed": cfg.seed,
            },
            "episode_count_per_task": cfg.num_trials_per_task,
            "artifact_paths": artifact_paths,
        }
        summary_path = _write_json(Path(artifact_paths["summary"]), summary_payload)

        print(f"average_success_rate={average_success_rate:.4f}")
        print(f"summary_path={summary_path}")
        print(f"manifest_path={manifest_path}")
        print(f"episodes_path={artifact_paths['episodes']}")
        print(f"run_dir={run_dir}")
        log_file.write(f"average_success_rate={average_success_rate:.4f}\n")
        log_file.write(f"summary_path={summary_path}\n")
        log_file.write(f"manifest_path={manifest_path}\n")
        log_file.write(f"episodes_path={artifact_paths['episodes']}\n")
        log_file.write(f"run_dir={run_dir}\n")
        log_file.flush()

        if cfg.use_wandb:
            wandb.log(
                {
                    "success_rate/total": average_success_rate,
                    "num_episodes/total": total_episodes,
                }
            )
            wandb.save(local_log_filepath)
        return {
            "summary_path": str(summary_path),
            "manifest_path": str(manifest_path),
            "episodes_path": str(artifact_paths["episodes"]),
            "run_dir": str(run_dir),
            "average_success_rate": average_success_rate,
            "per_task_success_rate": per_task_success_rate,
            "checkpoint": str(cfg.pretrained_checkpoint),
            "artifact_paths": artifact_paths,
            "launch_metadata": launch_metadata,
        }
    except BaseException as exc:
        failure_details = failure_metadata_from_exception(exc, failure_phase="libero_runner_execution")
        emit_breadcrumb(
            {
                **child_launch_metadata_from_env(
                    default_launch_path="direct_libero_runner",
                    defaults={
                        "benchmark": "libero",
                        "model_family": cfg.model_family,
                    },
                ),
                **failure_details,
            },
            log_file=log_file,
        )
        raise
    finally:
        log_file.close()