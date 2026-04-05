"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.maniskill.artifacts import (
    append_episode_record,
    build_artifact_paths,
    build_episode_metadata,
    build_exemplar_manifest,
    create_run_layout,
    get_video_path,
    write_manifest,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


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
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    artifact_root: str = "rollouts/libero"          # Artifact root for machine-readable outputs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

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

    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    run_dir = create_run_layout(run_id=run_id, artifact_root=cfg.artifact_root)
    artifact_paths = build_artifact_paths(run_dir)

    try:
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
    finally:
        log_file.close()


if __name__ == "__main__":
    eval_libero()
