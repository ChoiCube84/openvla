from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.robot.maniskill.defaults import TASK_IDS


DIAGNOSTICS_ROOT = REPO_ROOT / "rollouts" / "diagnostics"
SUMMARY_PATH = DIAGNOSTICS_ROOT / "experiment_matrix_summary.json"
MATRIX_SCHEMA_VERSION = "1.0"
BASELINE_HORIZON = 200
RAISED_HORIZON = 400
CLASSIFICATION_LABELS = [
    "checkpoint_load_bug",
    "prompt_mismatch",
    "stats_mismatch",
    "camera_preprocess_mismatch",
    "horizon_truncation",
    "likely_model_weakness",
]


MANISKILL_DIAGNOSTIC_EPISODES_PER_TASK = 50
OPENVLA_REPO_DEFAULT_CHECKPOINT = "openvla/openvla-7b"
OPENVLA_FINETUNED_DEFAULT_CHECKPOINT = "Juelg/openvla-7b-finetuned-maniskill"


def _maniskill_command(*, cell_id: str, checkpoint_value: str, max_steps_per_episode: int, artifact_root: Path) -> list[str]:
    command = [
        "python",
        "experiments/robot/maniskill/run_maniskill_eval.py",
        "--model_family",
        "openvla",
        "--mode",
        "full",
        "--task_ids",
        ",".join(TASK_IDS),
        "--episodes_per_task",
        str(MANISKILL_DIAGNOSTIC_EPISODES_PER_TASK),
        "--max_steps_per_episode",
        str(max_steps_per_episode),
        "--run_id_note",
        cell_id,
        "--artifact_root",
        str(artifact_root),
    ]
    if checkpoint_value:
        command.extend(["--pretrained_checkpoint", checkpoint_value])
    return command


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


def _parse_key_values(log_path: Path, keys: list[str]) -> dict[str, str]:
    parsed = {key: "" for key in keys}
    if not log_path.is_file():
        return parsed

    for line in log_path.read_text().splitlines():
        for key in keys:
            prefix = f"{key}="
            if line.startswith(prefix):
                parsed[key] = line.split("=", 1)[1].strip()
    return parsed


def _tail_nonempty_line(log_path: Path) -> str:
    if not log_path.is_file():
        return f"log_missing:{log_path}"
    for line in reversed(log_path.read_text().splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return "no_output"


def _maniskill_cell(
    *,
    cell_id: str,
    variant_id: str,
    checkpoint_label: str,
    checkpoint_env_key: str,
    horizon_label: str,
    max_steps_per_episode: int,
    compare_set_id: str,
    is_control: bool,
    baseline_cell_id: str,
) -> dict[str, Any]:
    cell_root = DIAGNOSTICS_ROOT / "cells" / cell_id
    artifact_root = cell_root / "artifacts"
    checkpoint_value = os.environ.get(checkpoint_env_key, "")
    return {
        "cell_id": cell_id,
        "benchmark": "maniskill",
        "model_family": "openvla",
        "model_variant": variant_id,
        "comparison_set_ids": [compare_set_id],
        "control_role": "control" if is_control else "treatment",
        "control_cell_id": cell_id if is_control else baseline_cell_id,
        "factor_values": {
            "checkpoint_label": checkpoint_label,
            "horizon_policy": horizon_label,
            "max_steps_per_episode": max_steps_per_episode,
            "task_ids": list(TASK_IDS),
            "seed": 7,
            "episodes_per_task": MANISKILL_DIAGNOSTIC_EPISODES_PER_TASK,
        },
        "control_policy": {
            "varied_axis": "max_steps_per_episode",
            "fixed_axes": [
                "model_family",
                "model_variant",
                "checkpoint_label",
                "task_ids",
                "seed",
                "prompt_source",
            ],
            "prompt_source": "ManiSkill task_id",
            "isolation_rule": "Only horizon changes within this comparison set.",
        },
        "launch_surface": {
            "kind": "python",
            "entrypoint": "experiments/robot/maniskill/run_maniskill_eval.py",
            "cwd": str(REPO_ROOT),
            "command": _maniskill_command(
                cell_id=cell_id,
                checkpoint_value=checkpoint_value,
                max_steps_per_episode=max_steps_per_episode,
                artifact_root=artifact_root,
            ),
            "env_overrides": {
                "OPENVLA_MANISKILL_SKIP_CONDA_ACTIVATE": os.environ.get(
                    "OPENVLA_MANISKILL_SKIP_CONDA_ACTIVATE",
                    "0",
                ),
            },
        },
        "child_result_paths": {
            "cell_root": str(cell_root),
            "runner_log_path": str(cell_root / "runner.log"),
            "execution_record_path": str(cell_root / "execution.json"),
            "summary_path": None,
            "manifest_path": None,
            "episodes_path": None,
            "run_dir": None,
        },
        "telemetry_references": {
            "stdout_keys": [
                "summary_path",
                "manifest_path",
                "episodes_path",
                "average_success_rate",
                "run_dir",
            ],
            "telemetry_paths": [
                str(cell_root / "runner.log"),
                str(cell_root / "execution.json"),
            ],
        },
        "classification_candidates": [
            "checkpoint_load_bug",
            "stats_mismatch",
            "camera_preprocess_mismatch",
            "horizon_truncation",
            "likely_model_weakness",
        ],
        "status": "planned",
    }


def _libero_cell(
    *,
    cell_id: str,
    model_family: str,
    variant_id: str,
    checkpoint_env_key: str,
    compare_set_id: str,
    is_control: bool,
    baseline_cell_id: str,
) -> dict[str, Any]:
    cell_root = DIAGNOSTICS_ROOT / "cells" / cell_id
    task_suite_name = os.environ.get("OPENVLA_DIAGNOSTICS_LIBERO_TASK_SUITE", "libero_spatial")
    num_trials_per_task = os.environ.get("OPENVLA_DIAGNOSTICS_LIBERO_NUM_TRIALS", "50")
    checkpoint_value = os.environ.get(checkpoint_env_key, "").strip()
    if not checkpoint_value and model_family == "openvla":
        if variant_id == "openvla_repo_default":
            checkpoint_value = OPENVLA_REPO_DEFAULT_CHECKPOINT
        elif variant_id == "openvla_finetuned_maniskill":
            checkpoint_value = OPENVLA_FINETUNED_DEFAULT_CHECKPOINT

    return {
        "cell_id": cell_id,
        "benchmark": "libero",
        "model_family": model_family,
        "model_variant": variant_id,
        "comparison_set_ids": [compare_set_id],
        "control_role": "control" if is_control else "treatment",
        "control_cell_id": cell_id if is_control else baseline_cell_id,
        "factor_values": {
            "checkpoint_label": variant_id,
            "task_suite_name": task_suite_name,
            "num_trials_per_task": int(num_trials_per_task),
            "seed": 7,
        },
        "control_policy": {
            "varied_axis": "model_variant",
            "fixed_axes": [
                "task_suite_name",
                "num_trials_per_task",
                "seed",
                "prompt_source",
            ],
            "prompt_source": "LIBERO task description",
            "isolation_rule": "Only model/checkpoint changes within this comparison set.",
        },
        "launch_surface": {
            "kind": "python",
            "entrypoint": "experiments/robot/libero/run_libero_eval.py",
            "cwd": str(REPO_ROOT),
            "command": [
                "python",
                "experiments/robot/libero/run_libero_eval.py",
                "--model_family",
                model_family,
                "--pretrained_checkpoint",
                checkpoint_value,
                "--task_suite_name",
                task_suite_name,
                "--num_trials_per_task",
                num_trials_per_task,
                "--seed",
                "7",
                "--use_wandb",
                "False",
                "--run_id_note",
                cell_id,
                "--artifact_root",
                str(cell_root / "artifacts"),
            ],
            "env_overrides": {},
        },
        "child_result_paths": {
            "cell_root": str(cell_root),
            "runner_log_path": str(cell_root / "runner.log"),
            "execution_record_path": str(cell_root / "execution.json"),
            "summary_path": None,
            "manifest_path": None,
            "episodes_path": None,
            "run_dir": None,
        },
        "telemetry_references": {
            "stdout_keys": [
                "summary_path",
                "manifest_path",
                "episodes_path",
                "average_success_rate",
                "run_dir",
            ],
            "telemetry_paths": [
                str(cell_root / "runner.log"),
                str(cell_root / "execution.json"),
            ],
        },
        "classification_candidates": [
            "checkpoint_load_bug",
            "prompt_mismatch",
            "stats_mismatch",
            "camera_preprocess_mismatch",
            "likely_model_weakness",
        ],
        "status": "planned",
    }


def build_matrix_summary(plan_only: bool, summary_path: Path) -> dict[str, Any]:
    raised_horizon = int(os.environ.get("OPENVLA_DIAGNOSTICS_RAISED_HORIZON", str(RAISED_HORIZON)))
    cells = [
        _maniskill_cell(
            cell_id="maniskill-openvla-repo-default-baseline",
            variant_id="openvla_repo_default",
            checkpoint_label="repo_default",
            checkpoint_env_key="OPENVLA_DIAGNOSTICS_OPENVLA_REPO_DEFAULT_CHECKPOINT",
            horizon_label="baseline",
            max_steps_per_episode=BASELINE_HORIZON,
            compare_set_id="maniskill-openvla-repo-default-horizon",
            is_control=True,
            baseline_cell_id="maniskill-openvla-repo-default-baseline",
        ),
        _maniskill_cell(
            cell_id="maniskill-openvla-repo-default-raised-horizon",
            variant_id="openvla_repo_default",
            checkpoint_label="repo_default",
            checkpoint_env_key="OPENVLA_DIAGNOSTICS_OPENVLA_REPO_DEFAULT_CHECKPOINT",
            horizon_label="raised_horizon",
            max_steps_per_episode=raised_horizon,
            compare_set_id="maniskill-openvla-repo-default-horizon",
            is_control=False,
            baseline_cell_id="maniskill-openvla-repo-default-baseline",
        ),
        _maniskill_cell(
            cell_id="maniskill-openvla-finetuned-baseline",
            variant_id="openvla_finetuned_maniskill",
            checkpoint_label="finetuned_maniskill",
            checkpoint_env_key="OPENVLA_DIAGNOSTICS_OPENVLA_FINETUNED_CHECKPOINT",
            horizon_label="baseline",
            max_steps_per_episode=BASELINE_HORIZON,
            compare_set_id="maniskill-openvla-finetuned-horizon",
            is_control=True,
            baseline_cell_id="maniskill-openvla-finetuned-baseline",
        ),
        _maniskill_cell(
            cell_id="maniskill-openvla-finetuned-raised-horizon",
            variant_id="openvla_finetuned_maniskill",
            checkpoint_label="finetuned_maniskill",
            checkpoint_env_key="OPENVLA_DIAGNOSTICS_OPENVLA_FINETUNED_CHECKPOINT",
            horizon_label="raised_horizon",
            max_steps_per_episode=raised_horizon,
            compare_set_id="maniskill-openvla-finetuned-horizon",
            is_control=False,
            baseline_cell_id="maniskill-openvla-finetuned-baseline",
        ),
        _libero_cell(
            cell_id="libero-openvla-repo-default",
            model_family="openvla",
            variant_id="openvla_repo_default",
            checkpoint_env_key="OPENVLA_DIAGNOSTICS_OPENVLA_REPO_DEFAULT_CHECKPOINT",
            compare_set_id="libero-model-family-control",
            is_control=True,
            baseline_cell_id="libero-openvla-repo-default",
        ),
        _libero_cell(
            cell_id="libero-openvla-finetuned",
            model_family="openvla",
            variant_id="openvla_finetuned_maniskill",
            checkpoint_env_key="OPENVLA_DIAGNOSTICS_OPENVLA_FINETUNED_CHECKPOINT",
            compare_set_id="libero-model-family-control",
            is_control=False,
            baseline_cell_id="libero-openvla-repo-default",
        ),
        _libero_cell(
            cell_id="libero-pi0",
            model_family="pi0",
            variant_id="pi0",
            checkpoint_env_key="OPENVLA_DIAGNOSTICS_PI0_CHECKPOINT",
            compare_set_id="libero-model-family-control",
            is_control=False,
            baseline_cell_id="libero-openvla-repo-default",
        ),
    ]

    comparison_sets = [
        {
            "comparison_set_id": "maniskill-openvla-repo-default-horizon",
            "description": "Repo-default OpenVLA ManiSkill baseline versus raised horizon.",
            "control_cell_id": "maniskill-openvla-repo-default-baseline",
            "treatment_cell_ids": ["maniskill-openvla-repo-default-raised-horizon"],
            "varied_axis": "max_steps_per_episode",
            "expected_signal_family": ["horizon_truncation"],
        },
        {
            "comparison_set_id": "maniskill-openvla-finetuned-horizon",
            "description": "Finetuned OpenVLA ManiSkill baseline versus raised horizon.",
            "control_cell_id": "maniskill-openvla-finetuned-baseline",
            "treatment_cell_ids": ["maniskill-openvla-finetuned-raised-horizon"],
            "varied_axis": "max_steps_per_episode",
            "expected_signal_family": ["horizon_truncation"],
        },
        {
            "comparison_set_id": "libero-model-family-control",
            "description": "LIBERO model/control set with prompts, seed, and task suite fixed.",
            "control_cell_id": "libero-openvla-repo-default",
            "treatment_cell_ids": ["libero-openvla-finetuned", "libero-pi0"],
            "varied_axis": "model_variant",
            "expected_signal_family": [
                "checkpoint_load_bug",
                "prompt_mismatch",
                "stats_mismatch",
                "camera_preprocess_mismatch",
                "likely_model_weakness",
            ],
        },
    ]

    evidence_classification = {
        "labels": CLASSIFICATION_LABELS,
        "candidate_rules": {
            "checkpoint_load_bug": {
                "inspect_comparison_set_ids": [
                    "maniskill-openvla-repo-default-horizon",
                    "maniskill-openvla-finetuned-horizon",
                    "libero-model-family-control",
                ],
                "notes": "Use checkpoint-constant cells to separate load failures from behavioral differences.",
            },
            "prompt_mismatch": {
                "inspect_comparison_set_ids": ["libero-model-family-control"],
                "notes": "LIBERO cells keep task suite fixed so prompt-family effects can be reviewed later.",
            },
            "stats_mismatch": {
                "inspect_comparison_set_ids": [
                    "maniskill-openvla-repo-default-horizon",
                    "maniskill-openvla-finetuned-horizon",
                    "libero-model-family-control",
                ],
                "notes": "Cross-benchmark consistency is recorded without concluding causality here.",
            },
            "camera_preprocess_mismatch": {
                "inspect_comparison_set_ids": [
                    "maniskill-openvla-repo-default-horizon",
                    "libero-model-family-control",
                ],
                "notes": "Telemetry paths point to child logs for later frame/preprocess audits.",
            },
            "horizon_truncation": {
                "inspect_comparison_set_ids": [
                    "maniskill-openvla-repo-default-horizon",
                    "maniskill-openvla-finetuned-horizon",
                ],
                "notes": "Baseline and raised-horizon cells are intentionally separate and stable.",
            },
            "likely_model_weakness": {
                "inspect_comparison_set_ids": [
                    "maniskill-openvla-repo-default-horizon",
                    "maniskill-openvla-finetuned-horizon",
                    "libero-model-family-control",
                ],
                "notes": "Only classify as model weakness after alternative mismatch buckets remain unsupported.",
            },
        },
    }

    return {
        "summary_schema_version": MATRIX_SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "plan_only": bool(plan_only),
        "output_root": str(DIAGNOSTICS_ROOT),
        "summary_path": str(summary_path),
        "control_policy": {
            "matrix_goal": "Controlled diagnostics matrix with one varied factor per comparison set.",
            "global_fixed_axes": ["seed", "benchmark prompts/tasks within each comparison set"],
            "control_rule": "Keep prompts, seeds, and tasks fixed while isolating a single factor.",
        },
        "comparison_sets": comparison_sets,
        "evidence_classification": evidence_classification,
        "cells": cells,
    }


def _execute_cell(cell: dict[str, Any]) -> dict[str, Any]:
    launch_surface = cell["launch_surface"]
    result_paths = cell["child_result_paths"]
    cell_root = Path(result_paths["cell_root"])
    log_path = Path(result_paths["runner_log_path"])
    execution_record_path = Path(result_paths["execution_record_path"])
    cell_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({key: value for key, value in launch_surface["env_overrides"].items() if value != ""})

    start_time = time.time()
    with log_path.open("w") as log_file:
        process = subprocess.run(
            launch_surface["command"],
            cwd=launch_surface["cwd"],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    end_time = time.time()

    execution_record: dict[str, Any] = {
        "cell_id": cell["cell_id"],
        "command": launch_surface["command"],
        "cwd": launch_surface["cwd"],
        "env_overrides": launch_surface["env_overrides"],
        "started_at_unix": start_time,
        "finished_at_unix": end_time,
        "duration_seconds": end_time - start_time,
        "exit_code": process.returncode,
        "status": "complete" if process.returncode == 0 else "failed",
        "error": None if process.returncode == 0 else _tail_nonempty_line(log_path),
    }

    stdout_keys = list(cell.get("telemetry_references", {}).get("stdout_keys", []))
    parsed = _parse_key_values(log_path, stdout_keys) if stdout_keys else {}
    if "summary_path" in parsed:
        result_paths["summary_path"] = parsed["summary_path"] or None
    if "manifest_path" in parsed:
        result_paths["manifest_path"] = parsed["manifest_path"] or None
    if "episodes_path" in parsed:
        result_paths["episodes_path"] = parsed["episodes_path"] or None
    if "run_dir" in parsed:
        result_paths["run_dir"] = parsed["run_dir"] or None
    execution_record["parsed_stdout"] = parsed

    _write_json(execution_record_path, execution_record)
    cell["status"] = execution_record["status"]
    cell["execution"] = execution_record
    return cell


def run_matrix(plan_only: bool, summary_path: Path) -> Path:
    summary = build_matrix_summary(plan_only=plan_only, summary_path=summary_path)
    if not plan_only:
        executed_cells: list[dict[str, Any]] = []
        for cell in summary["cells"]:
            executed_cells.append(_execute_cell(cell))
        summary["cells"] = executed_cells
    return _write_json(summary_path, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and optionally execute the diagnostics experiment matrix.")
    parser.add_argument("--plan-only", action="store_true", help="Only write the machine-readable matrix summary.")
    parser.add_argument(
        "--summary-path",
        default=str(SUMMARY_PATH),
        help="Path for the machine-readable experiment matrix summary.",
    )
    args = parser.parse_args()

    plan_only = args.plan_only or _env_flag("OPENVLA_DIAGNOSTICS_PLAN_ONLY")
    summary_path = Path(args.summary_path)
    written_path = run_matrix(plan_only=plan_only, summary_path=summary_path)
    print(written_path)


if __name__ == "__main__":
    main()
