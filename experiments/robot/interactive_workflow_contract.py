from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from experiments.robot.maniskill.defaults import ARTIFACT_ROOT as MANISKILL_ARTIFACT_ROOT
from experiments.robot.maniskill.defaults import EPISODE_COUNT_DEFAULTS as MANISKILL_EPISODE_COUNT_DEFAULTS

WORKFLOW_KEYS = (
    "openvla_maniskill_ft",
    "openpi_maniskill",
    "openvla_libero",
    "openvla_libero_ft",
    "openpi_libero",
    "all",
)

PROMPT_SEQUENCE = (
    "workload selection",
    "mode",
    "parallelism request",
    "artifact root override",
    "confirmation",
)

MODE_KEYS = ("smoke", "full")
PARALLELISM_ANSWERS = {"y": True, "n": False}
CONFIRMATION_ANSWERS = {"y": True, "n": False}
CHECKPOINT_MAP = {
    "openvla_maniskill_ft": "Juelg/openvla-7b-finetuned-maniskill",
    "openpi_maniskill": "gs://openpi-assets/checkpoints/pi05_libero",
    "openvla_libero": "openvla/openvla-7b",
    "openvla_libero_ft": "Juelg/openvla-7b-finetuned-maniskill",
    "openpi_libero": "gs://openpi-assets/checkpoints/pi05_libero",
}
LIBERO_TASK_SUITE = "libero_spatial"
LIBERO_NUM_TRIALS_BY_MODE = {"smoke": 2, "full": 50}
LIBERO_ARTIFACT_ROOT = "rollouts/libero"
CANONICAL_PARENT_ARTIFACT_ROOT = "rollouts/cluster_workflow"
PARENT_WORKFLOW_SUMMARY_TEMPLATE = "rollouts/cluster_workflow/{session_id}/workflow_summary.json"
PARENT_WORKFLOW_SUMMARY_CONTRACT = "rollouts/cluster_workflow/<session_id>/workflow_summary.json"
CLEANUP_ORDER = ("introduce", "validate", "remove")
LEGACY_LAUNCHER_POLICY = "Legacy launchers are preserved until parity QA completes, then eligible for removal."

_CONCRETE_WORKFLOW_KEYS = tuple(key for key in WORKFLOW_KEYS if key != "all")
_VALID_MODES = set(MODE_KEYS)
_VALID_BOOLEAN_ANSWERS = {"y", "n"}
_WORKLOAD_DETAILS = {
    "openvla_maniskill_ft": {
        "benchmark": "maniskill",
        "model_family": "openvla",
        "runner": "experiments/robot/maniskill/run_maniskill_eval.py",
        "artifact_root": MANISKILL_ARTIFACT_ROOT,
    },
    "openpi_maniskill": {
        "benchmark": "maniskill",
        "model_family": "pi0",
        "runner": "experiments/robot/maniskill/run_maniskill_eval.py",
        "artifact_root": MANISKILL_ARTIFACT_ROOT,
    },
    "openvla_libero": {
        "benchmark": "libero",
        "model_family": "openvla",
        "runner": "experiments/robot/libero/run_libero_eval.py",
        "artifact_root": LIBERO_ARTIFACT_ROOT,
    },
    "openvla_libero_ft": {
        "benchmark": "libero",
        "model_family": "openvla",
        "runner": "experiments/robot/libero/run_libero_eval.py",
        "artifact_root": LIBERO_ARTIFACT_ROOT,
    },
    "openpi_libero": {
        "benchmark": "libero",
        "model_family": "pi0",
        "runner": "experiments/robot/libero/run_libero_eval.py",
        "artifact_root": LIBERO_ARTIFACT_ROOT,
    },
}

WORKFLOW_DEFAULTS: dict[str, object] = {
    "canonical_shell_entrypoint": "cluster/run_cluster_workflow.sh",
    "canonical_python_entrypoint": "experiments/robot/interactive_cluster_workflow.py",
    "canonical_maniskill_entrypoint": "experiments/robot/maniskill/run_maniskill_eval.py",
    "canonical_libero_entrypoint": "experiments/robot/libero/run_libero_eval.py",
    "canonical_parent_artifact_root": CANONICAL_PARENT_ARTIFACT_ROOT,
    "parent_workflow_summary_template": PARENT_WORKFLOW_SUMMARY_TEMPLATE,
    "parent_workflow_summary_contract": PARENT_WORKFLOW_SUMMARY_CONTRACT,
    "mode_keys": MODE_KEYS,
    "parallelism_answers": PARALLELISM_ANSWERS,
    "confirmation_answers": CONFIRMATION_ANSWERS,
    "checkpoint_map": CHECKPOINT_MAP,
    "cleanup_order": CLEANUP_ORDER,
    "legacy_launcher_policy": LEGACY_LAUNCHER_POLICY,
    "libero_task_suite": LIBERO_TASK_SUITE,
    "libero_num_trials_by_mode": LIBERO_NUM_TRIALS_BY_MODE,
    "maniskill_episode_count_defaults": dict(MANISKILL_EPISODE_COUNT_DEFAULTS),
}


def _normalize_text(value: object) -> str:
    return str(value).strip()


def _normalize_choice(*, value: object, field_name: str, supported_values: Sequence[str]) -> str:
    normalized = _normalize_text(value).lower()
    if normalized not in supported_values:
        supported = ", ".join(supported_values)
        raise ValueError(f"INVALID_{field_name}: {normalized or '<blank>'}. Supported values: {supported}")
    return normalized


def _normalize_workload_selection(
    *,
    value: object,
    supported_values: Sequence[str],
    allow_multiple: bool,
) -> str:
    normalized = _normalize_text(value).lower()
    if not allow_multiple:
        return _normalize_choice(value=normalized, field_name="WORKLOAD_KEY", supported_values=supported_values)

    if normalized == "all":
        if "all" not in supported_values:
            supported = ", ".join(supported_values)
            raise ValueError(f"INVALID_WORKLOAD_KEY: all. Supported values: {supported}")
        return normalized

    requested = [item.strip().lower() for item in normalized.split(",") if item.strip()]
    if not requested:
        supported = ", ".join(supported_values)
        raise ValueError(f"INVALID_WORKLOAD_KEY: <blank>. Supported values: {supported}")

    if "all" in requested:
        supported = ", ".join(supported_values)
        raise ValueError(f"INVALID_WORKLOAD_KEY: {normalized}. Supported values: {supported}")

    ordered_unique: list[str] = []
    seen: set[str] = set()
    for workload_key in requested:
        if workload_key not in supported_values:
            supported = ", ".join(supported_values)
            raise ValueError(f"INVALID_WORKLOAD_KEY: {workload_key}. Supported values: {supported}")
        if workload_key in seen:
            continue
        ordered_unique.append(workload_key)
        seen.add(workload_key)
    return ",".join(ordered_unique)


def _normalize_artifact_root(
    raw_value: object,
    *,
    default_artifact_root: str = CANONICAL_PARENT_ARTIFACT_ROOT,
) -> tuple[str, bool]:
    normalized = _normalize_text(raw_value)
    if not normalized:
        return str(Path(default_artifact_root)), False
    return str(Path(normalized)), True


def _resolve_workloads(selection: str) -> list[str]:
    if selection == "all":
        return list(_CONCRETE_WORKFLOW_KEYS)
    return [item.strip() for item in selection.split(",") if item.strip()]


def _build_mode_payload(mode: str) -> dict[str, object]:
    maniskill_episodes_per_task = int(MANISKILL_EPISODE_COUNT_DEFAULTS[mode])
    libero_num_trials_per_task = int(LIBERO_NUM_TRIALS_BY_MODE[mode])
    return {
        "selected_mode": mode,
        "mode_semantics": {
            "maniskill": {
                "runner_mode": mode,
                "episodes_per_task": maniskill_episodes_per_task,
            },
            "libero": {
                "task_suite_name": LIBERO_TASK_SUITE,
                "num_trials_per_task": libero_num_trials_per_task,
            },
        },
        "maniskill_mode": mode,
        "maniskill_episodes_per_task": maniskill_episodes_per_task,
        "libero_task_suite_name": LIBERO_TASK_SUITE,
        "libero_num_trials_per_task": libero_num_trials_per_task,
    }


def _build_workload_payload(resolved_workloads: list[str]) -> dict[str, object]:
    workload_details = {
        workload_key: dict(_WORKLOAD_DETAILS[workload_key], checkpoint=CHECKPOINT_MAP[workload_key])
        for workload_key in resolved_workloads
    }
    return {
        "workloads": resolved_workloads,
        "resolved_workloads": resolved_workloads,
        "workload_count": len(resolved_workloads),
        "workload_details": workload_details,
        "checkpoint_map": {workload_key: CHECKPOINT_MAP[workload_key] for workload_key in resolved_workloads},
    }


def build_workflow_request_preview(
    *,
    selection: object,
    mode: object,
    parallel: object,
    artifact_root: object,
    supported_workload_keys: Sequence[str] | None = None,
    default_artifact_root: str = CANONICAL_PARENT_ARTIFACT_ROOT,
    allow_multiple_workload_selection: bool = False,
) -> dict[str, object]:
    allowed_workloads = tuple(str(key) for key in (supported_workload_keys or WORKFLOW_KEYS))
    workload_selection = _normalize_workload_selection(
        value=selection,
        supported_values=allowed_workloads,
        allow_multiple=allow_multiple_workload_selection,
    )
    selected_mode = _normalize_choice(value=mode, field_name="MODE", supported_values=list(MODE_KEYS))
    parallel_choice = _normalize_choice(value=parallel, field_name="PARALLELISM_REQUEST", supported_values=["y", "n"])
    resolved_artifact_root, artifact_root_overridden = _normalize_artifact_root(
        artifact_root,
        default_artifact_root=default_artifact_root,
    )
    resolved_workloads = _resolve_workloads(workload_selection)
    payload = {
        "cancelled": False,
        "confirmed": False,
        "status": "preview_pending_confirmation",
        "selection": workload_selection,
        "selected_mode": selected_mode,
        "parallel": PARALLELISM_ANSWERS[parallel_choice],
        "parallel_requested": PARALLELISM_ANSWERS[parallel_choice],
        "artifact_root": resolved_artifact_root,
        "artifact_root_overridden": artifact_root_overridden,
        "canonical_shell_entrypoint": WORKFLOW_DEFAULTS["canonical_shell_entrypoint"],
        "canonical_python_entrypoint": WORKFLOW_DEFAULTS["canonical_python_entrypoint"],
        "parent_workflow_summary_template": PARENT_WORKFLOW_SUMMARY_TEMPLATE,
        "parent_workflow_summary_contract": PARENT_WORKFLOW_SUMMARY_CONTRACT,
        "cleanup_order": tuple(CLEANUP_ORDER),
        "legacy_launcher_policy": LEGACY_LAUNCHER_POLICY,
    }
    payload.update(_build_workload_payload(resolved_workloads))
    payload.update(_build_mode_payload(selected_mode))
    return payload


def _build_cancelled_payload(
    *,
    selection: str,
    mode: str,
    parallel: str,
    artifact_root: str,
    artifact_root_overridden: bool,
) -> dict[str, object]:
    return {
        "cancelled": True,
        "confirmed": False,
        "status": "cancelled",
        "selection": selection,
        "selected_mode": mode,
        "parallel": PARALLELISM_ANSWERS[parallel],
        "artifact_root": artifact_root,
        "artifact_root_overridden": artifact_root_overridden,
        "workloads": [],
        "resolved_workloads": [],
        "workload_count": 0,
        "workload_details": {},
        "checkpoint_map": {},
        "mode_semantics": {},
        "maniskill_mode": mode,
        "maniskill_episodes_per_task": None,
        "libero_task_suite_name": LIBERO_TASK_SUITE,
        "libero_num_trials_per_task": None,
        "cancellation_reason": "USER_CANCELLED",
    }


def validate_workflow_request(
    *,
    selection: object,
    mode: object,
    parallel: object,
    artifact_root: object,
    confirm: object,
    supported_workload_keys: Sequence[str] | None = None,
    default_artifact_root: str = CANONICAL_PARENT_ARTIFACT_ROOT,
    allow_multiple_workload_selection: bool = False,
) -> dict[str, object]:
    preview_payload = build_workflow_request_preview(
        selection=selection,
        mode=mode,
        parallel=parallel,
        artifact_root=artifact_root,
        supported_workload_keys=supported_workload_keys,
        default_artifact_root=default_artifact_root,
        allow_multiple_workload_selection=allow_multiple_workload_selection,
    )
    confirm_choice = _normalize_choice(value=confirm, field_name="CONFIRMATION", supported_values=["y", "n"])

    if not CONFIRMATION_ANSWERS[confirm_choice]:
        return _build_cancelled_payload(
            selection=str(preview_payload["selection"]),
            mode=str(preview_payload["selected_mode"]),
            parallel="y" if bool(preview_payload["parallel_requested"]) else "n",
            artifact_root=str(preview_payload["artifact_root"]),
            artifact_root_overridden=bool(preview_payload["artifact_root_overridden"]),
        )

    payload = dict(preview_payload)
    payload.update({"confirmed": True, "status": "ready"})
    return payload


def prompt_for_workflow_request(
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], object] | None = None,
    preview_callback: Callable[[dict[str, object]], object] | None = None,
    supported_workload_keys: Sequence[str] | None = None,
    default_artifact_root: str = CANONICAL_PARENT_ARTIFACT_ROOT,
    allow_multiple_workload_selection: bool = False,
) -> dict[str, object]:
    allowed_workloads = tuple(str(key) for key in (supported_workload_keys or WORKFLOW_KEYS))
    if output_fn is not None:
        _ = output_fn("Available workloads: " + ", ".join(allowed_workloads))
        _ = output_fn("Modes: smoke, full")
        _ = output_fn("Parallelism request: y or n")
        _ = output_fn("Artifact root override: blank uses " + str(Path(default_artifact_root)))

    selection = input_fn("Select workload: ")
    mode = input_fn("Select mode [smoke/full]: ")
    parallel = input_fn("Request parallel preparation where safe? [y/n]: ")
    artifact_root = input_fn("Override artifact root (blank for default): ")

    preview_payload = build_workflow_request_preview(
        selection=selection,
        mode=mode,
        parallel=parallel,
        artifact_root=artifact_root,
        supported_workload_keys=allowed_workloads,
        default_artifact_root=default_artifact_root,
        allow_multiple_workload_selection=allow_multiple_workload_selection,
    )
    if preview_callback is not None:
        _ = preview_callback(preview_payload)

    confirm = input_fn("Confirm workflow request? [y/n]: ")
    return validate_workflow_request(
        selection=selection,
        mode=mode,
        parallel=parallel,
        artifact_root=artifact_root,
        confirm=confirm,
        supported_workload_keys=allowed_workloads,
        default_artifact_root=default_artifact_root,
        allow_multiple_workload_selection=allow_multiple_workload_selection,
    )


__all__ = [
    "WORKFLOW_KEYS",
    "PROMPT_SEQUENCE",
    "WORKFLOW_DEFAULTS",
    "build_workflow_request_preview",
    "prompt_for_workflow_request",
    "validate_workflow_request",
]
