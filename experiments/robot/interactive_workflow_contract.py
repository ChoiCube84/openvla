from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Callable, cast

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
    "gpu choice",
    "confirmation",
)

MODE_KEYS = ("smoke", "full")
PARALLELISM_ANSWERS = {"y": True, "n": False}
CONFIRMATION_ANSWERS = {"y": True, "n": False}
DEFAULT_WORKLOAD_KEY = WORKFLOW_KEYS[0]
DEFAULT_MODE_KEY = MODE_KEYS[0]
DEFAULT_PARALLELISM_CHOICE = "n"
DEFAULT_CONFIRMATION_CHOICE = "n"
DEFAULT_GPU_CHOICE = "auto"
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
    "default_workload_key": DEFAULT_WORKLOAD_KEY,
    "default_mode_key": DEFAULT_MODE_KEY,
    "default_parallelism_choice": DEFAULT_PARALLELISM_CHOICE,
    "default_confirmation_choice": DEFAULT_CONFIRMATION_CHOICE,
    "default_gpu_choice": DEFAULT_GPU_CHOICE,
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


def _normalize_choice_with_default(
    *,
    value: object,
    field_name: str,
    supported_values: Sequence[str],
    default_value: str,
) -> tuple[str, bool]:
    normalized = _normalize_text(value).lower()
    if not normalized:
        normalized = default_value.lower()
        return _normalize_choice(value=normalized, field_name=field_name, supported_values=supported_values), True
    return _normalize_choice(value=normalized, field_name=field_name, supported_values=supported_values), False


def _normalize_workload_selection(
    *,
    value: object,
    supported_values: Sequence[str],
    allow_multiple: bool,
    default_value: str,
) -> tuple[str, bool]:
    normalized = _normalize_text(value).lower()
    if not normalized:
        normalized = default_value.lower()
        used_default = True
    else:
        used_default = False
    if not allow_multiple:
        return (
            _normalize_choice(value=normalized, field_name="WORKLOAD_KEY", supported_values=supported_values),
            used_default,
        )

    if normalized == "all":
        if "all" not in supported_values:
            supported = ", ".join(supported_values)
            raise ValueError(f"INVALID_WORKLOAD_KEY: all. Supported values: {supported}")
        return normalized, used_default

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
    return ",".join(ordered_unique), used_default


def _normalize_artifact_root(
    raw_value: object,
    *,
    default_artifact_root: str = CANONICAL_PARENT_ARTIFACT_ROOT,
) -> tuple[str, bool, bool]:
    normalized = _normalize_text(raw_value)
    if not normalized:
        return str(Path(default_artifact_root)), False, True
    return str(Path(normalized)), True, False


def _render_option_list(options: Sequence[str], recommended: str) -> str:
    rendered: list[str] = []
    for option in options:
        if option == recommended:
            rendered.append(f"{option} (recommended/default)")
        else:
            rendered.append(option)
    return ", ".join(rendered)


def _build_prompt_contract(
    *,
    supported_workload_keys: Sequence[str],
    default_artifact_root: str,
    supported_gpu_choices: Sequence[str],
    default_gpu_choice: str,
) -> dict[str, dict[str, object]]:
    gpu_options = tuple(str(choice).strip().lower() for choice in supported_gpu_choices if str(choice).strip())
    if not gpu_options:
        gpu_options = (DEFAULT_GPU_CHOICE,)
    normalized_default_gpu_choice = _normalize_choice(
        value=default_gpu_choice,
        field_name="GPU_CHOICE",
        supported_values=gpu_options,
    )
    workload_options = tuple(str(key) for key in supported_workload_keys)
    normalized_default_workload = _normalize_choice(
        value=DEFAULT_WORKLOAD_KEY if DEFAULT_WORKLOAD_KEY in workload_options else workload_options[0],
        field_name="WORKLOAD_KEY",
        supported_values=workload_options,
    )
    return {
        "workload selection": {
            "name": "workload selection",
            "options": workload_options,
            "recommended": normalized_default_workload,
            "blank_behavior": f"blank selects {normalized_default_workload}",
            "display": "Workload selection options: "
            + _render_option_list(workload_options, normalized_default_workload),
            "prompt": "Select workload: ",
        },
        "mode": {
            "name": "mode",
            "options": MODE_KEYS,
            "recommended": DEFAULT_MODE_KEY,
            "blank_behavior": f"blank selects {DEFAULT_MODE_KEY}",
            "display": "Mode options: " + _render_option_list(MODE_KEYS, DEFAULT_MODE_KEY),
            "prompt": f"Select mode [{'/'.join(MODE_KEYS)}]: ",
        },
        "parallelism request": {
            "name": "parallelism request",
            "options": tuple(PARALLELISM_ANSWERS.keys()),
            "recommended": DEFAULT_PARALLELISM_CHOICE,
            "blank_behavior": f"blank selects {DEFAULT_PARALLELISM_CHOICE}",
            "display": "Parallelism request options: "
            + _render_option_list(tuple(PARALLELISM_ANSWERS.keys()), DEFAULT_PARALLELISM_CHOICE),
            "prompt": "Request parallel preparation where safe? [y/n]: ",
        },
        "artifact root override": {
            "name": "artifact root override",
            "options": (
                f"blank -> {str(Path(default_artifact_root))}",
                "<path> -> explicit override",
            ),
            "recommended": str(Path(default_artifact_root)),
            "blank_behavior": f"blank selects {str(Path(default_artifact_root))}",
            "display": (
                "Artifact root behavior: blank -> "
                f"{str(Path(default_artifact_root))} (recommended/default), <path> -> explicit override"
            ),
            "prompt": "Override artifact root (blank for default): ",
        },
        "gpu choice": {
            "name": "gpu choice",
            "options": gpu_options,
            "recommended": normalized_default_gpu_choice,
            "blank_behavior": f"blank selects {normalized_default_gpu_choice}",
            "display": "GPU choice options: " + _render_option_list(gpu_options, normalized_default_gpu_choice),
            "prompt": "Select GPU choice: ",
        },
        "confirmation": {
            "name": "confirmation",
            "options": tuple(CONFIRMATION_ANSWERS.keys()),
            "recommended": DEFAULT_CONFIRMATION_CHOICE,
            "blank_behavior": f"blank selects {DEFAULT_CONFIRMATION_CHOICE}",
            "display": (
                "Confirmation semantics: y -> launch workflow, n -> cancel "
                f"(recommended/default: {DEFAULT_CONFIRMATION_CHOICE})"
            ),
            "prompt": "Confirm workflow request? [y/n]: ",
        },
    }


def _prompt_contract_value(prompt_contract: dict[str, dict[str, object]], prompt_name: str, key: str) -> str:
    return str(prompt_contract[prompt_name][key])


def _prompt_contract_options(prompt_contract: dict[str, dict[str, object]], prompt_name: str) -> tuple[str, ...]:
    options = cast(Sequence[object], prompt_contract[prompt_name]["options"])
    return tuple(str(choice) for choice in options)


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
    gpu_choice: object = DEFAULT_GPU_CHOICE,
    supported_workload_keys: Sequence[str] | None = None,
    supported_gpu_choices: Sequence[str] | None = None,
    default_gpu_choice: str = DEFAULT_GPU_CHOICE,
    default_artifact_root: str = CANONICAL_PARENT_ARTIFACT_ROOT,
    allow_multiple_workload_selection: bool = False,
) -> dict[str, object]:
    allowed_workloads = tuple(str(key) for key in (supported_workload_keys or WORKFLOW_KEYS))
    prompt_contract = _build_prompt_contract(
        supported_workload_keys=allowed_workloads,
        default_artifact_root=default_artifact_root,
        supported_gpu_choices=tuple(str(choice) for choice in (supported_gpu_choices or (DEFAULT_GPU_CHOICE,))),
        default_gpu_choice=default_gpu_choice,
    )
    workload_selection, workload_defaulted = _normalize_workload_selection(
        value=selection,
        supported_values=allowed_workloads,
        allow_multiple=allow_multiple_workload_selection,
        default_value=str(prompt_contract["workload selection"]["recommended"]),
    )
    selected_mode, mode_defaulted = _normalize_choice_with_default(
        value=mode,
        field_name="MODE",
        supported_values=list(MODE_KEYS),
        default_value=_prompt_contract_value(prompt_contract, "mode", "recommended"),
    )
    parallel_choice, parallel_defaulted = _normalize_choice_with_default(
        value=parallel,
        field_name="PARALLELISM_REQUEST",
        supported_values=["y", "n"],
        default_value=_prompt_contract_value(prompt_contract, "parallelism request", "recommended"),
    )
    resolved_artifact_root, artifact_root_overridden, artifact_root_defaulted = _normalize_artifact_root(
        artifact_root,
        default_artifact_root=default_artifact_root,
    )
    resolved_gpu_choice, gpu_choice_defaulted = _normalize_choice_with_default(
        value=gpu_choice,
        field_name="GPU_CHOICE",
        supported_values=_prompt_contract_options(prompt_contract, "gpu choice"),
        default_value=_prompt_contract_value(prompt_contract, "gpu choice", "recommended"),
    )
    resolved_workloads = _resolve_workloads(workload_selection)
    payload = {
        "cancelled": False,
        "confirmed": False,
        "status": "preview_pending_confirmation",
        "prompt_sequence": PROMPT_SEQUENCE,
        "prompt_contract": prompt_contract,
        "selection": workload_selection,
        "selection_defaulted": workload_defaulted,
        "selected_mode": selected_mode,
        "mode_defaulted": mode_defaulted,
        "parallel": PARALLELISM_ANSWERS[parallel_choice],
        "parallel_requested": PARALLELISM_ANSWERS[parallel_choice],
        "parallel_choice": parallel_choice,
        "parallel_defaulted": parallel_defaulted,
        "artifact_root": resolved_artifact_root,
        "artifact_root_overridden": artifact_root_overridden,
        "artifact_root_defaulted": artifact_root_defaulted,
        "gpu_choice": resolved_gpu_choice,
        "gpu_choice_defaulted": gpu_choice_defaulted,
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
    preview_payload: dict[str, object],
    confirm_choice: str,
    confirmation_defaulted: bool,
) -> dict[str, object]:
    payload = dict(preview_payload)
    payload.update({
        "cancelled": True,
        "confirmed": False,
        "status": "cancelled",
        "workloads": [],
        "resolved_workloads": [],
        "workload_count": 0,
        "workload_details": {},
        "checkpoint_map": {},
        "mode_semantics": {},
        "maniskill_mode": str(preview_payload["selected_mode"]),
        "maniskill_episodes_per_task": None,
        "libero_task_suite_name": LIBERO_TASK_SUITE,
        "libero_num_trials_per_task": None,
        "confirm_choice": confirm_choice,
        "confirmation_defaulted": confirmation_defaulted,
        "cancellation_reason": "USER_CANCELLED",
    })
    return payload


def validate_workflow_request(
    *,
    selection: object,
    mode: object,
    parallel: object,
    artifact_root: object,
    confirm: object,
    gpu_choice: object = DEFAULT_GPU_CHOICE,
    supported_workload_keys: Sequence[str] | None = None,
    supported_gpu_choices: Sequence[str] | None = None,
    default_gpu_choice: str = DEFAULT_GPU_CHOICE,
    default_artifact_root: str = CANONICAL_PARENT_ARTIFACT_ROOT,
    allow_multiple_workload_selection: bool = False,
) -> dict[str, object]:
    preview_payload = build_workflow_request_preview(
        selection=selection,
        mode=mode,
        parallel=parallel,
        artifact_root=artifact_root,
        gpu_choice=gpu_choice,
        supported_workload_keys=supported_workload_keys,
        supported_gpu_choices=supported_gpu_choices,
        default_gpu_choice=default_gpu_choice,
        default_artifact_root=default_artifact_root,
        allow_multiple_workload_selection=allow_multiple_workload_selection,
    )
    confirm_choice, confirmation_defaulted = _normalize_choice_with_default(
        value=confirm,
        field_name="CONFIRMATION",
        supported_values=["y", "n"],
        default_value=_prompt_contract_value(
            cast(dict[str, dict[str, object]], preview_payload["prompt_contract"]),
            "confirmation",
            "recommended",
        ),
    )

    if not CONFIRMATION_ANSWERS[confirm_choice]:
        return _build_cancelled_payload(
            preview_payload=preview_payload,
            confirm_choice=confirm_choice,
            confirmation_defaulted=confirmation_defaulted,
        )

    payload = dict(preview_payload)
    payload.update(
        {
            "confirmed": True,
            "status": "ready",
            "confirm_choice": confirm_choice,
            "confirmation_defaulted": confirmation_defaulted,
        }
    )
    return payload


def prompt_for_workflow_request(
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], object] | None = None,
    preview_callback: Callable[[dict[str, object]], object] | None = None,
    supported_workload_keys: Sequence[str] | None = None,
    supported_gpu_choices: Sequence[str] | None = None,
    default_gpu_choice: str = DEFAULT_GPU_CHOICE,
    default_artifact_root: str = CANONICAL_PARENT_ARTIFACT_ROOT,
    allow_multiple_workload_selection: bool = False,
) -> dict[str, object]:
    allowed_workloads = tuple(str(key) for key in (supported_workload_keys or WORKFLOW_KEYS))
    prompt_contract = _build_prompt_contract(
        supported_workload_keys=allowed_workloads,
        default_artifact_root=default_artifact_root,
        supported_gpu_choices=tuple(str(choice) for choice in (supported_gpu_choices or (DEFAULT_GPU_CHOICE,))),
        default_gpu_choice=default_gpu_choice,
    )
    if output_fn is not None:
        for prompt_name in PROMPT_SEQUENCE:
            _ = output_fn(str(prompt_contract[prompt_name]["display"]))

    selection = input_fn(str(prompt_contract["workload selection"]["prompt"]))
    mode = input_fn(str(prompt_contract["mode"]["prompt"]))
    parallel = input_fn(str(prompt_contract["parallelism request"]["prompt"]))
    artifact_root = input_fn(str(prompt_contract["artifact root override"]["prompt"]))
    gpu_choice = input_fn(str(prompt_contract["gpu choice"]["prompt"]))

    preview_payload = build_workflow_request_preview(
        selection=selection,
        mode=mode,
        parallel=parallel,
        artifact_root=artifact_root,
        gpu_choice=gpu_choice,
        supported_workload_keys=allowed_workloads,
        supported_gpu_choices=_prompt_contract_options(prompt_contract, "gpu choice"),
        default_gpu_choice=default_gpu_choice,
        default_artifact_root=default_artifact_root,
        allow_multiple_workload_selection=allow_multiple_workload_selection,
    )
    if preview_callback is not None:
        _ = preview_callback(preview_payload)

    confirm = input_fn(str(prompt_contract["confirmation"]["prompt"]))
    return validate_workflow_request(
        selection=selection,
        mode=mode,
        parallel=parallel,
        artifact_root=artifact_root,
        confirm=confirm,
        gpu_choice=gpu_choice,
        supported_workload_keys=allowed_workloads,
        supported_gpu_choices=_prompt_contract_options(prompt_contract, "gpu choice"),
        default_gpu_choice=default_gpu_choice,
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
