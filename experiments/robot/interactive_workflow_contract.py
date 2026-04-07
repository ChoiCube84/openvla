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
)

PROMPT_SEQUENCE = (
    "workload selection",
    "mode",
    "artifact root override",
    "gpu number",
    "confirmation",
)

MODE_KEYS = ("smoke", "full")
CONFIRMATION_ANSWERS = {"y": True, "n": False}
DEFAULT_WORKLOAD_KEY = "all"
DEFAULT_MODE_KEY = "full"
SUPPORTED_GPU_NUMBERS = tuple(str(index) for index in range(8))
CHECKPOINT_MAP = {
    "openvla_maniskill_ft": "Juelg/openvla-7b-finetuned-maniskill",
    "openpi_maniskill": "gs://openpi-assets/checkpoints/pi05_libero",
    "openvla_libero": "openvla/openvla-7b",
    "openvla_libero_ft": "openvla/openvla-7b-finetuned-libero-spatial",
    "openpi_libero": "gs://openpi-assets/checkpoints/pi05_libero",
}
LIBERO_TASK_SUITE = "libero_spatial"
LIBERO_NUM_TRIALS_BY_MODE = {"smoke": 2, "full": 50}
LIBERO_ARTIFACT_ROOT = "rollouts/libero"
CANONICAL_PARENT_ARTIFACT_ROOT = "rollouts/cluster_workflow"

_CONCRETE_WORKFLOW_KEYS = WORKFLOW_KEYS
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


def _normalize_required_choice(*, value: object, field_name: str, supported_values: Sequence[str]) -> str:
    normalized = _normalize_text(value).lower()
    if not normalized:
        supported = ", ".join(supported_values)
        raise ValueError(f"INVALID_{field_name}: <blank>. Supported values: {supported}")
    return _normalize_choice(value=normalized, field_name=field_name, supported_values=supported_values)


def _normalize_workload_selection(*, value: object, supported_values: Sequence[str], default_value: str) -> tuple[str, bool]:
    normalized = _normalize_text(value).lower()
    if not normalized:
        normalized = default_value.lower()
        used_default = True
    else:
        used_default = False
    return (
        _normalize_choice(value=normalized, field_name="WORKLOAD_KEY", supported_values=supported_values),
        used_default,
    )


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


def _normalize_gpu_number(*, value: object, default_value: str) -> tuple[str, bool]:
    normalized = _normalize_text(value)
    if not normalized:
        return default_value, True
    try:
        canonical = str(int(normalized, 10))
    except ValueError as exc:
        supported = ", ".join(SUPPORTED_GPU_NUMBERS)
        raise ValueError(f"INVALID_GPU_NUMBER: {normalized}. Supported values: {supported}") from exc
    if canonical not in SUPPORTED_GPU_NUMBERS:
        supported = ", ".join(SUPPORTED_GPU_NUMBERS)
        raise ValueError(f"INVALID_GPU_NUMBER: {normalized}. Supported values: {supported}")
    return canonical, False


def _build_prompt_contract(
    *,
    supported_workload_keys: Sequence[str],
    default_artifact_root: str,
) -> dict[str, dict[str, object]]:
    workload_options = (DEFAULT_WORKLOAD_KEY, *(str(key) for key in supported_workload_keys))
    normalized_default_workload = _normalize_choice(
        value=DEFAULT_WORKLOAD_KEY,
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
        "gpu number": {
            "name": "gpu number",
            "options": SUPPORTED_GPU_NUMBERS,
            "recommended": "1",
            "blank_behavior": "blank selects 1",
            "display": "GPU number options: " + _render_option_list(SUPPORTED_GPU_NUMBERS, "1"),
            "prompt": "Select GPU number [0/1/2/3/4/5/6/7]: ",
        },
        "confirmation": {
            "name": "confirmation",
            "options": tuple(CONFIRMATION_ANSWERS.keys()),
            "recommended": None,
            "blank_behavior": "blank is invalid; explicit confirmation required",
            "display": "Confirmation semantics: y -> launch workflow, n -> cancel (no default)",
            "prompt": "Confirm workflow request? [y/n]: ",
        },
    }


def _prompt_contract_value(prompt_contract: dict[str, dict[str, object]], prompt_name: str, key: str) -> str:
    return str(prompt_contract[prompt_name][key])


def _resolve_workloads(selection: str) -> list[str]:
    if selection == DEFAULT_WORKLOAD_KEY:
        return list(WORKFLOW_KEYS)
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
    artifact_root: object,
    gpu_number: object,
    supported_workload_keys: Sequence[str] | None = None,
    default_artifact_root: str = CANONICAL_PARENT_ARTIFACT_ROOT,
) -> dict[str, object]:
    allowed_workloads = tuple(str(key) for key in (supported_workload_keys or WORKFLOW_KEYS))
    supported_workload_options = (DEFAULT_WORKLOAD_KEY, *allowed_workloads)
    prompt_contract = _build_prompt_contract(
        supported_workload_keys=allowed_workloads,
        default_artifact_root=default_artifact_root,
    )
    workload_selection, workload_defaulted = _normalize_workload_selection(
        value=selection,
        supported_values=supported_workload_options,
        default_value=str(prompt_contract["workload selection"]["recommended"]),
    )
    selected_mode, mode_defaulted = _normalize_choice_with_default(
        value=mode,
        field_name="MODE",
        supported_values=list(MODE_KEYS),
        default_value=_prompt_contract_value(prompt_contract, "mode", "recommended"),
    )
    resolved_artifact_root, artifact_root_overridden, artifact_root_defaulted = _normalize_artifact_root(
        artifact_root,
        default_artifact_root=default_artifact_root,
    )
    resolved_gpu_number, gpu_number_defaulted = _normalize_gpu_number(
        value=gpu_number,
        default_value=_prompt_contract_value(prompt_contract, "gpu number", "recommended"),
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
        "artifact_root": resolved_artifact_root,
        "artifact_root_overridden": artifact_root_overridden,
        "artifact_root_defaulted": artifact_root_defaulted,
        "gpu_number": resolved_gpu_number,
        "gpu_number_defaulted": gpu_number_defaulted,
        "selected_gpu_number": resolved_gpu_number,
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
    artifact_root: object,
    gpu_number: object,
    confirm: object,
    supported_workload_keys: Sequence[str] | None = None,
    default_artifact_root: str = CANONICAL_PARENT_ARTIFACT_ROOT,
) -> dict[str, object]:
    preview_payload = build_workflow_request_preview(
        selection=selection,
        mode=mode,
        artifact_root=artifact_root,
        gpu_number=gpu_number,
        supported_workload_keys=supported_workload_keys,
        default_artifact_root=default_artifact_root,
    )
    confirm_choice = _normalize_required_choice(
        value=confirm,
        field_name="CONFIRMATION",
        supported_values=["y", "n"],
    )
    confirmation_defaulted = False

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
    default_artifact_root: str = CANONICAL_PARENT_ARTIFACT_ROOT,
) -> dict[str, object]:
    allowed_workloads = tuple(str(key) for key in (supported_workload_keys or WORKFLOW_KEYS))
    prompt_contract = _build_prompt_contract(
        supported_workload_keys=allowed_workloads,
        default_artifact_root=default_artifact_root,
    )
    if output_fn is not None:
        for prompt_name in PROMPT_SEQUENCE:
            _ = output_fn(str(prompt_contract[prompt_name]["display"]))

    selection = input_fn(str(prompt_contract["workload selection"]["prompt"]))
    mode = input_fn(str(prompt_contract["mode"]["prompt"]))
    artifact_root = input_fn(str(prompt_contract["artifact root override"]["prompt"]))
    gpu_number = input_fn(str(prompt_contract["gpu number"]["prompt"]))

    preview_payload = build_workflow_request_preview(
        selection=selection,
        mode=mode,
        artifact_root=artifact_root,
        gpu_number=gpu_number,
        supported_workload_keys=allowed_workloads,
        default_artifact_root=default_artifact_root,
    )
    if preview_callback is not None:
        _ = preview_callback(preview_payload)

    confirm = input_fn(str(prompt_contract["confirmation"]["prompt"]))
    return validate_workflow_request(
        selection=selection,
        mode=mode,
        artifact_root=artifact_root,
        gpu_number=gpu_number,
        confirm=confirm,
        supported_workload_keys=allowed_workloads,
        default_artifact_root=default_artifact_root,
    )


__all__ = [
    "WORKFLOW_KEYS",
    "CANONICAL_PARENT_ARTIFACT_ROOT",
    "CHECKPOINT_MAP",
    "SUPPORTED_GPU_NUMBERS",
    "prompt_for_workflow_request",
]