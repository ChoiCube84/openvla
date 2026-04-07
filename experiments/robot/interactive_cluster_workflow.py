from __future__ import annotations

import json
import os
import shutil
import shlex
import subprocess
import sys
import time
import importlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.robot.interactive_workflow_contract import (
    CANONICAL_PARENT_ARTIFACT_ROOT,
    CHECKPOINT_MAP,
    SUPPORTED_GPU_NUMBERS,
    prompt_for_workflow_request,
)
from experiments.robot.maniskill.defaults import DEFAULT_GPU_INDEX_ENV_KEY
from experiments.robot.workflow_logging import (
    WORKFLOW_BENCHMARK_ENV_KEY,
    WORKFLOW_CONTROLLER_PYTHON_ENV_KEY,
    WORKFLOW_LAUNCH_PATH_ENV_KEY,
    WORKFLOW_MODEL_FAMILY_ENV_KEY,
    WORKFLOW_SELECTED_CUDA_VISIBLE_DEVICES_ENV_KEY,
    WORKFLOW_SELECTED_GPU_ENV_KEY,
    WORKFLOW_WORKLOAD_KEY_ENV_KEY,
    append_breadcrumb_block,
    failure_metadata_from_exception,
    iso_timestamp,
)

PARENT_SUMMARY_FILENAME = "workflow_summary.json"
RUNTIME_PLAN_FILENAME = "runtime_plan.json"

OPENPI_BOOTSTRAP_STDOUT_KEYS = (
    "openpi_runtime_status",
    "openpi_error_state",
    "openpi_error_message",
    "openpi_cache_state",
    "openpi_bootstrap_action",
    "openpi_repo_root",
    "openpi_managed_repo_root",
    "openpi_bootstrap_marker",
    "openpi_bootstrap_source_url",
    "openpi_bootstrap_ref",
    "openpi_bootstrap_git_revision",
    "openpi_policy_server_status",
    "openpi_policy_server_url",
    "openpi_policy_server_python",
    "openpi_policy_server_entrypoint",
    "openpi_policy_server_launch_prefix",
    "openpi_checkpoint",
    "openpi_conda_env",
    "openpi_bootstrap_python",
)
MANISKILL_WORKLOADS = {"openvla_maniskill_ft", "openpi_maniskill"}
LIBERO_WORKLOADS = {"openvla_libero", "openvla_libero_ft", "openpi_libero"}
MANAGED_OPENPI_POLICY_SERVER_WORKLOADS = {"openpi_maniskill", "openpi_libero"}
DEFAULT_OPENPI_POLICY_CONFIG = "pi05_libero"
POLICY_SERVER_HEALTH_TIMEOUT_SECONDS = 45.0
POLICY_SERVER_HEALTH_POLL_SECONDS = 1.0
GPU_HEAVY_PHASE_NAME = "gpu_eval"


class ManagedOpenPILifecycleError(RuntimeError):
    def __init__(self, message: str, *, failure_phase: str, payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.failure_phase = failure_phase
        self.payload = payload


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


def _parse_key_values(text: str, keys: tuple[str, ...]) -> dict[str, str]:
    parsed = {key: "" for key in keys}
    for line in text.splitlines():
        for key in keys:
            prefix = f"{key}="
            if line.startswith(prefix):
                parsed[key] = line.split("=", 1)[1].strip()
    return parsed


def _tail_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return "no_output"


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _as_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _build_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"WORKFLOW-{timestamp}-pid{os.getpid()}"


def _controller_input_required_message() -> str:
    return (
        "INTERACTIVE_INPUT_REQUIRED: `python3 experiments/robot/interactive_cluster_workflow.py` "
        "requires operator input on stdin. Launch the controller directly and provide workload, mode, "
        "artifact root, GPU number, and confirmation responses."
    )


def _controller_prompt_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError as exc:
        raise ValueError(_controller_input_required_message()) from exc

def _describe_controller_interpreter(resolution: dict[str, Any]) -> str:
    status = str(resolution.get("status") or "active")
    source = str(resolution.get("selection_source") or "direct_python")
    selected_python = str(resolution.get("selected_python") or sys.executable)
    selected_conda_env = str(resolution.get("selected_conda_env") or "none")
    torch_version = str(resolution.get("torch_version") or "unknown")
    execution_mode = str(resolution.get("execution_mode") or "current_process")
    return (
        "CONTROLLER_INTERPRETER: "
        f"status={status}; source={source}; python={selected_python}; "
        f"conda_env={selected_conda_env}; torch_version={torch_version}; execution_mode={execution_mode}."
    )


def _current_controller_interpreter_resolution() -> dict[str, Any]:
    torch_version = None
    try:
        torch = __import__("torch")
        torch_version = str(getattr(torch, "__version__", "unknown"))
    except Exception:
        torch_version = None
    return {
        "status": "active",
        "selection_source": "direct_python",
        "selected_python": sys.executable,
        "selected_conda_env": os.environ.get("CONDA_DEFAULT_ENV", "").strip() or None,
        "torch_version": torch_version,
        "execution_mode": "current_process",
    }


def _build_parent_paths(session_id: str, artifact_root: str | None = None) -> dict[str, Path]:
    root = artifact_root if artifact_root else CANONICAL_PARENT_ARTIFACT_ROOT
    session_dir = Path(root) / session_id
    return {
        "session_dir": session_dir,
        "summary_path": session_dir / PARENT_SUMMARY_FILENAME,
        "runtime_plan_path": session_dir / RUNTIME_PLAN_FILENAME,
        "controller_log_path": session_dir / "controller.log",
    }


def _json_dumps_compact(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=_json_default)


def _normalize_manual_gpu_number(value: object) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    try:
        canonical = str(int(normalized, 10))
    except ValueError:
        return None
    if canonical not in SUPPORTED_GPU_NUMBERS:
        return None
    return canonical


def _query_nvidia_smi_gpu_rows() -> tuple[list[dict[str, int]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return [], "nvidia-smi unavailable on PATH"

    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "nvidia-smi failed").strip()
        return [], detail or "nvidia-smi failed"

    rows: list[dict[str, int]] = []
    for line in process.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "memory_used_mb": int(parts[1]),
                    "utilization_gpu_percent": int(parts[2]),
                }
            )
        except ValueError:
            continue
    if not rows:
        return [], "nvidia-smi returned no parseable GPU rows"
    rows.sort(key=lambda row: (row["memory_used_mb"], row["utilization_gpu_percent"], row["index"]))
    return rows, None


def _compute_gpu_prompt_context() -> dict[str, Any]:
    visible_devices_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_devices: list[str] = []
    if visible_devices_raw is not None:
        visible_devices = [entry.strip() for entry in visible_devices_raw.split(",") if entry.strip()]

    context: dict[str, Any] = {
        "status": "ready",
        "execution_model": "serial_controller_dispatch",
        "supported_gpu_numbers": list(SUPPORTED_GPU_NUMBERS),
        "reason": "Manual GPU numbers 0..7 are trusted without controller-visible GPU gating.",
        "blocker": None,
        "host_observation": {
            "cuda_visible_devices_env": visible_devices_raw,
            "visible_devices_entries": visible_devices,
            "required_gpu_index_env_key": DEFAULT_GPU_INDEX_ENV_KEY,
        },
    }

    try:
        torch = __import__("torch")
        context["host_observation"]["torch_cuda_available"] = bool(torch.cuda.is_available())
        context["host_observation"]["torch_visible_device_count"] = (
            int(torch.cuda.device_count()) if bool(torch.cuda.is_available()) else 0
        )
    except Exception as exc:
        context["host_observation"]["torch_cuda_available"] = None
        context["host_observation"]["torch_visible_device_count"] = None
        context["host_observation"]["torch_discovery_error"] = str(exc)

    smi_rows, smi_error = _query_nvidia_smi_gpu_rows()
    context["host_observation"]["nvidia_smi_rows"] = smi_rows
    context["host_observation"]["nvidia_smi_error"] = smi_error
    return context


def _resolve_gpu_assignment(
    workflow_request: dict[str, Any],
    *,
    gpu_prompt_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_context = dict(gpu_prompt_context or _compute_gpu_prompt_context())
    requested_gpu_number = _normalize_manual_gpu_number(
        workflow_request.get("selected_gpu_number", workflow_request.get("gpu_number"))
    )
    assignment: dict[str, Any] = {
        "status": str(prompt_context.get("status") or "blocked"),
        "execution_model": str(prompt_context.get("execution_model") or "serial_controller_dispatch"),
        "supported_gpu_numbers": _as_string_list(prompt_context.get("supported_gpu_numbers")),
        "requested_gpu_number": requested_gpu_number,
        "selected_gpu": None,
        "selected_cuda_visible_devices": None,
        "selection_source": None,
        "reason": str(prompt_context.get("reason") or "GPU preflight not evaluated."),
        "blocker": prompt_context.get("blocker"),
        "host_observation": _as_dict(prompt_context.get("host_observation")),
        "selection_trace": [],
    }

    if assignment["status"] != "ready":
        return assignment

    if requested_gpu_number is None:
        assignment["status"] = "blocked"
        supported = ", ".join(SUPPORTED_GPU_NUMBERS)
        raw_value = workflow_request.get("selected_gpu_number", workflow_request.get("gpu_number"))
        assignment["blocker"] = f"INVALID_GPU_NUMBER: {str(raw_value).strip() or '<blank>'}. Supported values: {supported}"
        assignment["reason"] = "The runtime plan accepts only manual GPU numbers 0..7."
        return assignment

    selected_cuda_visible_devices = requested_gpu_number

    gpu_index = int(requested_gpu_number)
    try:
        import torch
        if torch.cuda.is_available():
            visible_count = torch.cuda.device_count()
            if gpu_index >= visible_count:
                assignment["status"] = "blocked"
                assignment["blocker"] = (
                    f"INVALID_GPU_NUMBER: GPU {gpu_index} requested but only "
                    f"{visible_count} GPU(s) visible to this process."
                )
                assignment["reason"] = "Requested GPU index exceeds visible device count."
                return assignment
    except Exception:
        pass  # torch 없거나 CUDA 없는 환경은 cluster-only 실행으로 간주하고 통과

    assignment["selection_trace"] = ["using explicit workflow_request.selected_gpu_number without controller-visible gating"]
    assignment["selected_gpu"] = requested_gpu_number
    assignment["selected_cuda_visible_devices"] = selected_cuda_visible_devices
    assignment["selection_source"] = "workflow_request.selected_gpu_number"
    assignment["reason"] = "Explicit manual GPU number selection was accepted unchanged for all GPU-heavy phases."
    return assignment


def _load_maniskill_runtime_estimate(mode: str) -> dict[str, Any]:
    try:
        estimate_runtime = importlib.import_module("experiments.robot.maniskill.estimate_runtime")
        probe = estimate_runtime._probe_env()
        estimate_payload = estimate_runtime._estimate(probe)
        mode_payload = _as_dict(_as_dict(estimate_payload.get("mode_estimates")).get(mode))
        return {
            "status": "ready",
            "source": "experiments.robot.maniskill.estimate_runtime",
            "mode": mode,
            "estimate_payload": estimate_payload,
            "selected_mode_estimate": mode_payload,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "source": "experiments.robot.maniskill.estimate_runtime",
            "mode": mode,
            "error": str(exc),
            "estimate_payload": {},
            "selected_mode_estimate": {},
        }


def _estimated_work_payload(workload_key: str, runtime_estimate: dict[str, Any]) -> dict[str, Any]:
    if workload_key in MANISKILL_WORKLOADS and runtime_estimate.get("status") == "ready":
        selected_mode_estimate = _as_dict(runtime_estimate.get("selected_mode_estimate"))
        return {
            "status": "ready",
            "seconds": selected_mode_estimate.get("estimated_total_seconds"),
            "storage_bytes": selected_mode_estimate.get("estimated_storage_bytes"),
            "source": runtime_estimate.get("source"),
            "mode": runtime_estimate.get("mode"),
        }
    if workload_key in MANISKILL_WORKLOADS:
        return {
            "status": "unavailable",
            "seconds": None,
            "storage_bytes": None,
            "source": runtime_estimate.get("source"),
            "mode": runtime_estimate.get("mode"),
            "reason": runtime_estimate.get("error") or "ManiSkill runtime estimate unavailable.",
        }
    return {
        "status": "not_supported",
        "seconds": None,
        "storage_bytes": None,
        "source": None,
        "mode": None,
        "reason": "No LIBERO runtime estimator exists in Task 8 scope, so estimated work is left explicit-null.",
    }


def _integrated_workflow_blockers(workflow_request: dict[str, Any]) -> list[dict[str, Any]]:
    blockers = []
    resolved_workloads = _as_string_list(workflow_request.get("resolved_workloads"))

    # OpenPI 워크로드 포함 시 conda 및 환경 확인
    openpi_workloads = [w for w in resolved_workloads if w in MANAGED_OPENPI_POLICY_SERVER_WORKLOADS]
    if openpi_workloads:
        if not shutil.which("conda"):
            blockers.append({
                "workload_key": ",".join(openpi_workloads),
                "blocker": "CONDA_NOT_FOUND: `conda` executable not on PATH; required for OpenPI workloads.",
            })
        openpi_repo_root = _resolve_openpi_repo_root()
        if not openpi_repo_root and not _managed_repo_root_exists():
            blockers.append({
                "workload_key": ",".join(openpi_workloads),
                "blocker": "OPENPI_REPO_NOT_FOUND: no OPENPI_REPO_ROOT set and no managed cache exists yet.",
            })

    return blockers

def _managed_repo_root_exists() -> bool:
    try:
        from experiments.robot.maniskill.bootstrap_openpi import _managed_repo_root
        return _managed_repo_root().exists()
    except Exception:
        return False

def _build_runtime_plan(
    workflow_request: dict[str, Any],
    session_id: str,
    parent_paths: dict[str, Path],
    *,
    gpu_prompt_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_workloads = _as_string_list(workflow_request.get("resolved_workloads"))
    controller_interpreter = _current_controller_interpreter_resolution()
    gpu_assignment = _resolve_gpu_assignment(workflow_request, gpu_prompt_context=gpu_prompt_context)
    runtime_estimate = _load_maniskill_runtime_estimate(str(workflow_request.get("selected_mode", "")))
    integrated_workflow_blockers = _integrated_workflow_blockers(workflow_request)
    scheduling_reason = (
        "GPU-heavy evaluation phases run serially under the direct Python controller. "
        "The controller does not offer operator-facing parallel scheduling controls and does not claim concurrent GPU evaluation."
    )

    plan_items: list[dict[str, Any]] = []
    for order, workload_key in enumerate(resolved_workloads, start=1):
        workload_detail = _as_dict(_as_dict(workflow_request.get("workload_details")).get(workload_key))
        estimated_work = _estimated_work_payload(workload_key, runtime_estimate)
        artifact_root = _resolve_child_artifact_root(workflow_request, workload_key)
        plan_items.append(
            {
                "order": order,
                "workload_key": workload_key,
                "runner": str(workload_detail.get("runner", "")),
                "benchmark": str(workload_detail.get("benchmark", "")),
                "model_family": str(workload_detail.get("model_family", "")),
                "artifact_root": artifact_root,
                "selected_gpu": gpu_assignment.get("selected_gpu"),
                "selected_cuda_visible_devices": gpu_assignment.get("selected_cuda_visible_devices"),
                "estimated_work": estimated_work,
                "workload_blockers": [
                    blocker for blocker in integrated_workflow_blockers if blocker.get("workload_key") == workload_key
                ],
                "phases": {
                    "bootstrap_cpu_overlap_allowed": False,
                    "gpu_heavy_execution": "serialized_single_gpu",
                    "gpu_heavy_phase_name": GPU_HEAVY_PHASE_NAME,
                    "logging_collection": "controller_managed",
                },
                "scheduler_notes": ["operator-facing execution remains serial", "gpu_heavy_phase_serialized_on_single_gpu"],
            }
        )

    plan_payload = {
        "session_id": session_id,
        "status": "ready" if gpu_assignment.get("status") == "ready" and not integrated_workflow_blockers else "blocked",
        "gpu_heavy_execution": "serialized_single_gpu",
        "scheduling_reason": scheduling_reason,
        "gpu_assignment": gpu_assignment,
        "controller_interpreter": controller_interpreter,
        "workflow_blockers": integrated_workflow_blockers,
        "runtime_estimate": runtime_estimate,
        "planned_workloads": plan_items,
        "artifact_paths": {
            "runtime_plan_path": str(parent_paths["runtime_plan_path"]),
            "workflow_summary_path": str(parent_paths["summary_path"]),
            "session_dir": str(parent_paths["session_dir"]),
        },
    }
    _write_json(parent_paths["runtime_plan_path"], plan_payload)
    return plan_payload


def _scheduler_execution_policy(runtime_plan: dict[str, Any]) -> dict[str, Any]:
    gpu_assignment = _as_dict(runtime_plan.get("gpu_assignment"))
    return {
        "execution_model": "serial_controller_dispatch",
        "actual_execution": "serialized_single_gpu_gpu_heavy_phases",
        "gpu_heavy_execution": str(runtime_plan.get("gpu_heavy_execution", "serialized_single_gpu")),
        "reason": str(runtime_plan.get("scheduling_reason", "")),
        "selected_gpu": gpu_assignment.get("selected_gpu"),
        "selected_cuda_visible_devices": gpu_assignment.get("selected_cuda_visible_devices"),
        "preflight_status": gpu_assignment.get("status"),
        "preflight_blocker": gpu_assignment.get("blocker"),
    }


def _build_child_env(
    runtime_plan: dict[str, Any],
    *,
    managed_openpi_runtime: dict[str, str] | None = None,
    openpi_checkpoint: str | None = None,
) -> dict[str, str]:
    child_env = dict(os.environ)
    gpu_assignment = _as_dict(runtime_plan.get("gpu_assignment"))
    compatibility_gpu_index = str(gpu_assignment.get("selected_gpu") or "").strip()
    selected_cuda_visible_devices = str(gpu_assignment.get("selected_cuda_visible_devices") or "").strip()
    if compatibility_gpu_index:
        child_env[WORKFLOW_SELECTED_GPU_ENV_KEY] = compatibility_gpu_index
    if selected_cuda_visible_devices:
        child_env["CUDA_VISIBLE_DEVICES"] = selected_cuda_visible_devices
        child_env[WORKFLOW_SELECTED_CUDA_VISIBLE_DEVICES_ENV_KEY] = selected_cuda_visible_devices
    if compatibility_gpu_index:
        child_env[DEFAULT_GPU_INDEX_ENV_KEY] = compatibility_gpu_index
    if managed_openpi_runtime is not None:
        policy_server_url = str(managed_openpi_runtime.get("policy_server_url", "")).strip()
        openpi_conda_env = str(managed_openpi_runtime.get("openpi_conda_env", "")).strip()
        openpi_repo_root = str(managed_openpi_runtime.get("openpi_repo_root", "")).strip()
        if policy_server_url:
            child_env["OPENPI_POLICY_SERVER_URL"] = policy_server_url
            child_env["OPENVLA_MANISKILL_PI0_POLICY_SERVER_URL"] = policy_server_url
        if openpi_conda_env:
            child_env["OPENPI_CONDA_ENV"] = openpi_conda_env
            child_env["OPENVLA_MANISKILL_OPENPI_CONDA_ENV"] = openpi_conda_env
        if openpi_repo_root:
            child_env["OPENPI_REPO_ROOT"] = openpi_repo_root
            child_env["OPENVLA_MANISKILL_OPENPI_REPO_ROOT"] = openpi_repo_root
    if openpi_checkpoint:
        child_env["OPENPI_CHECKPOINT"] = openpi_checkpoint
    return child_env


def _controller_launch_metadata(
    *,
    workload_key: str,
    workflow_request: dict[str, Any],
    runtime_plan: dict[str, Any],
    launch_path: str,
    effective_workload_python: str | None = None,
) -> dict[str, Any]:
    workload_details = _as_dict(_as_dict(workflow_request.get("workload_details")).get(workload_key))
    gpu_assignment = _as_dict(runtime_plan.get("gpu_assignment"))
    controller_interpreter = _current_controller_interpreter_resolution()
    return {
        "timestamp": iso_timestamp(),
        "launch_path": launch_path,
        "workload_key": workload_key,
        "benchmark": str(workload_details.get("benchmark", "")),
        "model_family": str(workload_details.get("model_family", "")),
        "controller_python": str(controller_interpreter.get("selected_python") or sys.executable),
        "effective_workload_python": effective_workload_python,
        "selected_gpu": gpu_assignment.get("selected_gpu"),
        "selected_cuda_visible_devices": gpu_assignment.get("selected_cuda_visible_devices"),
    }

def _write_controller_event(controller_log_path: Path, payload: dict[str, Any], *, heading: str) -> None:
    append_breadcrumb_block(controller_log_path, payload, heading=heading)


@contextmanager
def _temporary_env(updates: dict[str, str]) -> Any:
    previous: dict[str, str | None] = {}
    try:
        for key, value in updates.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _apply_workload_child_env(child_env: dict[str, str], launch_metadata: dict[str, Any], *, workload_key: str) -> dict[str, str]:
    child_env[WORKFLOW_LAUNCH_PATH_ENV_KEY] = str(launch_metadata.get("launch_path") or "controller_subprocess")
    child_env[WORKFLOW_WORKLOAD_KEY_ENV_KEY] = str(launch_metadata.get("workload_key") or workload_key)
    child_env[WORKFLOW_BENCHMARK_ENV_KEY] = str(launch_metadata.get("benchmark") or "")
    child_env[WORKFLOW_MODEL_FAMILY_ENV_KEY] = str(launch_metadata.get("model_family") or "")
    child_env[WORKFLOW_CONTROLLER_PYTHON_ENV_KEY] = str(launch_metadata.get("controller_python") or sys.executable)
    child_env[WORKFLOW_SELECTED_GPU_ENV_KEY] = str(launch_metadata.get("selected_gpu") or "")
    child_env[WORKFLOW_SELECTED_CUDA_VISIBLE_DEVICES_ENV_KEY] = str(
        launch_metadata.get("selected_cuda_visible_devices") or ""
    )
    return child_env


def _build_execution_record(
    *,
    workload_key: str,
    command: list[str],
    log_path: Path,
    started_at: float,
    finished_at: float,
    status: str,
    exit_code: int,
    parsed_stdout: dict[str, Any],
    failure_details: dict[str, Any],
    policy_server: dict[str, Any] | None,
    launch_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": iso_timestamp(),
        "launch_path": launch_metadata.get("launch_path"),
        "workload_key": workload_key,
        "benchmark": launch_metadata.get("benchmark"),
        "model_family": launch_metadata.get("model_family"),
        "controller_python": launch_metadata.get("controller_python"),
        "effective_workload_python": parsed_stdout.get("effective_workload_python") or launch_metadata.get("effective_workload_python"),
        "selected_gpu": parsed_stdout.get("selected_gpu") or launch_metadata.get("selected_gpu"),
        "selected_cuda_visible_devices": parsed_stdout.get("selected_cuda_visible_devices")
        or launch_metadata.get("selected_cuda_visible_devices"),
        "failure_phase": failure_details.get("failure_phase"),
        "failure_location": failure_details.get("failure_location"),
        "exception_type": failure_details.get("exception_type"),
        "exception_message": failure_details.get("exception_message"),
        "traceback_tail": failure_details.get("traceback_tail"),
        "subprocess_detail": failure_details.get("subprocess_detail"),
        "command": command,
        "cwd": str(REPO_ROOT),
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "duration_seconds": finished_at - started_at,
        "status": status,
        "exit_code": exit_code,
        "parsed_stdout": parsed_stdout,
        "log_path": str(log_path),
        "policy_server": policy_server,
    }


def _controller_child_result(
    *,
    workload_key: str,
    status: str,
    exit_code: int,
    command: list[str],
    log_path: Path,
    execution_record_path: Path,
    parsed_stdout: dict[str, Any],
    failure_details: dict[str, Any],
    policy_server: dict[str, Any] | None,
    launch_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "workload_key": workload_key,
        "status": status,
        "exit_code": exit_code,
        "blocker_reason": failure_details.get("exception_message"),
        "failure_phase": failure_details.get("failure_phase"),
        "failure_location": failure_details.get("failure_location"),
        "exception_type": failure_details.get("exception_type"),
        "exception_message": failure_details.get("exception_message"),
        "traceback_tail": failure_details.get("traceback_tail"),
        "launch_path": launch_metadata.get("launch_path"),
        "controller_python": launch_metadata.get("controller_python"),
        "effective_workload_python": parsed_stdout.get("effective_workload_python") or launch_metadata.get("effective_workload_python"),
        "selected_gpu": parsed_stdout.get("selected_gpu") or launch_metadata.get("selected_gpu"),
        "selected_cuda_visible_devices": parsed_stdout.get("selected_cuda_visible_devices")
        or launch_metadata.get("selected_cuda_visible_devices"),
        "command": command,
        "summary_path": parsed_stdout.get("summary_path") or None,
        "manifest_path": parsed_stdout.get("manifest_path") or None,
        "episodes_path": parsed_stdout.get("episodes_path") or None,
        "run_dir": parsed_stdout.get("run_dir") or None,
        "checkpoint": None,
        "average_success_rate": None,
        "per_task_success_rate": {},
        "artifact_paths": {},
        "log_path": str(log_path),
        "parsed_stdout": parsed_stdout,
        "execution_record_path": str(execution_record_path),
        "policy_server": policy_server,
    }


def _resolve_child_artifact_root(workflow_request: dict[str, Any], workload_key: str) -> str:
    if bool(workflow_request.get("artifact_root_overridden", False)):
        return str(workflow_request.get("artifact_root", "")).strip()
    workload_details = workflow_request.get("workload_details", {})
    detail = workload_details.get(workload_key, {}) if isinstance(workload_details, dict) else {}
    return str(detail.get("artifact_root", "")).strip()


def _resolve_openpi_policy_server_url() -> str:
    return (
        os.environ.get("OPENVLA_MANISKILL_PI0_POLICY_SERVER_URL")
        or os.environ.get("OPENPI_POLICY_SERVER_URL")
        or "http://127.0.0.1:8000"
    ).strip()


def _resolve_openpi_conda_env() -> str:
    return (
        os.environ.get("OPENVLA_MANISKILL_OPENPI_CONDA_ENV")
        or os.environ.get("OPENPI_CONDA_ENV")
        or "openpi"
    ).strip()


def _resolve_openpi_repo_root() -> str:
    return (
        os.environ.get("OPENVLA_MANISKILL_OPENPI_REPO_ROOT")
        or os.environ.get("OPENPI_REPO_ROOT")
        or ""
    ).strip()


def _resolve_openpi_policy_config(checkpoint: str) -> str:
    candidate = checkpoint.rstrip("/").rsplit("/", 1)[-1].strip()
    return candidate or DEFAULT_OPENPI_POLICY_CONFIG


def _resolve_openpi_bootstrap_python() -> str:
    controller_interpreter = _current_controller_interpreter_resolution()
    return str(controller_interpreter.get("selected_python") or sys.executable)


def _openpi_lifecycle_metadata(
    *,
    workload_key: str,
    checkpoint: str,
    phase: str,
    policy_server_url: str,
    bootstrap_command: list[str] | None = None,
    bootstrap_metadata: dict[str, str] | None = None,
    policy_server_command: list[str] | None = None,
    error_message: str | None = None,
    teardown_reason: str | None = None,
    teardown_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    controller_interpreter = _current_controller_interpreter_resolution()
    bootstrap_metadata = bootstrap_metadata or {}
    payload: dict[str, Any] = {
        "timestamp": iso_timestamp(),
        "launch_path": "controller_managed_openpi",
        "workload_key": workload_key,
        "controller_python": str(controller_interpreter.get("selected_python") or sys.executable),
        "effective_workload_python": str(bootstrap_metadata.get("openpi_policy_server_python") or ""),
        "openpi_phase": phase,
        "openpi_checkpoint": checkpoint,
        "openpi_policy_server_url": policy_server_url,
        "openpi_conda_env": str(bootstrap_metadata.get("openpi_conda_env") or _resolve_openpi_conda_env()),
        "openpi_repo_root": str(bootstrap_metadata.get("openpi_repo_root") or _resolve_openpi_repo_root()),
        "openpi_bootstrap_python": str(
            bootstrap_metadata.get("openpi_bootstrap_python")
            or (bootstrap_command[0] if bootstrap_command else _resolve_openpi_bootstrap_python())
        ),
    }
    if bootstrap_command:
        payload["openpi_bootstrap_command"] = shlex.join(bootstrap_command)
    if policy_server_command:
        payload["openpi_policy_server_command"] = shlex.join(policy_server_command)
    if bootstrap_metadata.get("openpi_policy_server_entrypoint"):
        payload["openpi_policy_server_entrypoint"] = bootstrap_metadata["openpi_policy_server_entrypoint"]
    if bootstrap_metadata.get("openpi_policy_server_launch_prefix"):
        payload["openpi_policy_server_launch_prefix"] = bootstrap_metadata["openpi_policy_server_launch_prefix"]
    if error_message:
        payload["exception_message"] = error_message
    if teardown_reason:
        payload["openpi_stop_reason"] = teardown_reason
    if teardown_result is not None:
        payload["openpi_teardown_result"] = json.dumps(teardown_result, sort_keys=True, default=_json_default)
    return payload


def _append_openpi_lifecycle_event(
    *,
    log_path: Path,
    lifecycle_events: list[dict[str, Any]],
    workload_key: str,
    checkpoint: str,
    phase: str,
    policy_server_url: str,
    bootstrap_command: list[str] | None = None,
    bootstrap_metadata: dict[str, str] | None = None,
    policy_server_command: list[str] | None = None,
    error_message: str | None = None,
    teardown_reason: str | None = None,
    teardown_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _openpi_lifecycle_metadata(
        workload_key=workload_key,
        checkpoint=checkpoint,
        phase=phase,
        policy_server_url=policy_server_url,
        bootstrap_command=bootstrap_command,
        bootstrap_metadata=bootstrap_metadata,
        policy_server_command=policy_server_command,
        error_message=error_message,
        teardown_reason=teardown_reason,
        teardown_result=teardown_result,
    )
    append_breadcrumb_block(log_path, payload, heading=phase)
    lifecycle_events.append(payload)
    return payload


def _openpi_teardown_result(*, process_started: bool, result: dict[str, Any] | None = None) -> dict[str, Any]:
    if result is None:
        return {
            "process_started": process_started,
            "terminated": False,
            "kill_used": False,
            "returncode": None,
        }
    return {"process_started": process_started, **result}


def _build_openpi_bootstrap_command(*, checkpoint: str, policy_server_url: str, require_policy_server_health: bool) -> list[str]:
    command = [
        _resolve_openpi_bootstrap_python(),
        str((REPO_ROOT / "experiments/robot/maniskill/bootstrap_openpi.py").resolve()),
        "--checkpoint",
        checkpoint,
        "--policy-server-url",
        policy_server_url,
    ]
    openpi_conda_env = _resolve_openpi_conda_env()
    if openpi_conda_env:
        command.extend(["--openpi-conda-env", openpi_conda_env])
    openpi_repo_root = _resolve_openpi_repo_root()
    if openpi_repo_root:
        command.extend(["--openpi-repo-root", openpi_repo_root])
    if require_policy_server_health:
        command.append("--require-policy-server-health")
    return command


def _run_openpi_bootstrap_command(
    *,
    checkpoint: str,
    policy_server_url: str,
    require_policy_server_health: bool,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> tuple[int, str, dict[str, str]]:
    command = _build_openpi_bootstrap_command(
        checkpoint=checkpoint,
        policy_server_url=policy_server_url,
        require_policy_server_health=require_policy_server_health,
    )
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output_text = process.stdout or ""
    log_path.write_text(output_text)
    return process.returncode, output_text, _parse_key_values(output_text, OPENPI_BOOTSTRAP_STDOUT_KEYS)


def _build_openpi_policy_server_command(metadata: dict[str, str]) -> list[str]:
    conda_executable = shutil.which("conda")
    if not conda_executable:
        raise RuntimeError("OPENPI_POLICY_SERVER_STARTUP_FAILED: `conda` executable not found for managed OpenPI env startup.")

    entrypoint = metadata.get("openpi_policy_server_entrypoint", "").strip()
    if not entrypoint:
        raise RuntimeError("OPENPI_POLICY_SERVER_STARTUP_FAILED: bootstrap metadata did not include a policy server entrypoint.")

    python_executable = metadata.get("openpi_policy_server_python", "python3").strip() or "python3"
    checkpoint = metadata.get("openpi_checkpoint", "").strip()
    policy_config = _resolve_openpi_policy_config(checkpoint)
    return [
        conda_executable,
        "run",
        "--no-capture-output",
        "-n",
        metadata.get("openpi_conda_env", "openpi").strip() or "openpi",
        python_executable,
        entrypoint,
        "policy:checkpoint",
        f"--policy.config={policy_config}",
        f"--policy.dir={checkpoint}",
    ]


def _terminate_process(process: subprocess.Popen[Any], *, grace_seconds: float = 10.0) -> dict[str, Any]:
    already_exited = process.poll()
    if already_exited is not None:
        return {"terminated": False, "kill_used": False, "returncode": already_exited}

    process.terminate()
    kill_used = False
    try:
        returncode = process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        kill_used = True
        process.kill()
        returncode = process.wait(timeout=grace_seconds)
    return {"terminated": True, "kill_used": kill_used, "returncode": returncode}


def _wait_for_openpi_policy_server_health(
    *,
    checkpoint: str,
    policy_server_url: str,
    process: subprocess.Popen[str],
    log_path: Path,
    env: dict[str, str] | None,
    timeout_seconds: float = POLICY_SERVER_HEALTH_TIMEOUT_SECONDS,
) -> tuple[bool, str, dict[str, str]]:
    deadline = time.time() + timeout_seconds
    last_output = ""
    last_parsed: dict[str, str] = {key: "" for key in OPENPI_BOOTSTRAP_STDOUT_KEYS}
    while time.time() < deadline:
        if process.poll() is not None:
            message = log_path.read_text() if log_path.exists() else ""
            return False, message, last_parsed
        returncode, output_text, parsed = _run_openpi_bootstrap_command(
            checkpoint=checkpoint,
            policy_server_url=policy_server_url,
            require_policy_server_health=True,
            log_path=log_path,
            env=env,
        )
        last_output = output_text
        last_parsed = parsed
        if returncode == 0 and parsed.get("openpi_policy_server_status") == "healthy":
            return True, output_text, parsed
        time.sleep(POLICY_SERVER_HEALTH_POLL_SECONDS)
    return False, last_output, last_parsed


def _start_managed_openpi_policy_server(
    *,
    workload_key: str,
    checkpoint: str,
    workload_dir: Path,
    log_path: Path,
    env: dict[str, str] | None,
) -> dict[str, Any]:
    bootstrap_log_path = workload_dir / "policy_server_bootstrap.log"
    health_log_path = workload_dir / "policy_server_health.log"
    server_log_path = workload_dir / "policy_server.log"
    policy_server_url = _resolve_openpi_policy_server_url()
    lifecycle_events: list[dict[str, Any]] = []
    bootstrap_command = _build_openpi_bootstrap_command(
        checkpoint=checkpoint,
        policy_server_url=policy_server_url,
        require_policy_server_health=False,
    )
    _append_openpi_lifecycle_event(
        log_path=log_path,
        lifecycle_events=lifecycle_events,
        workload_key=workload_key,
        checkpoint=checkpoint,
        phase="openpi_start",
        policy_server_url=policy_server_url,
        bootstrap_command=bootstrap_command,
    )
    bootstrap_exit_code, bootstrap_output, bootstrap_metadata = _run_openpi_bootstrap_command(
        checkpoint=checkpoint,
        policy_server_url=policy_server_url,
        require_policy_server_health=False,
        log_path=bootstrap_log_path,
        env=env,
    )
    if bootstrap_exit_code != 0 or bootstrap_metadata.get("openpi_runtime_status") != "ready":
        message = _tail_nonempty_line(bootstrap_output)
        stop_payload = _append_openpi_lifecycle_event(
            log_path=log_path,
            lifecycle_events=lifecycle_events,
            workload_key=workload_key,
            checkpoint=checkpoint,
            phase="openpi_stop",
            policy_server_url=policy_server_url,
            bootstrap_command=bootstrap_command,
            bootstrap_metadata=bootstrap_metadata,
            error_message=message,
            teardown_reason="bootstrap_failed",
            teardown_result=_openpi_teardown_result(process_started=False),
        )
        raise ManagedOpenPILifecycleError(
            message or "OPENPI_BOOTSTRAP_ERROR: unable to prepare managed OpenPI runtime.",
            failure_phase="openpi_bootstrap",
            payload={
                "state": "bootstrap_failed",
                "error_message": message or "OPENPI_BOOTSTRAP_ERROR: unable to prepare managed OpenPI runtime.",
                "failure_location": str(bootstrap_log_path),
                "traceback_tail": message or "OPENPI_BOOTSTRAP_ERROR: unable to prepare managed OpenPI runtime.",
                "policy_server_url": policy_server_url,
                "openpi_conda_env": bootstrap_metadata.get("openpi_conda_env") or _resolve_openpi_conda_env(),
                "openpi_repo_root": bootstrap_metadata.get("openpi_repo_root") or _resolve_openpi_repo_root(),
                "checkpoint": checkpoint,
                "bootstrap_log_path": str(bootstrap_log_path),
                "health_log_path": str(health_log_path),
                "server_log_path": str(server_log_path),
                "bootstrap_metadata": bootstrap_metadata,
                "command": bootstrap_command,
                "lifecycle_events": lifecycle_events,
                "teardown": {
                    "reason": "bootstrap_failed",
                    "bootstrap_log_path": str(bootstrap_log_path),
                    "health_log_path": str(health_log_path),
                    "server_log_path": str(server_log_path),
                    "command": bootstrap_command,
                    "result": _openpi_teardown_result(process_started=False),
                    "event": stop_payload,
                },
            },
        )

    try:
        command = _build_openpi_policy_server_command(bootstrap_metadata)
    except Exception as exc:
        error_message = str(exc).strip() or "OPENPI_POLICY_SERVER_STARTUP_FAILED"
        stop_payload = _append_openpi_lifecycle_event(
            log_path=log_path,
            lifecycle_events=lifecycle_events,
            workload_key=workload_key,
            checkpoint=checkpoint,
            phase="openpi_stop",
            policy_server_url=policy_server_url,
            bootstrap_command=bootstrap_command,
            bootstrap_metadata=bootstrap_metadata,
            error_message=error_message,
            teardown_reason="startup_failed",
            teardown_result=_openpi_teardown_result(process_started=False),
        )
        raise ManagedOpenPILifecycleError(
            error_message,
            failure_phase="openpi_policy_server_startup",
            payload={
                "state": "startup_failed",
                "error_message": error_message,
                "failure_location": str(bootstrap_log_path),
                "traceback_tail": error_message,
                "policy_server_url": policy_server_url,
                "openpi_conda_env": bootstrap_metadata.get("openpi_conda_env", ""),
                "openpi_repo_root": bootstrap_metadata.get("openpi_repo_root", ""),
                "checkpoint": checkpoint,
                "bootstrap_log_path": str(bootstrap_log_path),
                "health_log_path": str(health_log_path),
                "server_log_path": str(server_log_path),
                "bootstrap_metadata": bootstrap_metadata,
                "command": bootstrap_command,
                "lifecycle_events": lifecycle_events,
                "teardown": {
                    "reason": "startup_failed",
                    "bootstrap_log_path": str(bootstrap_log_path),
                    "health_log_path": str(health_log_path),
                    "server_log_path": str(server_log_path),
                    "command": bootstrap_command,
                    "result": _openpi_teardown_result(process_started=False),
                    "event": stop_payload,
                },
            },
        ) from exc
    repo_root = Path(bootstrap_metadata["openpi_repo_root"]).resolve()
    log_handle = server_log_path.open("w")
    try:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as exc:
        log_handle.close()
        error_message = str(exc).strip() or "OPENPI_POLICY_SERVER_STARTUP_FAILED"
        stop_payload = _append_openpi_lifecycle_event(
            log_path=log_path,
            lifecycle_events=lifecycle_events,
            workload_key=workload_key,
            checkpoint=checkpoint,
            phase="openpi_stop",
            policy_server_url=policy_server_url,
            bootstrap_command=bootstrap_command,
            bootstrap_metadata=bootstrap_metadata,
            policy_server_command=command,
            error_message=error_message,
            teardown_reason="startup_failed",
            teardown_result=_openpi_teardown_result(process_started=False),
        )
        raise ManagedOpenPILifecycleError(
            error_message,
            failure_phase="openpi_policy_server_startup",
            payload={
                "state": "startup_failed",
                "error_message": error_message,
                "failure_location": str(server_log_path),
                "traceback_tail": error_message,
                "policy_server_url": policy_server_url,
                "openpi_conda_env": bootstrap_metadata.get("openpi_conda_env", ""),
                "openpi_repo_root": bootstrap_metadata.get("openpi_repo_root", ""),
                "checkpoint": checkpoint,
                "bootstrap_log_path": str(bootstrap_log_path),
                "health_log_path": str(health_log_path),
                "server_log_path": str(server_log_path),
                "bootstrap_metadata": bootstrap_metadata,
                "command": command,
                "lifecycle_events": lifecycle_events,
                "teardown": {
                    "reason": "startup_failed",
                    "bootstrap_log_path": str(bootstrap_log_path),
                    "health_log_path": str(health_log_path),
                    "server_log_path": str(server_log_path),
                    "command": command,
                    "result": _openpi_teardown_result(process_started=False),
                    "event": stop_payload,
                },
            },
        ) from exc

    _append_openpi_lifecycle_event(
        log_path=log_path,
        lifecycle_events=lifecycle_events,
        workload_key=workload_key,
        checkpoint=checkpoint,
        phase="openpi_healthcheck_begin",
        policy_server_url=policy_server_url,
        bootstrap_command=bootstrap_command,
        bootstrap_metadata=bootstrap_metadata,
        policy_server_command=command,
    )

    healthy, health_output, health_metadata = _wait_for_openpi_policy_server_health(
        checkpoint=checkpoint,
        policy_server_url=policy_server_url,
        process=process,
        log_path=health_log_path,
        env=env,
    )
    if not healthy:
        teardown_result = _openpi_teardown_result(process_started=True, result=_terminate_process(process))
        log_handle.close()
        details = _tail_nonempty_line(health_output)
        if server_log_path.exists() and (not details or details == "no_output"):
            details = _tail_nonempty_line(server_log_path.read_text())
        error_message = (
            details
            or "OPENPI_POLICY_SERVER_STARTUP_FAILED: managed policy server failed health validation before child launch."
        )
        _append_openpi_lifecycle_event(
            log_path=log_path,
            lifecycle_events=lifecycle_events,
            workload_key=workload_key,
            checkpoint=checkpoint,
            phase="openpi_healthcheck_fail",
            policy_server_url=policy_server_url,
            bootstrap_command=bootstrap_command,
            bootstrap_metadata=bootstrap_metadata,
            policy_server_command=command,
            error_message=error_message,
        )
        stop_payload = _append_openpi_lifecycle_event(
            log_path=log_path,
            lifecycle_events=lifecycle_events,
            workload_key=workload_key,
            checkpoint=checkpoint,
            phase="openpi_stop",
            policy_server_url=policy_server_url,
            bootstrap_command=bootstrap_command,
            bootstrap_metadata=bootstrap_metadata,
            policy_server_command=command,
            error_message=error_message,
            teardown_reason="startup_failed",
            teardown_result=teardown_result,
        )
        raise ManagedOpenPILifecycleError(
            error_message,
            failure_phase="openpi_healthcheck_fail",
            payload={
                "state": "startup_failed",
                "error_message": error_message,
                "failure_location": str(health_log_path),
                "traceback_tail": error_message,
                "policy_server_url": policy_server_url,
                "openpi_conda_env": bootstrap_metadata.get("openpi_conda_env", ""),
                "openpi_repo_root": bootstrap_metadata.get("openpi_repo_root", ""),
                "checkpoint": checkpoint,
                "bootstrap_log_path": str(bootstrap_log_path),
                "health_log_path": str(health_log_path),
                "server_log_path": str(server_log_path),
                "bootstrap_metadata": bootstrap_metadata,
                "health_metadata": health_metadata,
                "command": command,
                "lifecycle_events": lifecycle_events,
                "teardown": {
                    "reason": "startup_failed",
                    "bootstrap_log_path": str(bootstrap_log_path),
                    "health_log_path": str(health_log_path),
                    "server_log_path": str(server_log_path),
                    "command": command,
                    "result": teardown_result,
                    "event": stop_payload,
                },
            },
        )

    _append_openpi_lifecycle_event(
        log_path=log_path,
        lifecycle_events=lifecycle_events,
        workload_key=workload_key,
        checkpoint=checkpoint,
        phase="openpi_healthcheck_pass",
        policy_server_url=policy_server_url,
        bootstrap_command=bootstrap_command,
        bootstrap_metadata=bootstrap_metadata,
        policy_server_command=command,
    )
    return {
        "process": process,
        "log_handle": log_handle,
        "command": command,
        "policy_server_url": policy_server_url,
        "openpi_conda_env": bootstrap_metadata.get("openpi_conda_env", ""),
        "openpi_repo_root": bootstrap_metadata.get("openpi_repo_root", ""),
        "checkpoint": checkpoint,
        "bootstrap_log_path": str(bootstrap_log_path),
        "health_log_path": str(health_log_path),
        "server_log_path": str(server_log_path),
        "bootstrap_metadata": bootstrap_metadata,
        "health_metadata": health_metadata,
        "lifecycle_events": lifecycle_events,
        "log_path": str(log_path),
        "bootstrap_command": bootstrap_command,
    }


def _teardown_managed_openpi_policy_server(workload_key: str, runtime: dict[str, Any] | None, *, reason: str) -> dict[str, Any] | None:
    if runtime is None:
        return None

    log_handle = runtime.get("log_handle")
    process = runtime.get("process")
    teardown_result: dict[str, Any] | None = None
    if isinstance(process, subprocess.Popen):
        teardown_result = _openpi_teardown_result(process_started=True, result=_terminate_process(process))
    if log_handle is not None:
        log_handle.close()
    lifecycle_events = runtime.setdefault("lifecycle_events", [])
    stop_event = _append_openpi_lifecycle_event(
        log_path=Path(str(runtime.get("log_path") or runtime.get("server_log_path") or "")),
        lifecycle_events=lifecycle_events,
        workload_key=workload_key,
        checkpoint=str(runtime.get("checkpoint") or ""),
        phase="openpi_stop",
        policy_server_url=str(runtime.get("policy_server_url") or _resolve_openpi_policy_server_url()),
        bootstrap_command=runtime.get("bootstrap_command"),
        bootstrap_metadata=dict(runtime.get("bootstrap_metadata") or {}),
        policy_server_command=runtime.get("command"),
        teardown_reason=reason,
        teardown_result=teardown_result or _openpi_teardown_result(process_started=False),
    )
    return {
        "reason": reason,
        "bootstrap_log_path": runtime.get("bootstrap_log_path"),
        "health_log_path": runtime.get("health_log_path"),
        "server_log_path": runtime.get("server_log_path"),
        "command": runtime.get("command"),
        "result": teardown_result,
        "event": stop_event,
    }


def _build_policy_server_payload(
    *,
    runtime: dict[str, Any] | None,
    state: str,
    teardown: dict[str, Any] | None = None,
    error_message: str | None = None,
    lifecycle_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if runtime is None and teardown is None and error_message is None:
        return None
    payload: dict[str, Any] = {
        "state": state,
        "error_message": error_message,
        "teardown": teardown,
        "lifecycle_events": lifecycle_events or [],
    }
    if runtime is not None:
        payload.update(
            {
                "policy_server_url": runtime.get("policy_server_url"),
                "openpi_conda_env": runtime.get("openpi_conda_env"),
                "openpi_repo_root": runtime.get("openpi_repo_root"),
                "checkpoint": runtime.get("checkpoint"),
                "bootstrap_log_path": runtime.get("bootstrap_log_path"),
                "health_log_path": runtime.get("health_log_path"),
                "server_log_path": runtime.get("server_log_path"),
                "command": runtime.get("command"),
                "bootstrap_command": runtime.get("bootstrap_command"),
                "bootstrap_metadata": runtime.get("bootstrap_metadata"),
                "health_metadata": runtime.get("health_metadata"),
            }
        )
    return payload


def _failed_execution_result(
    *,
    workload_key: str,
    command: list[str],
    execution_record_path: Path,
    log_path: Path,
    started_at: float,
    error_message: str,
    policy_server: dict[str, Any] | None,
    launch_metadata: dict[str, Any],
    failure_details: dict[str, Any],
) -> dict[str, Any]:
    finished_at = time.time()
    execution_record = _build_execution_record(
        workload_key=workload_key,
        command=command,
        log_path=log_path,
        started_at=started_at,
        finished_at=finished_at,
        status="failed",
        exit_code=1,
        parsed_stdout={},
        failure_details=failure_details,
        policy_server=policy_server,
        launch_metadata=launch_metadata,
    )
    _write_json(execution_record_path, execution_record)
    child = _controller_child_result(
        workload_key=workload_key,
        status="failed",
        exit_code=1,
        command=command,
        log_path=log_path,
        execution_record_path=execution_record_path,
        parsed_stdout={},
        failure_details=failure_details,
        policy_server=policy_server,
        launch_metadata=launch_metadata,
    )
    child["blocker_reason"] = error_message
    return child

def _build_maniskill_inprocess_command() -> list[str]:
    return ["controller_inprocess", "experiments.robot.maniskill.run_maniskill_eval:_eval_maniskill_impl"]


def _build_libero_inprocess_command() -> list[str]:
    return ["controller_inprocess", "experiments.robot.libero.run_libero_eval:_eval_libero_impl"]


def _run_maniskill_inprocess(
    *,
    workflow_request: dict[str, Any],
    workload_key: str,
    session_id: str,
    runtime_plan: dict[str, Any],
    log_path: Path,
    managed_openpi_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command = _build_maniskill_inprocess_command()
    child_env = _build_child_env(
        runtime_plan,
        managed_openpi_runtime={
            "policy_server_url": str(managed_openpi_runtime["policy_server_url"]),
            "openpi_conda_env": str(managed_openpi_runtime["openpi_conda_env"]),
            "openpi_repo_root": str(managed_openpi_runtime["openpi_repo_root"]),
        }
        if managed_openpi_runtime is not None
        else None,
        openpi_checkpoint=str(workflow_request["checkpoint_map"][workload_key]) if workload_key in MANAGED_OPENPI_POLICY_SERVER_WORKLOADS else None,
    )
    launch_metadata = _controller_launch_metadata(
        workload_key=workload_key,
        workflow_request=workflow_request,
        runtime_plan=runtime_plan,
        launch_path="controller_inprocess",
        effective_workload_python=sys.executable,
    )
    child_env = _apply_workload_child_env(child_env, launch_metadata, workload_key=workload_key)

    try:
        with _temporary_env(child_env):
            maniskill_runner = importlib.import_module("experiments.robot.maniskill.run_maniskill_eval")
            cfg = maniskill_runner.build_maniskill_config_from_workflow_request(
                {**workflow_request, "selection": workload_key}
            )
            cfg.run_id_note = f"{session_id}--{workload_key}"
            cfg.controller_log_path = str(log_path)
            result = maniskill_runner._eval_maniskill_impl(cfg)
    except BaseException as exc:
        failure_details = failure_metadata_from_exception(exc, failure_phase="controller_inprocess_maniskill")
        append_breadcrumb_block(
            log_path,
            {**launch_metadata, "timestamp": iso_timestamp(), **failure_details},
            heading="failure",
        )
        return {
            "workload_key": workload_key,
            "status": "failed",
            "exit_code": 1,
            "command": command,
            "parsed_stdout": {},
            "summary_path": None,
            "manifest_path": None,
            "episodes_path": None,
            "run_dir": None,
            "checkpoint": None,
            "average_success_rate": None,
            "per_task_success_rate": {},
            "artifact_paths": {},
            "log_path": str(log_path),
            "blocker_reason": failure_details.get("exception_message"),
            "failure_phase": failure_details.get("failure_phase"),
            "failure_location": failure_details.get("failure_location"),
            "exception_type": failure_details.get("exception_type"),
            "exception_message": failure_details.get("exception_message"),
            "traceback_tail": failure_details.get("traceback_tail"),
            "subprocess_detail": None,
            "launch_path": launch_metadata.get("launch_path"),
            "controller_python": launch_metadata.get("controller_python"),
            "effective_workload_python": launch_metadata.get("effective_workload_python"),
            "selected_gpu": launch_metadata.get("selected_gpu"),
            "selected_cuda_visible_devices": launch_metadata.get("selected_cuda_visible_devices"),
        }

    return {
        "workload_key": workload_key,
        "status": "complete",
        "exit_code": 0,
        "command": command,
        "parsed_stdout": {},
        "summary_path": result.get("summary_path"),
        "manifest_path": result.get("manifest_path"),
        "episodes_path": result.get("episodes_path"),
        "run_dir": result.get("run_dir"),
        "checkpoint": result.get("checkpoint"),
        "average_success_rate": result.get("average_success_rate"),
        "per_task_success_rate": _as_dict(result.get("per_task_success_rate")),
        "artifact_paths": _as_dict(result.get("artifact_paths")),
        "log_path": str(log_path),
        "blocker_reason": None,
        "failure_phase": None,
        "failure_location": None,
        "exception_type": None,
        "exception_message": None,
        "traceback_tail": None,
        "subprocess_detail": None,
        "launch_path": launch_metadata.get("launch_path"),
        "controller_python": launch_metadata.get("controller_python"),
        "effective_workload_python": launch_metadata.get("effective_workload_python"),
        "selected_gpu": launch_metadata.get("selected_gpu"),
        "selected_cuda_visible_devices": launch_metadata.get("selected_cuda_visible_devices"),
    }


def _run_libero_inprocess(
    *,
    workflow_request: dict[str, Any],
    workload_key: str,
    session_id: str,
    runtime_plan: dict[str, Any],
    log_path: Path,
    managed_openpi_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command = _build_libero_inprocess_command()
    child_env = _build_child_env(
        runtime_plan,
        managed_openpi_runtime={
            "policy_server_url": str(managed_openpi_runtime["policy_server_url"]),
            "openpi_conda_env": str(managed_openpi_runtime["openpi_conda_env"]),
            "openpi_repo_root": str(managed_openpi_runtime["openpi_repo_root"]),
        }
        if managed_openpi_runtime is not None
        else None,
        openpi_checkpoint=str(workflow_request["checkpoint_map"][workload_key]) if workload_key in MANAGED_OPENPI_POLICY_SERVER_WORKLOADS else None,
    )
    launch_metadata = _controller_launch_metadata(
        workload_key=workload_key,
        workflow_request=workflow_request,
        runtime_plan=runtime_plan,
        launch_path="controller_inprocess",
        effective_workload_python=sys.executable,
    )
    child_env = _apply_workload_child_env(child_env, launch_metadata, workload_key=workload_key)

    try:
        with _temporary_env(child_env):
            libero_runner = importlib.import_module("experiments.robot.libero.run_libero_eval")
            cfg = libero_runner.build_libero_config_from_workflow_request(
                {**workflow_request, "selection": workload_key}
            )
            cfg.run_id_note = f"{session_id}--{workload_key}"
            cfg.controller_log_path = str(log_path)
            cfg.use_wandb = False
            result = libero_runner._eval_libero_impl(cfg)
    except BaseException as exc:
        failure_details = failure_metadata_from_exception(exc, failure_phase="controller_inprocess_libero")
        append_breadcrumb_block(
            log_path,
            {**launch_metadata, "timestamp": iso_timestamp(), **failure_details},
            heading="failure",
        )
        return {
            "workload_key": workload_key,
            "status": "failed",
            "exit_code": 1,
            "command": command,
            "parsed_stdout": {},
            "summary_path": None,
            "manifest_path": None,
            "episodes_path": None,
            "run_dir": None,
            "checkpoint": None,
            "average_success_rate": None,
            "per_task_success_rate": {},
            "artifact_paths": {},
            "log_path": str(log_path),
            "blocker_reason": failure_details.get("exception_message"),
            "failure_phase": failure_details.get("failure_phase"),
            "failure_location": failure_details.get("failure_location"),
            "exception_type": failure_details.get("exception_type"),
            "exception_message": failure_details.get("exception_message"),
            "traceback_tail": failure_details.get("traceback_tail"),
            "subprocess_detail": None,
            "launch_path": launch_metadata.get("launch_path"),
            "controller_python": launch_metadata.get("controller_python"),
            "effective_workload_python": launch_metadata.get("effective_workload_python"),
            "selected_gpu": launch_metadata.get("selected_gpu"),
            "selected_cuda_visible_devices": launch_metadata.get("selected_cuda_visible_devices"),
        }

    return {
        "workload_key": workload_key,
        "status": "complete",
        "exit_code": 0,
        "command": command,
        "parsed_stdout": {},
        "summary_path": result.get("summary_path"),
        "manifest_path": result.get("manifest_path"),
        "episodes_path": result.get("episodes_path"),
        "run_dir": result.get("run_dir"),
        "checkpoint": result.get("checkpoint"),
        "average_success_rate": result.get("average_success_rate"),
        "per_task_success_rate": _as_dict(result.get("per_task_success_rate")),
        "artifact_paths": _as_dict(result.get("artifact_paths")),
        "log_path": str(log_path),
        "blocker_reason": None,
        "failure_phase": None,
        "failure_location": None,
        "exception_type": None,
        "exception_message": None,
        "traceback_tail": None,
        "subprocess_detail": None,
        "launch_path": launch_metadata.get("launch_path"),
        "controller_python": launch_metadata.get("controller_python"),
        "effective_workload_python": launch_metadata.get("effective_workload_python"),
        "selected_gpu": launch_metadata.get("selected_gpu"),
        "selected_cuda_visible_devices": launch_metadata.get("selected_cuda_visible_devices"),
    }


def _execute_workload(
    workflow_request: dict[str, Any],
    workload_key: str,
    session_id: str,
    session_dir: Path,
    runtime_plan: dict[str, Any],
) -> dict[str, Any]:
    workload_dir = session_dir / workload_key
    workload_dir.mkdir(parents=True, exist_ok=True)
    log_path = workload_dir / "runner.log"
    execution_record_path = workload_dir / "execution.json"
    started_at = time.time()
    managed_openpi_runtime: dict[str, Any] | None = None
    policy_server_payload: dict[str, Any] | None = None
    teardown_reason = "not_started"
    command = (
        _build_maniskill_inprocess_command()
        if workload_key in MANISKILL_WORKLOADS
        else _build_libero_inprocess_command()
    )
    child_env = _build_child_env(runtime_plan)
    launch_metadata = _controller_launch_metadata(
        workload_key=workload_key,
        workflow_request=workflow_request,
        runtime_plan=runtime_plan,
        launch_path="controller_inprocess",
        effective_workload_python=sys.executable,
    )
    append_breadcrumb_block(log_path, launch_metadata, heading="launch")

    if workload_key in MANAGED_OPENPI_POLICY_SERVER_WORKLOADS:
        checkpoint = str(workflow_request["checkpoint_map"][workload_key])
        try:
            managed_openpi_runtime = _start_managed_openpi_policy_server(
                workload_key=workload_key,
                checkpoint=checkpoint,
                workload_dir=workload_dir,
                log_path=log_path,
                env=child_env,
            )
            policy_server_payload = _build_policy_server_payload(
                runtime=managed_openpi_runtime,
                state="healthy",
                lifecycle_events=list(managed_openpi_runtime.get("lifecycle_events") or []),
            )
            child_env = _build_child_env(
                runtime_plan,
                managed_openpi_runtime={
                    "policy_server_url": str(managed_openpi_runtime["policy_server_url"]),
                    "openpi_conda_env": str(managed_openpi_runtime["openpi_conda_env"]),
                    "openpi_repo_root": str(managed_openpi_runtime["openpi_repo_root"]),
                },
                openpi_checkpoint=checkpoint,
            )
            child_env = _apply_workload_child_env(child_env, launch_metadata, workload_key=workload_key)
        except Exception as exc:
            if isinstance(exc, ManagedOpenPILifecycleError):
                payload = dict(exc.payload)
                message = str(payload.get("error_message") or str(exc).strip() or "OPENPI_POLICY_SERVER_STARTUP_FAILED")
                policy_server_payload = dict(payload)
                failure_details = {
                    "failure_phase": exc.failure_phase,
                    "failure_location": payload.get("failure_location") or str(workload_dir / "policy_server_bootstrap.log"),
                    "exception_type": type(exc).__name__,
                    "exception_message": message,
                    "traceback_tail": payload.get("traceback_tail") or message,
                    "subprocess_detail": shlex.join(payload.get("command") or command),
                }
            else:
                message = str(exc).strip() or "OPENPI_POLICY_SERVER_STARTUP_FAILED"
                failure_details = failure_metadata_from_exception(
                    exc,
                    failure_phase="openpi_policy_server_startup",
                    subprocess_detail=shlex.join(command),
                )
            append_breadcrumb_block(
                log_path,
                {**launch_metadata, "timestamp": iso_timestamp(), **failure_details},
                heading="failure",
            )
            if policy_server_payload is None:
                policy_server_payload = {
                    "state": "startup_failed",
                    "error_message": message,
                    "bootstrap_log_path": str(workload_dir / "policy_server_bootstrap.log"),
                    "health_log_path": str(workload_dir / "policy_server_health.log"),
                    "server_log_path": str(workload_dir / "policy_server.log"),
                    "lifecycle_events": [],
                }
            child = _failed_execution_result(
                workload_key=workload_key,
                command=command,
                execution_record_path=execution_record_path,
                log_path=log_path,
                started_at=started_at,
                error_message=message,
                policy_server=policy_server_payload,
                launch_metadata=launch_metadata,
                failure_details=failure_details,
            )
            child["runner"] = str(workflow_request["workload_details"][workload_key]["runner"])
            child["benchmark"] = str(workflow_request["workload_details"][workload_key]["benchmark"])
            child["model_family"] = str(workflow_request["workload_details"][workload_key]["model_family"])
            child["requested_checkpoint"] = str(CHECKPOINT_MAP[workload_key])
            child["requested_artifact_root"] = _resolve_child_artifact_root(workflow_request, workload_key)
            return child

    try:
        if workload_key in MANISKILL_WORKLOADS:
            child = _run_maniskill_inprocess(
                workflow_request=workflow_request,
                workload_key=workload_key,
                session_id=session_id,
                runtime_plan=runtime_plan,
                log_path=log_path,
                managed_openpi_runtime=managed_openpi_runtime,
            )
            process_returncode = int(child["exit_code"])
        elif workload_key in LIBERO_WORKLOADS:
            child = _run_libero_inprocess(
                workflow_request=workflow_request,
                workload_key=workload_key,
                session_id=session_id,
                runtime_plan=runtime_plan,
                log_path=log_path,
                managed_openpi_runtime=managed_openpi_runtime,
            )
            process_returncode = int(child["exit_code"])
        teardown_reason = "child_complete" if child["status"] == "complete" else "child_failed"
    except KeyboardInterrupt:
        teardown_reason = "cancelled"
        raise
    finally:
        teardown_payload = _teardown_managed_openpi_policy_server(
            workload_key,
            managed_openpi_runtime,
            reason=teardown_reason,
        )
        if policy_server_payload is not None:
            policy_server_payload["teardown"] = teardown_payload
            if managed_openpi_runtime is not None:
                policy_server_payload["lifecycle_events"] = list(managed_openpi_runtime.get("lifecycle_events") or [])

    finished_at = time.time()
    failure_details = {
        "failure_phase": child.get("failure_phase"),
        "failure_location": child.get("failure_location"),
        "exception_type": child.get("exception_type"),
        "exception_message": child.get("exception_message") or child.get("blocker_reason"),
        "traceback_tail": child.get("traceback_tail"),
        "subprocess_detail": child.get("subprocess_detail") or shlex.join(command),
    }
    execution_record = _build_execution_record(
        workload_key=workload_key,
        command=command,
        log_path=log_path,
        started_at=started_at,
        finished_at=finished_at,
        status=child["status"],
        exit_code=process_returncode,
        parsed_stdout=child["parsed_stdout"],
        failure_details=failure_details,
        policy_server=policy_server_payload,
        launch_metadata=launch_metadata,
    )
    _write_json(execution_record_path, execution_record)
    child["execution_record_path"] = str(execution_record_path)
    child["runner"] = str(workflow_request["workload_details"][workload_key]["runner"])
    child["benchmark"] = str(workflow_request["workload_details"][workload_key]["benchmark"])
    child["model_family"] = str(workflow_request["workload_details"][workload_key]["model_family"])
    child["requested_checkpoint"] = str(CHECKPOINT_MAP[workload_key])
    child["requested_artifact_root"] = _resolve_child_artifact_root(workflow_request, workload_key)
    child["policy_server"] = policy_server_payload
    return child


def _workflow_status(children: list[dict[str, Any]]) -> str:
    completions = [child["status"] == "complete" for child in children]
    if children and all(completions):
        return "complete"
    if any(completions):
        return "partial"
    return "failed"


def _run_controller() -> int:
    interpreter_resolution = _current_controller_interpreter_resolution()
    print(f"controller_interpreter_resolution={_json_dumps_compact(interpreter_resolution)}")
    print(_describe_controller_interpreter(interpreter_resolution))
    gpu_prompt_context = _compute_gpu_prompt_context()
    try:
        workflow_request = prompt_for_workflow_request(
            input_fn=_controller_prompt_input,
            output_fn=print,
            default_artifact_root=CANONICAL_PARENT_ARTIFACT_ROOT,
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    if bool(workflow_request.get("cancelled")):
        print(f"status={workflow_request['status']}")
        print(f"cancellation_reason={workflow_request['cancellation_reason']}")
        return 0

    session_id = _build_session_id()
    requested_root = str(workflow_request.get("artifact_root", "")).strip() or None
    parent_paths = _build_parent_paths(session_id, artifact_root=requested_root)
    session_dir = parent_paths["session_dir"]
    session_dir.mkdir(parents=True, exist_ok=True)
    controller_log_path = parent_paths["controller_log_path"]

    resolved_workloads = _as_string_list(workflow_request.get("resolved_workloads"))
    runtime_plan = _build_runtime_plan(workflow_request, session_id, parent_paths, gpu_prompt_context=gpu_prompt_context)
    execution_policy = _scheduler_execution_policy(runtime_plan)
    controller_launch_metadata = {
        "timestamp": iso_timestamp(),
        "launch_path": "interactive_cluster_workflow",
        "workload_key": ",".join(resolved_workloads),
        "benchmark": ",".join(
            str(_as_dict(_as_dict(workflow_request.get("workload_details")).get(key)).get("benchmark", ""))
            for key in resolved_workloads
        ),
        "model_family": ",".join(
            str(_as_dict(_as_dict(workflow_request.get("workload_details")).get(key)).get("model_family", ""))
            for key in resolved_workloads
        ),
        "controller_python": str(_current_controller_interpreter_resolution().get("selected_python") or sys.executable),
        "effective_workload_python": None,
        "selected_gpu": execution_policy.get("selected_gpu"),
        "selected_cuda_visible_devices": execution_policy.get("selected_cuda_visible_devices"),
    }
    _write_controller_event(controller_log_path, controller_launch_metadata, heading="launch")
    print(f"execution_model={execution_policy['execution_model']}")
    print(f"scheduler_gpu_heavy_execution={execution_policy['gpu_heavy_execution']}")
    print(f"scheduler_preflight_status={execution_policy['preflight_status']}")
    print(f"scheduler_selected_gpu={execution_policy['selected_gpu']}")
    print(f"scheduler_selected_cuda_visible_devices={execution_policy['selected_cuda_visible_devices']}")
    print(f"scheduler_reason={execution_policy['reason']}")
    if execution_policy.get("preflight_blocker"):
        print(f"scheduler_blocker={execution_policy['preflight_blocker']}")
    workflow_blockers = runtime_plan.get("workflow_blockers") or []
    if workflow_blockers:
        print(f"workflow_blockers={_json_dumps_compact({'workflow_blockers': workflow_blockers})}")
    print(f"workflow_runtime_plan_path={parent_paths['runtime_plan_path']}")
    print(f"scheduled_workloads={len(resolved_workloads)}")
    children: list[dict[str, Any]] = []
    controller_exit_code = 0
    workflow_status_override: str | None = None
    if runtime_plan.get("status") != "ready":
        workflow_status_override = "blocked"
        controller_exit_code = 1
        print("workflow_status=blocked")
        _write_controller_event(
            controller_log_path,
            {
                **controller_launch_metadata,
                "timestamp": iso_timestamp(),
                "failure_phase": "workflow_preflight",
                "failure_location": parent_paths["runtime_plan_path"].name,
                "exception_message": str(_as_dict(runtime_plan.get("gpu_assignment")).get("blocker") or "workflow blocked"),
            },
            heading="failure",
        )
    else:
        try:
            for workload_key in resolved_workloads:
                print(f"launching_workload={workload_key}")
                _write_controller_event(
                    controller_log_path,
                    {
                        **controller_launch_metadata,
                        "timestamp": iso_timestamp(),
                        "workload_key": workload_key,
                        "benchmark": str(_as_dict(_as_dict(workflow_request.get("workload_details")).get(workload_key)).get("benchmark", "")),
                        "model_family": str(_as_dict(_as_dict(workflow_request.get("workload_details")).get(workload_key)).get("model_family", "")),
                        "failure_phase": "dispatch_started",
                    },
                    heading="dispatch",
                )
                child = _execute_workload(workflow_request, workload_key, session_id, session_dir, runtime_plan)
                children.append(child)
                print(f"workload_status={workload_key}:{child['status']}")
                if child["status"] != "complete":
                    _write_controller_event(
                        controller_log_path,
                        {
                            **controller_launch_metadata,
                            "timestamp": iso_timestamp(),
                            "workload_key": workload_key,
                            "benchmark": child.get("benchmark"),
                            "model_family": child.get("model_family"),
                            "effective_workload_python": child.get("effective_workload_python"),
                            "failure_phase": child.get("failure_phase"),
                            "failure_location": child.get("failure_location"),
                            "exception_type": child.get("exception_type"),
                            "exception_message": child.get("exception_message"),
                            "traceback_tail": child.get("traceback_tail"),
                        },
                        heading="failure",
                    )
        except KeyboardInterrupt:
            workflow_status_override = "cancelled"
            controller_exit_code = 130
            print("workflow_status=cancelled")

    summary_payload = {
        "session_id": session_id,
        "workflow_status": workflow_status_override or _workflow_status(children),
        "generated_at_unix": time.time(),
        "selection": workflow_request["selection"],
        "selected_mode": workflow_request["selected_mode"],
        "execution_policy": execution_policy,
        "controller_interpreter": _current_controller_interpreter_resolution(),
        "preflight": _as_dict(runtime_plan.get("gpu_assignment")),
        "runtime_plan": runtime_plan,
        "resolved_workloads": resolved_workloads,
        "workload_count": len(resolved_workloads),
        "checkpoint_map": _as_dict(workflow_request.get("checkpoint_map")),
        "artifact_root": str(parent_paths["session_dir"].parent),
        "artifact_root_overridden": bool(workflow_request.get("artifact_root_overridden", False)),
        "requested_artifact_root": str(workflow_request.get("artifact_root", CANONICAL_PARENT_ARTIFACT_ROOT)),
        "artifact_paths": {
            "session_dir": str(session_dir),
            "summary_path": str(parent_paths["summary_path"]),
            "runtime_plan_path": str(parent_paths["runtime_plan_path"]),
            "controller_log_path": str(controller_log_path),
        },
        "mode_semantics": _as_dict(workflow_request.get("mode_semantics")),
        "children": {child["workload_key"]: child for child in children},
    }
    summary_path = _write_json(parent_paths["summary_path"], summary_payload)
    print(f"workflow_summary_path={summary_path}")
    return controller_exit_code


if __name__ == "__main__":
    raise SystemExit(_run_controller())
