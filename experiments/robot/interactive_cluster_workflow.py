from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import importlib
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.robot.interactive_workflow_contract import (
    CANONICAL_PARENT_ARTIFACT_ROOT,
    CHECKPOINT_MAP,
    prompt_for_workflow_request,
)
from experiments.robot.maniskill.defaults import DEFAULT_GPU_INDEX_ENV_KEY

PARENT_SUMMARY_FILENAME = "workflow_summary.json"
RUNTIME_PLAN_FILENAME = "runtime_plan.json"
CHILD_STDOUT_KEYS = (
    "summary_path",
    "manifest_path",
    "episodes_path",
    "average_success_rate",
    "run_dir",
)
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
)
MANISKILL_WORKLOADS = {"openvla_maniskill_ft", "openpi_maniskill"}
LIBERO_WORKLOADS = {"openvla_libero", "openvla_libero_ft", "openpi_libero"}
MANAGED_OPENPI_POLICY_SERVER_WORKLOADS = {"openpi_maniskill", "openpi_libero"}
DEFAULT_OPENPI_POLICY_CONFIG = "pi05_libero"
POLICY_SERVER_HEALTH_TIMEOUT_SECONDS = 45.0
POLICY_SERVER_HEALTH_POLL_SECONDS = 1.0
GPU_HEAVY_PHASE_NAME = "gpu_eval"
SINGLE_GPU_POLICY_NAME = "single_gpu_v1"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


def _parse_key_values(text: str, keys: tuple[str, ...] = CHILD_STDOUT_KEYS) -> dict[str, str]:
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


def _load_summary_payload(summary_path: str) -> dict[str, Any] | None:
    if not summary_path:
        return None
    path = Path(summary_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _as_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _build_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"WORKFLOW-{timestamp}-pid{os.getpid()}"


def _build_parent_paths(session_id: str) -> dict[str, Path]:
    session_dir = Path(CANONICAL_PARENT_ARTIFACT_ROOT) / session_id
    return {
        "session_dir": session_dir,
        "summary_path": session_dir / PARENT_SUMMARY_FILENAME,
        "runtime_plan_path": session_dir / RUNTIME_PLAN_FILENAME,
    }


def _json_dumps_compact(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=_json_default)


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


def _detect_gpu_assignment() -> dict[str, Any]:
    visible_devices_raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    explicit_gpu_index_raw = os.environ.get(DEFAULT_GPU_INDEX_ENV_KEY, "").strip()
    visible_devices = []
    if visible_devices_raw is not None:
        visible_devices = [entry.strip() for entry in visible_devices_raw.split(",") if entry.strip()]

    assignment: dict[str, Any] = {
        "status": "blocked",
        "policy": SINGLE_GPU_POLICY_NAME,
        "selected_gpu": None,
        "cuda_visible_devices": None,
        "reason": "GPU preflight not evaluated.",
        "blocker": None,
        "host_observation": {
            "cuda_visible_devices_env": visible_devices_raw,
            "visible_devices_entries": visible_devices,
            "required_gpu_index_env_key": DEFAULT_GPU_INDEX_ENV_KEY,
            "required_gpu_index": explicit_gpu_index_raw or None,
        },
        "selection_trace": [],
    }

    if visible_devices_raw is not None and not visible_devices:
        assignment["blocker"] = (
            "GPU_PREFLIGHT_BLOCKED: `CUDA_VISIBLE_DEVICES` is explicitly set but hides all GPUs; "
            "single-GPU scheduling cannot proceed."
        )
        assignment["reason"] = "The host environment explicitly masked every CUDA device."
        return assignment

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        assignment["blocker"] = f"GPU_PREFLIGHT_BLOCKED: unable to import `torch` for CUDA discovery: {exc}"
        assignment["reason"] = "Cannot validate or assign a GPU without torch CUDA discovery."
        return assignment

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    assignment["host_observation"].update(
        {
            "torch_cuda_available": cuda_available,
            "torch_visible_device_count": device_count,
        }
    )

    if not cuda_available or device_count < 1:
        required_label = explicit_gpu_index_raw or (visible_devices_raw if visible_devices_raw is not None else "auto")
        assignment["blocker"] = (
            "GPU_PREFLIGHT_BLOCKED: no CUDA device is available to the controller "
            f"(requested={required_label})."
        )
        assignment["reason"] = "The controller will not pretend single-GPU scheduling succeeded on a no-GPU host."
        return assignment

    if explicit_gpu_index_raw:
        if visible_devices and explicit_gpu_index_raw not in visible_devices:
            assignment["blocker"] = (
                "GPU_PREFLIGHT_BLOCKED: explicit GPU index "
                f"{explicit_gpu_index_raw} is not present in CUDA_VISIBLE_DEVICES={visible_devices_raw!r}."
            )
            assignment["reason"] = "The explicit GPU request conflicts with the visible-device mask."
            return assignment
        assignment["selection_trace"].append("using explicit OPENVLA_MANISKILL_GPU_INDEX")
        selected_gpu = explicit_gpu_index_raw
        selection_source = DEFAULT_GPU_INDEX_ENV_KEY
    elif visible_devices:
        selected_gpu = visible_devices[0]
        selection_source = "CUDA_VISIBLE_DEVICES[first_visible]"
        assignment["selection_trace"].append("selected first visible device from CUDA_VISIBLE_DEVICES")
    else:
        smi_rows, smi_error = _query_nvidia_smi_gpu_rows()
        assignment["host_observation"]["nvidia_smi_rows"] = smi_rows
        assignment["host_observation"]["nvidia_smi_error"] = smi_error
        if smi_rows:
            selected_gpu = str(smi_rows[0]["index"])
            selection_source = "nvidia-smi least-busy GPU"
            assignment["selection_trace"].append("selected least-busy GPU from nvidia-smi")
        else:
            selected_gpu = "0"
            selection_source = "torch first visible GPU fallback"
            assignment["selection_trace"].append("fell back to GPU 0 because nvidia-smi was unavailable")

    assignment.update(
        {
            "status": "ready",
            "selected_gpu": {
                "gpu": selected_gpu,
                "selection_source": selection_source,
                "cuda_visible_devices": selected_gpu,
            },
            "cuda_visible_devices": selected_gpu,
            "reason": "Single-GPU scheduling selected one deterministic CUDA device for all GPU-heavy phases.",
            "blocker": None,
        }
    )
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
    return []


def _build_runtime_plan(
    workflow_request: dict[str, Any],
    session_id: str,
    parent_paths: dict[str, Path],
    *,
    write_output: bool = True,
) -> dict[str, Any]:
    resolved_workloads = _as_string_list(workflow_request.get("resolved_workloads"))
    gpu_assignment = _detect_gpu_assignment()
    runtime_estimate = _load_maniskill_runtime_estimate(str(workflow_request.get("selected_mode", "")))
    requested_parallelism = bool(workflow_request.get("parallel_requested", False))
    integrated_workflow_blockers = _integrated_workflow_blockers(workflow_request)
    scheduling_reason = (
        "GPU-heavy evaluation phases are serialized because Task 8 enforces an honest single-GPU v1 policy. "
        "Only CPU/bootstrap/logging overlap would ever be considered safe, and this controller keeps workload launches serial "
        "instead of faking concurrent GPU evaluation."
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
                "selected_gpu": _as_dict(gpu_assignment.get("selected_gpu")) or None,
                "estimated_work": estimated_work,
                "workload_blockers": [
                    blocker for blocker in integrated_workflow_blockers if blocker.get("workload_key") == workload_key
                ],
                "phases": {
                    "bootstrap_cpu_overlap_allowed": bool(
                        requested_parallelism and workload_key in MANAGED_OPENPI_POLICY_SERVER_WORKLOADS
                    ),
                    "gpu_heavy_execution": "serialized_single_gpu",
                    "gpu_heavy_phase_name": GPU_HEAVY_PHASE_NAME,
                    "logging_collection": "controller_managed",
                },
                "scheduler_notes": [
                    "requested_parallelism_recorded" if requested_parallelism else "parallelism_not_requested",
                    "gpu_heavy_phase_serialized_on_single_gpu",
                ],
            }
        )

    plan_payload = {
        "session_id": session_id,
        "status": "ready" if gpu_assignment.get("status") == "ready" and not integrated_workflow_blockers else "blocked",
        "requested_parallelism": requested_parallelism,
        "actual_parallelism": False,
        "scheduler_policy": SINGLE_GPU_POLICY_NAME,
        "gpu_heavy_execution": "serialized_single_gpu",
        "scheduling_reason": scheduling_reason,
        "gpu_assignment": gpu_assignment,
        "workflow_blockers": integrated_workflow_blockers,
        "runtime_estimate": runtime_estimate,
        "planned_workloads": plan_items,
        "artifact_paths": {
            "runtime_plan_path": str(parent_paths["runtime_plan_path"]),
            "workflow_summary_path": str(parent_paths["summary_path"]),
            "session_dir": str(parent_paths["session_dir"]),
        },
    }
    if write_output:
        _write_json(parent_paths["runtime_plan_path"], plan_payload)
    return plan_payload


def _scheduler_execution_policy(runtime_plan: dict[str, Any]) -> dict[str, Any]:
    gpu_assignment = _as_dict(runtime_plan.get("gpu_assignment"))
    return {
        "policy": SINGLE_GPU_POLICY_NAME,
        "requested_parallelism": bool(runtime_plan.get("requested_parallelism", False)),
        "actual_execution": "serialized_single_gpu_gpu_heavy_phases",
        "actual_parallelism": bool(runtime_plan.get("actual_parallelism", False)),
        "gpu_heavy_execution": str(runtime_plan.get("gpu_heavy_execution", "serialized_single_gpu")),
        "reason": str(runtime_plan.get("scheduling_reason", "")),
        "selected_gpu": _as_dict(gpu_assignment.get("selected_gpu")) or None,
        "cuda_visible_devices": gpu_assignment.get("cuda_visible_devices"),
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
    selected_gpu = _as_dict(gpu_assignment.get("selected_gpu"))
    selected_cuda_visible_devices = str(selected_gpu.get("cuda_visible_devices", "")).strip()
    if selected_cuda_visible_devices:
        child_env["CUDA_VISIBLE_DEVICES"] = selected_cuda_visible_devices
        child_env[DEFAULT_GPU_INDEX_ENV_KEY] = selected_cuda_visible_devices
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


def _resolve_child_artifact_root(workflow_request: dict[str, Any], workload_key: str) -> str:
    if bool(workflow_request.get("artifact_root_overridden", False)):
        return str(workflow_request.get("artifact_root", "")).strip()
    workload_details = workflow_request.get("workload_details", {})
    detail = workload_details.get(workload_key, {}) if isinstance(workload_details, dict) else {}
    return str(detail.get("artifact_root", "")).strip()


def _build_preview_runtime_plan(workflow_request: dict[str, Any]) -> dict[str, Any]:
    parent_paths = {
        "runtime_plan_path": Path(CANONICAL_PARENT_ARTIFACT_ROOT) / "<pending_confirmation>" / RUNTIME_PLAN_FILENAME,
        "summary_path": Path(CANONICAL_PARENT_ARTIFACT_ROOT) / "<pending_confirmation>" / PARENT_SUMMARY_FILENAME,
        "session_dir": Path(CANONICAL_PARENT_ARTIFACT_ROOT) / "<pending_confirmation>",
    }
    runtime_plan = _build_runtime_plan(
        workflow_request,
        "PENDING_CONFIRMATION",
        parent_paths,
        write_output=False,
    )
    runtime_plan["preview_only"] = True
    runtime_plan["artifact_paths"] = {
        "runtime_plan_path": None,
        "workflow_summary_path": None,
        "session_dir": None,
    }
    return runtime_plan


def _emit_plan_preview(workflow_request: dict[str, Any]) -> None:
    runtime_plan = _build_preview_runtime_plan(workflow_request)
    preview_payload = {
        "status": runtime_plan.get("status"),
        "selected_mode": workflow_request.get("selected_mode"),
        "parallel_requested": workflow_request.get("parallel_requested"),
        "artifact_root_overridden": workflow_request.get("artifact_root_overridden"),
        "requested_artifact_root": workflow_request.get("artifact_root"),
        "scheduler_policy": runtime_plan.get("scheduler_policy"),
        "gpu_heavy_execution": runtime_plan.get("gpu_heavy_execution"),
        "gpu_assignment": runtime_plan.get("gpu_assignment"),
        "workflow_blockers": runtime_plan.get("workflow_blockers"),
        "planned_workloads": runtime_plan.get("planned_workloads"),
    }
    print(f"workflow_plan_preview={_json_dumps_compact(preview_payload)}")


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


def _maniskill_openpi_runtime_args() -> list[str]:
    runtime_args: list[str] = []
    policy_server_url = _resolve_openpi_policy_server_url()
    openpi_conda_env = _resolve_openpi_conda_env()
    openpi_repo_root = _resolve_openpi_repo_root()
    if policy_server_url:
        runtime_args.extend(["--pi0_policy_server_url", policy_server_url])
    if openpi_conda_env:
        runtime_args.extend(["--openpi_conda_env", openpi_conda_env])
    if openpi_repo_root:
        runtime_args.extend(["--openpi_repo_root", openpi_repo_root])
    return runtime_args


def _build_command(
    workflow_request: dict[str, Any],
    workload_key: str,
    session_id: str,
    *,
    managed_openpi_runtime: dict[str, str] | None = None,
) -> list[str]:
    workload_details = workflow_request["workload_details"][workload_key]
    model_family = str(workload_details["model_family"])
    checkpoint = str(workflow_request["checkpoint_map"][workload_key])
    artifact_root = _resolve_child_artifact_root(workflow_request, workload_key)
    run_id_note = f"{session_id}--{workload_key}"

    if workload_key in MANISKILL_WORKLOADS:
        command = [
            "python3",
            "experiments/robot/maniskill/run_maniskill_eval.py",
            "--model_family",
            model_family,
            "--mode",
            str(workflow_request["maniskill_mode"]),
            "--episodes_per_task",
            str(workflow_request["maniskill_episodes_per_task"]),
            "--artifact_root",
            artifact_root,
            "--run_id_note",
            run_id_note,
        ]
        if model_family == "pi0":
            command.extend(["--openpi_checkpoint", checkpoint])
            if managed_openpi_runtime is not None:
                command.extend(["--pi0_policy_server_url", managed_openpi_runtime["policy_server_url"]])
                command.extend(["--openpi_conda_env", managed_openpi_runtime["openpi_conda_env"]])
                command.extend(["--openpi_repo_root", managed_openpi_runtime["openpi_repo_root"]])
            else:
                command.extend(_maniskill_openpi_runtime_args())
        else:
            command.extend(["--pretrained_checkpoint", checkpoint])
        return command

    if workload_key in LIBERO_WORKLOADS:
        return [
            "python3",
            "experiments/robot/libero/run_libero_eval.py",
            "--model_family",
            model_family,
            "--pretrained_checkpoint",
            checkpoint,
            "--task_suite_name",
            str(workflow_request["libero_task_suite_name"]),
            "--num_trials_per_task",
            str(workflow_request["libero_num_trials_per_task"]),
            "--use_wandb",
            "False",
            "--artifact_root",
            artifact_root,
            "--run_id_note",
            run_id_note,
        ]

    raise ValueError(f"UNSUPPORTED_WORKLOAD: {workload_key}")


def _build_openpi_bootstrap_command(*, checkpoint: str, policy_server_url: str, require_policy_server_health: bool) -> list[str]:
    command = [
        "python3",
        "experiments/robot/maniskill/bootstrap_openpi.py",
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
    env: dict[str, str] | None,
) -> dict[str, Any]:
    bootstrap_log_path = workload_dir / "policy_server_bootstrap.log"
    health_log_path = workload_dir / "policy_server_health.log"
    server_log_path = workload_dir / "policy_server.log"
    policy_server_url = _resolve_openpi_policy_server_url()
    bootstrap_exit_code, bootstrap_output, bootstrap_metadata = _run_openpi_bootstrap_command(
        checkpoint=checkpoint,
        policy_server_url=policy_server_url,
        require_policy_server_health=False,
        log_path=bootstrap_log_path,
        env=env,
    )
    if bootstrap_exit_code != 0 or bootstrap_metadata.get("openpi_runtime_status") != "ready":
        message = _tail_nonempty_line(bootstrap_output)
        raise RuntimeError(message or "OPENPI_BOOTSTRAP_ERROR: unable to prepare managed OpenPI runtime.")

    command = _build_openpi_policy_server_command(bootstrap_metadata)
    repo_root = Path(bootstrap_metadata["openpi_repo_root"]).resolve()
    print(f"policy_server_lifecycle=start:{workload_key}:{policy_server_url}")
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
    except Exception:
        log_handle.close()
        raise

    healthy, health_output, health_metadata = _wait_for_openpi_policy_server_health(
        checkpoint=checkpoint,
        policy_server_url=policy_server_url,
        process=process,
        log_path=health_log_path,
        env=env,
    )
    if not healthy:
        _ = _terminate_process(process)
        log_handle.close()
        print(f"policy_server_lifecycle=startup_failed:{workload_key}:{policy_server_url}")
        print(f"policy_server_lifecycle=teardown:{workload_key}:startup_failed")
        details = _tail_nonempty_line(health_output)
        if server_log_path.exists() and not details:
            details = _tail_nonempty_line(server_log_path.read_text())
        raise RuntimeError(
            details
            or "OPENPI_POLICY_SERVER_STARTUP_FAILED: managed policy server failed health validation before child launch."
        )

    print(f"policy_server_lifecycle=healthy:{workload_key}:{policy_server_url}")
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
    }


def _teardown_managed_openpi_policy_server(workload_key: str, runtime: dict[str, Any] | None, *, reason: str) -> dict[str, Any] | None:
    if runtime is None:
        return None

    log_handle = runtime.get("log_handle")
    process = runtime.get("process")
    teardown_result: dict[str, Any] | None = None
    if isinstance(process, subprocess.Popen):
        teardown_result = _terminate_process(process)
    if log_handle is not None:
        log_handle.close()
    print(f"policy_server_lifecycle=teardown:{workload_key}:{reason}")
    return {
        "reason": reason,
        "bootstrap_log_path": runtime.get("bootstrap_log_path"),
        "health_log_path": runtime.get("health_log_path"),
        "server_log_path": runtime.get("server_log_path"),
        "command": runtime.get("command"),
        "result": teardown_result,
    }


def _build_policy_server_payload(
    *,
    runtime: dict[str, Any] | None,
    state: str,
    teardown: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any] | None:
    if runtime is None and teardown is None and error_message is None:
        return None
    payload: dict[str, Any] = {
        "state": state,
        "error_message": error_message,
        "teardown": teardown,
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
) -> dict[str, Any]:
    finished_at = time.time()
    execution_record = {
        "workload_key": workload_key,
        "command": command,
        "cwd": str(REPO_ROOT),
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "duration_seconds": finished_at - started_at,
        "status": "failed",
        "exit_code": 1,
        "parsed_stdout": {},
        "error": error_message,
        "log_path": str(log_path),
        "policy_server": policy_server,
    }
    _write_json(execution_record_path, execution_record)
    return {
        "workload_key": workload_key,
        "status": "failed",
        "exit_code": 1,
        "blocker_reason": error_message,
        "command": command,
        "summary_path": None,
        "manifest_path": None,
        "episodes_path": None,
        "run_dir": None,
        "checkpoint": None,
        "average_success_rate": None,
        "per_task_success_rate": {},
        "artifact_paths": {},
        "log_path": str(log_path),
        "parsed_stdout": {},
        "execution_record_path": str(execution_record_path),
        "policy_server": policy_server,
    }


def _parse_child_result(*, workload_key: str, command: list[str], output_text: str, exit_code: int, log_path: Path) -> dict[str, Any]:
    parsed_stdout = _parse_key_values(output_text)
    summary_path = parsed_stdout.get("summary_path", "")
    summary_payload = _load_summary_payload(summary_path)
    manifest_path = parsed_stdout.get("manifest_path", "")
    episodes_path = parsed_stdout.get("episodes_path", "")
    run_dir = parsed_stdout.get("run_dir", "")
    average_success_rate: float | None = None
    checkpoint = None
    artifact_paths: dict[str, Any] = {}
    per_task_success_rate: dict[str, Any] = {}
    blocker_reason = None

    if parsed_stdout.get("average_success_rate"):
        try:
            average_success_rate = float(parsed_stdout["average_success_rate"])
        except Exception:
            average_success_rate = None

    if summary_payload is not None:
        if average_success_rate is None:
            raw_average = summary_payload.get("average_success_rate")
            if raw_average is not None:
                try:
                    average_success_rate = float(raw_average)
                except Exception:
                    average_success_rate = None
        if not manifest_path:
            manifest_path = str(summary_payload.get("artifact_paths", {}).get("manifest", ""))
        if not episodes_path:
            episodes_path = str(summary_payload.get("artifact_paths", {}).get("episodes", ""))
        if not run_dir:
            run_dir = str(summary_payload.get("artifact_paths", {}).get("run_dir", ""))
        checkpoint = summary_payload.get("checkpoint")
        raw_artifact_paths = summary_payload.get("artifact_paths")
        if isinstance(raw_artifact_paths, dict):
            artifact_paths = dict(raw_artifact_paths)
        raw_per_task = summary_payload.get("per_task_success_rate")
        if isinstance(raw_per_task, dict):
            per_task_success_rate = dict(raw_per_task)

    if exit_code == 0 and summary_path:
        status = "complete"
    elif exit_code == 0:
        status = "failed"
        blocker_reason = "missing_summary_path_from_child_stdout"
    else:
        status = "failed"
        blocker_reason = _tail_nonempty_line(output_text)

    return {
        "workload_key": workload_key,
        "status": status,
        "exit_code": exit_code,
        "blocker_reason": blocker_reason,
        "command": command,
        "summary_path": summary_path or None,
        "manifest_path": manifest_path or None,
        "episodes_path": episodes_path or None,
        "run_dir": run_dir or None,
        "checkpoint": checkpoint,
        "average_success_rate": average_success_rate,
        "per_task_success_rate": per_task_success_rate,
        "artifact_paths": artifact_paths,
        "log_path": str(log_path),
        "parsed_stdout": parsed_stdout,
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
    command = _build_command(workflow_request, workload_key, session_id)
    child_env = _build_child_env(runtime_plan)

    if workload_key in MANAGED_OPENPI_POLICY_SERVER_WORKLOADS:
        checkpoint = str(workflow_request["checkpoint_map"][workload_key])
        try:
            managed_openpi_runtime = _start_managed_openpi_policy_server(
                workload_key=workload_key,
                checkpoint=checkpoint,
                workload_dir=workload_dir,
                env=child_env,
            )
            policy_server_payload = _build_policy_server_payload(
                runtime=managed_openpi_runtime,
                state="healthy",
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
            command = _build_command(
                workflow_request,
                workload_key,
                session_id,
                managed_openpi_runtime={
                    "policy_server_url": str(managed_openpi_runtime["policy_server_url"]),
                    "openpi_conda_env": str(managed_openpi_runtime["openpi_conda_env"]),
                    "openpi_repo_root": str(managed_openpi_runtime["openpi_repo_root"]),
                },
            )
        except Exception as exc:
            message = str(exc).strip() or "OPENPI_POLICY_SERVER_STARTUP_FAILED"
            log_path.write_text(f"{message}\n")
            policy_server_payload = {
                "state": "startup_failed",
                "error_message": message,
                "bootstrap_log_path": str(workload_dir / "policy_server_bootstrap.log"),
                "health_log_path": str(workload_dir / "policy_server_health.log"),
                "server_log_path": str(workload_dir / "policy_server.log"),
            }
            child = _failed_execution_result(
                workload_key=workload_key,
                command=command,
                execution_record_path=execution_record_path,
                log_path=log_path,
                started_at=started_at,
                error_message=message,
                policy_server=policy_server_payload,
            )
            child["runner"] = str(workflow_request["workload_details"][workload_key]["runner"])
            child["benchmark"] = str(workflow_request["workload_details"][workload_key]["benchmark"])
            child["model_family"] = str(workflow_request["workload_details"][workload_key]["model_family"])
            child["requested_checkpoint"] = str(CHECKPOINT_MAP[workload_key])
            child["requested_artifact_root"] = _resolve_child_artifact_root(workflow_request, workload_key)
            return child

    try:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output_text = process.stdout or ""
        log_path.write_text(output_text)
        child = _parse_child_result(
            workload_key=workload_key,
            command=command,
            output_text=output_text,
            exit_code=process.returncode,
            log_path=log_path,
        )
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

    finished_at = time.time()
    execution_record = {
        "workload_key": workload_key,
        "command": command,
        "cwd": str(REPO_ROOT),
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "duration_seconds": finished_at - started_at,
        "status": child["status"],
        "exit_code": process.returncode,
        "parsed_stdout": child["parsed_stdout"],
        "error": child["blocker_reason"],
        "log_path": str(log_path),
        "policy_server": policy_server_payload,
    }
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
    try:
        workflow_request = prompt_for_workflow_request(
            preview_callback=_emit_plan_preview,
            default_artifact_root=CANONICAL_PARENT_ARTIFACT_ROOT,
            allow_multiple_workload_selection=True,
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    if bool(workflow_request.get("cancelled")):
        print(f"status={workflow_request['status']}")
        print(f"cancellation_reason={workflow_request['cancellation_reason']}")
        return 0

    session_id = _build_session_id()
    parent_paths = _build_parent_paths(session_id)
    session_dir = parent_paths["session_dir"]
    session_dir.mkdir(parents=True, exist_ok=True)

    resolved_workloads = _as_string_list(workflow_request.get("resolved_workloads"))
    runtime_plan = _build_runtime_plan(workflow_request, session_id, parent_paths)
    execution_policy = _scheduler_execution_policy(runtime_plan)
    print(f"parallel_requested={bool(workflow_request.get('parallel_requested', False))}")
    print(f"scheduler_policy={execution_policy['policy']}")
    print(f"scheduler_gpu_heavy_execution={execution_policy['gpu_heavy_execution']}")
    print(f"scheduler_preflight_status={execution_policy['preflight_status']}")
    print(f"scheduler_selected_cuda_visible_devices={execution_policy['cuda_visible_devices']}")
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
    else:
        try:
            for workload_key in resolved_workloads:
                print(f"launching_workload={workload_key}")
                child = _execute_workload(workflow_request, workload_key, session_id, session_dir, runtime_plan)
                children.append(child)
                print(f"workload_status={workload_key}:{child['status']}")
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
        "parallel_requested": bool(workflow_request.get("parallel_requested", False)),
        "execution_policy": execution_policy,
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
        },
        "mode_semantics": _as_dict(workflow_request.get("mode_semantics")),
        "children": {child["workload_key"]: child for child in children},
    }
    summary_path = _write_json(parent_paths["summary_path"], summary_payload)
    print(f"workflow_summary_path={summary_path}")
    return controller_exit_code


if __name__ == "__main__":
    raise SystemExit(_run_controller())
