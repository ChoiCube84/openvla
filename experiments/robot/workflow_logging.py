from __future__ import annotations

import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

WORKFLOW_LAUNCH_PATH_ENV_KEY = "OPENVLA_WORKFLOW_LAUNCH_PATH"
WORKFLOW_WORKLOAD_KEY_ENV_KEY = "OPENVLA_WORKFLOW_WORKLOAD_KEY"
WORKFLOW_BENCHMARK_ENV_KEY = "OPENVLA_WORKFLOW_BENCHMARK"
WORKFLOW_MODEL_FAMILY_ENV_KEY = "OPENVLA_WORKFLOW_MODEL_FAMILY"
WORKFLOW_CONTROLLER_PYTHON_ENV_KEY = "OPENVLA_WORKFLOW_CONTROLLER_PYTHON"
WORKFLOW_SELECTED_GPU_COUNT_ENV_KEY = "OPENVLA_WORKFLOW_SELECTED_GPU_COUNT"
WORKFLOW_SELECTED_GPU_IDS_ENV_KEY = "OPENVLA_WORKFLOW_SELECTED_GPU_IDS"

_TRACEBACK_FRAME_RE = re.compile(r'^\s*File "(?P<file>.+)", line (?P<line>\d+), in (?P<function>.+)$')
_FIELD_ORDER = (
    "timestamp",
    "launch_path",
    "workload_key",
    "benchmark",
    "model_family",
    "controller_python",
    "effective_workload_python",
    "selected_gpu_count",
    "selected_gpu_ids",
    "failure_phase",
    "failure_location",
    "exception_type",
    "exception_message",
    "traceback_tail",
    "subprocess_detail",
)


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stringify_log_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def render_breadcrumb_lines(payload: Mapping[str, Any]) -> str:
    keys = list(_FIELD_ORDER) + [key for key in payload if key not in _FIELD_ORDER]
    lines: list[str] = []
    for key in keys:
        if key not in payload:
            continue
        value = stringify_log_value(payload.get(key))
        if value == "":
            continue
        lines.append(f"{key}={value}")
    return "\n".join(lines)


def append_breadcrumb_block(log_path: Path, payload: Mapping[str, Any], *, heading: str | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    body = render_breadcrumb_lines(payload)
    with log_path.open("a") as f:
        if heading:
            f.write(f"[{heading}]\n")
        if body:
            f.write(body)
            f.write("\n")


def emit_breadcrumb(payload: Mapping[str, Any], *, log_file: Any = None) -> None:
    body = render_breadcrumb_lines(payload)
    if body:
        print(body)
        if log_file is not None:
            log_file.write(body)
            log_file.write("\n")
            log_file.flush()


def traceback_tail_from_text(text: str, *, keep: int = 8) -> str | None:
    lines = text.splitlines()
    start_index = None
    for index, line in enumerate(lines):
        if line.strip().startswith("Traceback (most recent call last):"):
            start_index = index
    if start_index is None:
        return None
    traceback_lines = [line.rstrip() for line in lines[start_index:] if line.strip()]
    if not traceback_lines:
        return None
    return " | ".join(traceback_lines[-keep:])


def traceback_location_from_text(text: str) -> str | None:
    location = None
    for line in text.splitlines():
        match = _TRACEBACK_FRAME_RE.match(line)
        if match:
            location = f"{match.group('file')}:{match.group('line')}:{match.group('function')}"
    return location


def failure_metadata_from_exception(
    exc: BaseException,
    *,
    failure_phase: str,
    subprocess_detail: str | None = None,
) -> dict[str, Any]:
    tb = traceback.extract_tb(exc.__traceback__)
    failure_location = None
    if tb:
        frame = tb[-1]
        failure_location = f"{frame.filename}:{frame.lineno}:{frame.name}"
    formatted = traceback.format_exception(type(exc), exc, exc.__traceback__)
    traceback_tail = None
    if formatted:
        traceback_tail = " | ".join(line.strip() for line in formatted[-8:] if line.strip())
    return {
        "failure_phase": failure_phase,
        "failure_location": failure_location or failure_phase,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc).strip() or type(exc).__name__,
        "traceback_tail": traceback_tail,
        "subprocess_detail": subprocess_detail,
    }


def failure_metadata_from_output(
    output_text: str,
    *,
    failure_phase: str,
    subprocess_detail: str | None = None,
    fallback_exception_message: str | None = None,
) -> dict[str, Any]:
    failure_location = traceback_location_from_text(output_text)
    traceback_tail = traceback_tail_from_text(output_text)
    exception_type = None
    exception_message = fallback_exception_message or "child process failed"
    if traceback_tail:
        tail_parts = [part.strip() for part in traceback_tail.split("|") if part.strip()]
        if tail_parts:
            final_line = tail_parts[-1]
            if ":" in final_line:
                exception_type, _, tail_message = final_line.partition(":")
                exception_type = exception_type.strip() or None
                exception_message = tail_message.strip() or final_line.strip()
            else:
                exception_message = final_line.strip()
    return {
        "failure_phase": failure_phase,
        "failure_location": failure_location or subprocess_detail or failure_phase,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "traceback_tail": traceback_tail,
        "subprocess_detail": subprocess_detail,
    }


def child_launch_metadata_from_env(*, default_launch_path: str, defaults: Mapping[str, Any] | None = None) -> dict[str, Any]:
    defaults = defaults or {}
    return {
        "timestamp": iso_timestamp(),
        "launch_path": os.environ.get(WORKFLOW_LAUNCH_PATH_ENV_KEY, "").strip() or default_launch_path,
        "workload_key": os.environ.get(WORKFLOW_WORKLOAD_KEY_ENV_KEY, "").strip() or defaults.get("workload_key"),
        "benchmark": os.environ.get(WORKFLOW_BENCHMARK_ENV_KEY, "").strip() or defaults.get("benchmark"),
        "model_family": os.environ.get(WORKFLOW_MODEL_FAMILY_ENV_KEY, "").strip() or defaults.get("model_family"),
        "controller_python": os.environ.get(WORKFLOW_CONTROLLER_PYTHON_ENV_KEY, "").strip() or defaults.get("controller_python"),
        "effective_workload_python": sys.executable,
        "selected_gpu_count": os.environ.get(WORKFLOW_SELECTED_GPU_COUNT_ENV_KEY, "").strip() or defaults.get("selected_gpu_count"),
        "selected_gpu_ids": os.environ.get(WORKFLOW_SELECTED_GPU_IDS_ENV_KEY, "").strip() or defaults.get("selected_gpu_ids"),
    }
