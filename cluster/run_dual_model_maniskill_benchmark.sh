#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
  printf '[dual-maniskill-compare] %s\n' "$*"
}

print_usage() {
  cat <<'EOF'
Usage: bash cluster/run_dual_model_maniskill_benchmark.sh

Runs the compare-both dual-model workflow by default:
  1) OpenVLA child eval
  2) pi0 child eval (OpenPI/policy-server path)
  3) parent comparison summary write

Outputs (default root):
  rollouts/maniskill_comparisons/<compare_id>/openvla/
  rollouts/maniskill_comparisons/<compare_id>/pi0/
  rollouts/maniskill_comparisons/<compare_id>/openvla_child.log
  rollouts/maniskill_comparisons/<compare_id>/pi0_child.log
  rollouts/maniskill_comparisons/<compare_id>/comparison_summary.json

OpenVLA and pi0 runtime assumptions are separate. pi0 expects OpenPI settings
(`OPENVLA_MANISKILL_OPENPI_CONDA_ENV`, `OPENVLA_MANISKILL_OPENPI_REPO_ROOT`)
and a reachable policy server endpoint.

Optional environment variables:
  OPENVLA_MANISKILL_COMPARE_ID
  OPENVLA_MANISKILL_COMPARE_ARTIFACT_ROOT (default: rollouts/maniskill_comparisons)
  OPENVLA_MANISKILL_COMPARE_MODE (default: full)
  OPENVLA_MANISKILL_TASK_IDS
  OPENVLA_MANISKILL_EPISODES_PER_TASK
  OPENVLA_MANISKILL_MAX_STEPS_PER_EPISODE
  OPENVLA_MANISKILL_OPENVLA_CHECKPOINT
  OPENVLA_MANISKILL_PI0_CHECKPOINT
  OPENVLA_MANISKILL_PI0_POLICY_SERVER_URL
  OPENVLA_MANISKILL_OPENPI_CONDA_ENV
  OPENVLA_MANISKILL_OPENPI_REPO_ROOT
  OPENVLA_MANISKILL_EVAL_ENTRYPOINT (default: experiments/robot/maniskill/run_maniskill_eval.py)
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print_usage
  exit 0
fi

cd "${REPO_ROOT}"

COMPARE_MODE="${OPENVLA_MANISKILL_COMPARE_MODE:-full}"
COMPARE_ID="${OPENVLA_MANISKILL_COMPARE_ID:-COMPARE-maniskill-$(date +%Y%m%d-%H%M%S)}"
COMPARE_ARTIFACT_ROOT="${OPENVLA_MANISKILL_COMPARE_ARTIFACT_ROOT:-rollouts/maniskill_comparisons}"
COMPARE_DIR="${COMPARE_ARTIFACT_ROOT}/${COMPARE_ID}"
EVAL_ENTRYPOINT="${OPENVLA_MANISKILL_EVAL_ENTRYPOINT:-experiments/robot/maniskill/run_maniskill_eval.py}"
OPENPI_BOOTSTRAP_HELPER="${REPO_ROOT}/experiments/robot/maniskill/bootstrap_openpi.py"

python - "${COMPARE_ID}" "${COMPARE_ARTIFACT_ROOT}" <<'PY'
import sys

from experiments.robot.maniskill.artifacts import create_comparison_layout

create_comparison_layout(compare_id=sys.argv[1], artifact_root=sys.argv[2])
PY

mkdir -p "${COMPARE_DIR}"

OPENVLA_LOG_PATH="${COMPARE_DIR}/openvla_child.log"
PI0_LOG_PATH="${COMPARE_DIR}/pi0_child.log"

COMMON_ARGS=(--mode "${COMPARE_MODE}" --run_id_note "${COMPARE_ID}")
if [[ -n "${OPENVLA_MANISKILL_TASK_IDS:-}" ]]; then
  COMMON_ARGS+=(--task_ids "${OPENVLA_MANISKILL_TASK_IDS}")
fi
if [[ -n "${OPENVLA_MANISKILL_EPISODES_PER_TASK:-}" ]]; then
  COMMON_ARGS+=(--episodes_per_task "${OPENVLA_MANISKILL_EPISODES_PER_TASK}")
fi
if [[ -n "${OPENVLA_MANISKILL_MAX_STEPS_PER_EPISODE:-}" ]]; then
  COMMON_ARGS+=(--max_steps_per_episode "${OPENVLA_MANISKILL_MAX_STEPS_PER_EPISODE}")
fi

OPENVLA_ARGS=(
  --model_family openvla
  --artifact_root "${COMPARE_DIR}/openvla"
)
if [[ -n "${OPENVLA_MANISKILL_OPENVLA_CHECKPOINT:-}" ]]; then
  OPENVLA_ARGS+=(--pretrained_checkpoint "${OPENVLA_MANISKILL_OPENVLA_CHECKPOINT}")
fi

PI0_ARGS=(
  --model_family pi0
  --artifact_root "${COMPARE_DIR}/pi0"
)
if [[ -n "${OPENVLA_MANISKILL_PI0_CHECKPOINT:-}" ]]; then
  PI0_ARGS+=(--openpi_checkpoint "${OPENVLA_MANISKILL_PI0_CHECKPOINT}")
fi

log "Running child model=openvla mode=${COMPARE_MODE}"
set +e
python "${EVAL_ENTRYPOINT}" "${COMMON_ARGS[@]}" "${OPENVLA_ARGS[@]}" >"${OPENVLA_LOG_PATH}" 2>&1
OPENVLA_EXIT_CODE=$?
set -e
log "openvla exit_code=${OPENVLA_EXIT_CODE} log_path=${OPENVLA_LOG_PATH}"

PI0_RUNTIME_ENV_PATH="$(mktemp -t openpi-runtime.XXXXXX.env)"
trap 'rm -f "${PI0_RUNTIME_ENV_PATH}"' EXIT
PI0_RUNTIME_ARGS=(
  --emit-env-file "${PI0_RUNTIME_ENV_PATH}"
  --checkpoint "${OPENVLA_MANISKILL_PI0_CHECKPOINT:-}"
  --policy-server-url "${OPENVLA_MANISKILL_PI0_POLICY_SERVER_URL:-}"
  --openpi-conda-env "${OPENVLA_MANISKILL_OPENPI_CONDA_ENV:-}"
  --require-policy-server-health
)
if [[ -n "${OPENVLA_MANISKILL_OPENPI_REPO_ROOT:-}" ]]; then
  PI0_RUNTIME_ARGS+=(--openpi-repo-root "${OPENVLA_MANISKILL_OPENPI_REPO_ROOT}")
fi

log "Preparing pi0 runtime mode=${COMPARE_MODE}"
set +e
python "${OPENPI_BOOTSTRAP_HELPER}" "${PI0_RUNTIME_ARGS[@]}" >"${PI0_LOG_PATH}" 2>&1
PI0_RUNTIME_EXIT_CODE=$?
set -e
if [[ ${PI0_RUNTIME_EXIT_CODE} -ne 0 ]]; then
  PI0_EXIT_CODE=${PI0_RUNTIME_EXIT_CODE}
  log "pi0 runtime prep failed exit_code=${PI0_EXIT_CODE} log_path=${PI0_LOG_PATH}"
else
  source "${PI0_RUNTIME_ENV_PATH}"
  PI0_ARGS+=(--pi0_policy_server_url "${OPENPI_POLICY_SERVER_URL}")
  PI0_ARGS+=(--openpi_conda_env "${OPENPI_CONDA_ENV}")
  PI0_ARGS+=(--openpi_repo_root "${OPENPI_REPO_ROOT}")
  log "pi0 runtime prep complete cache_state=${OPENPI_BOOTSTRAP_CACHE_STATE} repo_root=${OPENPI_REPO_ROOT}"
  log "Running child model=pi0 mode=${COMPARE_MODE}"
  set +e
  python "${EVAL_ENTRYPOINT}" "${COMMON_ARGS[@]}" "${PI0_ARGS[@]}" >>"${PI0_LOG_PATH}" 2>&1
  PI0_EXIT_CODE=$?
  set -e
fi
log "pi0 exit_code=${PI0_EXIT_CODE} log_path=${PI0_LOG_PATH}"

SUMMARY_PATH="$(python - \
  "${COMPARE_ID}" \
  "${COMPARE_DIR}" \
  "${OPENVLA_EXIT_CODE}" \
  "${OPENVLA_LOG_PATH}" \
  "${PI0_EXIT_CODE}" \
  "${PI0_LOG_PATH}" <<'PY'
import json
import sys
from pathlib import Path
from typing import Any

from experiments.robot.maniskill.artifacts import write_comparison_summary


def parse_log(log_path: Path) -> dict[str, str]:
    parsed = {
        "summary_path": "",
        "manifest_path": "",
        "episodes_path": "",
        "average_success_rate": "",
    }
    if not log_path.exists():
        return parsed

    for line in log_path.read_text().splitlines():
        for key in tuple(parsed.keys()):
            prefix = f"{key}="
            if line.startswith(prefix):
                parsed[key] = line.split("=", 1)[1].strip()
    return parsed


def tail_error_reason(log_path: Path) -> str:
    if not log_path.exists():
        return f"log_missing:{log_path}"
    for line in reversed(log_path.read_text().splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return "no_error_output"


def parse_child(label: str, exit_code: int, log_path: Path) -> dict[str, Any]:
    parsed = parse_log(log_path)
    summary_path = parsed["summary_path"]
    manifest_path = parsed["manifest_path"]
    episodes_path = parsed["episodes_path"]
    average_success_rate = None
    checkpoint = None
    per_task_success_rate: dict[str, Any] = {}
    error = None

    if parsed["average_success_rate"]:
        try:
            average_success_rate = float(parsed["average_success_rate"])
        except Exception:
            average_success_rate = None

    if summary_path and Path(summary_path).is_file():
        summary_payload = json.loads(Path(summary_path).read_text())
        if average_success_rate is None:
            raw_average = summary_payload.get("average_success_rate")
            try:
                average_success_rate = float(raw_average)
            except Exception:
                average_success_rate = None
        checkpoint = summary_payload.get("checkpoint")
        raw_per_task = summary_payload.get("per_task_success_rate")
        if isinstance(raw_per_task, dict):
            per_task_success_rate = raw_per_task
    else:
        summary_path = ""

    if exit_code == 0 and summary_path:
        child_status = "complete"
    elif exit_code == 0 and not summary_path:
        child_status = "failed"
        error = "missing_summary_path_from_child_stdout"
    else:
        child_status = "failed"
        error = tail_error_reason(log_path)

    return {
        "model_family": label,
        "status": child_status,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "episodes_path": episodes_path,
        "checkpoint": checkpoint,
        "average_success_rate": average_success_rate,
        "per_task_success_rate": per_task_success_rate,
        "exit_code": exit_code,
        "error": error,
        "log_path": str(log_path),
    }


compare_id = sys.argv[1]
compare_dir = Path(sys.argv[2])
openvla_exit_code = int(sys.argv[3])
openvla_log_path = Path(sys.argv[4])
pi0_exit_code = int(sys.argv[5])
pi0_log_path = Path(sys.argv[6])

children = {
    "openvla": parse_child("openvla", openvla_exit_code, openvla_log_path),
    "pi0": parse_child("pi0", pi0_exit_code, pi0_log_path),
}

child_completions = [entry["status"] == "complete" for entry in children.values()]
if all(child_completions):
    comparison_status = "complete"
elif any(child_completions):
    comparison_status = "partial"
else:
    comparison_status = "failed"

summary_payload = {
    "compare_id": compare_id,
    "comparison_status": comparison_status,
    "children": children,
    "artifact_paths": {
        "openvla_root": str(compare_dir / "openvla"),
        "pi0_root": str(compare_dir / "pi0"),
    },
}

summary_path = write_comparison_summary(compare_dir=compare_dir, summary_payload=summary_payload)
print(summary_path)
PY
)"

log "comparison complete compare_id=${COMPARE_ID}"
printf 'summary_path=%s\n' "${SUMMARY_PATH}"
