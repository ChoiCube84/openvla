#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTROLLER_PYTHON_ENV_KEY="OPENVLA_CLUSTER_WORKFLOW_PYTHON"
CONTROLLER_CONDA_ENV_KEY="OPENVLA_CLUSTER_WORKFLOW_CONDA_ENV"
KNOWN_CONTROLLER_CONDA_ENVS=("openvla" "openpi")
INTERPRETER_PROBE_CODE=$'import sys\nimport torch\nprint(f"controller_probe_python={sys.executable}")\nprint(f"controller_probe_torch_version={getattr(torch, \'__version__\', \'unknown\')}")\n'
STARTUP_SELECTION_SOURCE=""
STARTUP_SELECTED_PYTHON=""
STARTUP_SELECTED_CONDA_ENV=""
STARTUP_TORCH_VERSION=""
STARTUP_ATTEMPTS=()

reset_probe_metadata() {
  STARTUP_SELECTED_PYTHON=""
  STARTUP_TORCH_VERSION=""
}

parse_probe_output() {
  local probe_output="$1"
  local line=""
  reset_probe_metadata
  while IFS= read -r line; do
    case "${line}" in
      controller_probe_python=*)
        STARTUP_SELECTED_PYTHON="${line#controller_probe_python=}"
        ;;
      controller_probe_torch_version=*)
        STARTUP_TORCH_VERSION="${line#controller_probe_torch_version=}"
        ;;
    esac
  done <<< "${probe_output}"
  [[ -n "${STARTUP_SELECTED_PYTHON}" ]]
}

record_startup_attempt() {
  local label="$1"
  local status="$2"
  STARTUP_ATTEMPTS+=("${label}:${status}")
}

startup_attempts_json() {
  local first=1
  local attempt=""
  printf '['
  for attempt in "${STARTUP_ATTEMPTS[@]}"; do
    if [[ ${first} -eq 0 ]]; then
      printf ','
    fi
    first=0
    printf '"%s"' "${attempt}"
  done
  printf ']'
}

emit_startup_resolution() {
  local selected_conda_env_json="null"
  if [[ -n "${STARTUP_SELECTED_CONDA_ENV}" ]]; then
    selected_conda_env_json="\"${STARTUP_SELECTED_CONDA_ENV}\""
  fi

  cat <<EOF
controller_startup_interpreter={"status":"ready","selection_source":"${STARTUP_SELECTION_SOURCE}","selected_python":"${STARTUP_SELECTED_PYTHON}","selected_conda_env":${selected_conda_env_json},"torch_version":"${STARTUP_TORCH_VERSION}"}
CONTROLLER_STARTUP_INTERPRETER: source=${STARTUP_SELECTION_SOURCE}; python=${STARTUP_SELECTED_PYTHON}; conda_env=${STARTUP_SELECTED_CONDA_ENV:-none}; torch_version=${STARTUP_TORCH_VERSION}.
EOF
}

probe_python_interpreter() {
  local python_executable="$1"
  local probe_output=""
  [[ -n "${python_executable}" ]] || return 1
  [[ -x "${python_executable}" ]] || return 1
  probe_output="$("${python_executable}" -c "${INTERPRETER_PROBE_CODE}" 2>/dev/null)" || return 1
  parse_probe_output "${probe_output}"
}

probe_conda_env() {
  local env_name="$1"
  local probe_output=""
  [[ -n "${env_name}" ]] || return 1
  command -v conda >/dev/null 2>&1 || return 1
  probe_output="$(conda run --no-capture-output -n "${env_name}" python3 -c "${INTERPRETER_PROBE_CODE}" 2>/dev/null)" || return 1
  parse_probe_output "${probe_output}"
}

resolve_controller_startup_command() {
  local explicit_python="${!CONTROLLER_PYTHON_ENV_KEY:-}"
  local explicit_conda_env="${!CONTROLLER_CONDA_ENV_KEY:-}"
  local env_name=""

  if [[ -n "${explicit_python}" ]] && probe_python_interpreter "${explicit_python}"; then
    STARTUP_COMMAND=("${explicit_python}" "experiments/robot/interactive_cluster_workflow.py")
    STARTUP_SELECTION_SOURCE="explicit_python_env"
    STARTUP_SELECTED_CONDA_ENV=""
    return 0
  elif [[ -n "${explicit_python}" ]]; then
    record_startup_attempt "${CONTROLLER_PYTHON_ENV_KEY}" "unusable"
  fi

  if [[ -n "${explicit_conda_env}" ]] && probe_conda_env "${explicit_conda_env}"; then
    STARTUP_COMMAND=(conda run --no-capture-output -n "${explicit_conda_env}" python3 "experiments/robot/interactive_cluster_workflow.py")
    STARTUP_SELECTION_SOURCE="explicit_conda_env"
    STARTUP_SELECTED_CONDA_ENV="${explicit_conda_env}"
    return 0
  elif [[ -n "${explicit_conda_env}" ]]; then
    record_startup_attempt "${CONTROLLER_CONDA_ENV_KEY}=${explicit_conda_env}" "unusable"
  fi

  for env_name in "${KNOWN_CONTROLLER_CONDA_ENVS[@]}"; do
    if [[ -n "${explicit_conda_env}" && "${env_name}" == "${explicit_conda_env}" ]]; then
      continue
    fi
    if probe_conda_env "${env_name}"; then
      STARTUP_COMMAND=(conda run --no-capture-output -n "${env_name}" python3 "experiments/robot/interactive_cluster_workflow.py")
      STARTUP_SELECTION_SOURCE="known_conda_env:${env_name}"
      STARTUP_SELECTED_CONDA_ENV="${env_name}"
      return 0
    fi
    record_startup_attempt "known_conda_env:${env_name}" "unusable"
  done

  if command -v python3 >/dev/null 2>&1 && probe_python_interpreter "$(command -v python3)"; then
    STARTUP_COMMAND=(python3 "experiments/robot/interactive_cluster_workflow.py")
    STARTUP_SELECTION_SOURCE="path_python3"
    STARTUP_SELECTED_CONDA_ENV=""
    return 0
  fi
  record_startup_attempt "PATH python3" "unusable"

  return 1
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash cluster/run_cluster_workflow.sh

Starts the integrated interactive cluster workflow controller.

Interpreter overrides:
  OPENVLA_CLUSTER_WORKFLOW_PYTHON=/path/to/python3
  OPENVLA_CLUSTER_WORKFLOW_CONDA_ENV=openvla

This wrapper stays attached to the current terminal session and forwards stdin,
stdout, and stderr directly to the selected controller interpreter for:
  experiments/robot/interactive_cluster_workflow.py

Controller interpreter precedence:
  1. OPENVLA_CLUSTER_WORKFLOW_PYTHON
  2. OPENVLA_CLUSTER_WORKFLOW_CONDA_ENV
  3. known conda envs: openvla, then openpi
  4. PATH python3 (must be torch-capable)
EOF
  exit 0
fi

cd "${REPO_ROOT}"
if ! resolve_controller_startup_command; then
  cat <<EOF
controller_interpreter_resolution={"status":"blocked","reason":"wrapper startup could not find a torch-capable controller interpreter before launching Python","requested_python":"${!CONTROLLER_PYTHON_ENV_KEY:-}","requested_conda_env":"${!CONTROLLER_CONDA_ENV_KEY:-}","attempts":$(startup_attempts_json),"remediation":"Set ${CONTROLLER_PYTHON_ENV_KEY}=/absolute/path/to/python3 with torch installed, or set ${CONTROLLER_CONDA_ENV_KEY}=openvla or ${CONTROLLER_CONDA_ENV_KEY}=openpi. Auto-probed conda envs: openvla, openpi; final fallback: PATH python3."}
CONTROLLER_INTERPRETER_RESOLUTION_FAILED: no torch-capable controller interpreter was found. Try ${CONTROLLER_CONDA_ENV_KEY}=openvla, ${CONTROLLER_CONDA_ENV_KEY}=openpi, or ${CONTROLLER_PYTHON_ENV_KEY}=/absolute/path/to/python3 with torch installed. Auto-probed conda envs: openvla, openpi; final fallback: PATH python3.
EOF
  exit 1
fi

emit_startup_resolution
exec "${STARTUP_COMMAND[@]}" "$@"
