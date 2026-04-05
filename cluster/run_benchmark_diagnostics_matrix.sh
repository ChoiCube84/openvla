#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPENPI_BOOTSTRAP_HELPER="${REPO_ROOT}/experiments/robot/maniskill/bootstrap_openpi.py"

log() {
  printf '[diagnostics-matrix] %s\n' "$*"
}

SUMMARY_PATH="${OPENVLA_DIAGNOSTICS_SUMMARY_PATH:-${REPO_ROOT}/rollouts/diagnostics/experiment_matrix_summary.json}"
PLAN_ONLY_ARGS=()
if [[ "${OPENVLA_DIAGNOSTICS_PLAN_ONLY:-0}" == "1" ]]; then
  PLAN_ONLY_ARGS+=(--plan-only)
fi

if [[ "${OPENVLA_DIAGNOSTICS_PLAN_ONLY:-0}" != "1" ]]; then
  DIAGNOSTICS_RUNTIME_ENV_PATH="$(mktemp -t openpi-diagnostics-runtime.XXXXXX.env)"
  trap 'rm -f "${DIAGNOSTICS_RUNTIME_ENV_PATH}"' EXIT
  RUNTIME_ARGS=(
    --emit-env-file "${DIAGNOSTICS_RUNTIME_ENV_PATH}"
    --checkpoint "${OPENVLA_DIAGNOSTICS_PI0_CHECKPOINT:-}"
    --policy-server-url "${OPENVLA_MANISKILL_PI0_POLICY_SERVER_URL:-}"
    --openpi-conda-env "${OPENVLA_MANISKILL_OPENPI_CONDA_ENV:-}"
    --require-policy-server-health
  )
  if [[ -n "${OPENVLA_MANISKILL_OPENPI_REPO_ROOT:-}" ]]; then
    RUNTIME_ARGS+=(--openpi-repo-root "${OPENVLA_MANISKILL_OPENPI_REPO_ROOT}")
  fi
  log "Preparing OpenPI runtime for diagnostics matrix pi0 cells"
  python "${OPENPI_BOOTSTRAP_HELPER}" "${RUNTIME_ARGS[@]}"
  source "${DIAGNOSTICS_RUNTIME_ENV_PATH}"
  log "OpenPI runtime ready cache_state=${OPENPI_BOOTSTRAP_CACHE_STATE} repo_root=${OPENPI_REPO_ROOT}"
else
  log "Skipping OpenPI runtime prep because diagnostics plan-only mode is enabled"
fi

log "Writing diagnostics matrix summary to ${SUMMARY_PATH}"
python "${REPO_ROOT}/experiments/robot/maniskill/diagnostics_matrix.py" \
  --summary-path "${SUMMARY_PATH}" \
  "${PLAN_ONLY_ARGS[@]}"
