#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
  printf '[openvla-maniskill] %s\n' "$*"
}

print_usage() {
  cat <<'EOF'
Usage: bash cluster/run_openvla_maniskill_benchmark.sh

Runs ManiSkill benchmark flow (setup -> estimate -> smoke -> full -> rebake).

Default checkpoint path (Juelg-first):
  Juelg/openvla-7b-finetuned-maniskill

Optional local override:
  OPENVLA_MANISKILL_CHECKPOINT=/path/to/openvla-run-or-checkpoint

Optional GPU/launcher environment variables:
  OPENVLA_MANISKILL_GPU_INDEX
  OPENVLA_MANISKILL_VISIBLE_DEVICES_OVERRIDE
  OPENVLA_MANISKILL_CONDA_ENV
  OPENVLA_MANISKILL_SKIP_CONDA_ACTIVATE=1
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print_usage
  exit 0
fi

choose_gpu_index() {
  python - <<'PY'
import subprocess
import sys

cmd = [
    "nvidia-smi",
    "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    sys.exit(result.returncode)

rows = []
for line in result.stdout.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        continue
    idx, mem, util = map(int, parts)
    rows.append((mem, util, idx))

if not rows:
    raise SystemExit(1)

rows.sort()
print(rows[0][2])
PY
}

OPENVLA_MANISKILL_CONDA_ENV="${OPENVLA_MANISKILL_CONDA_ENV:-openvla}"
if [[ "${OPENVLA_MANISKILL_SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
      conda activate "${OPENVLA_MANISKILL_CONDA_ENV}"
      log "Activated conda env: ${OPENVLA_MANISKILL_CONDA_ENV}"
    else
      log "Conda detected but profile script unavailable; continuing without activation."
    fi
  else
    log "Conda not found on PATH; continuing without activation."
  fi
fi

if [[ "${OPENVLA_MANISKILL_VISIBLE_DEVICES_OVERRIDE+x}" == "x" ]]; then
  export CUDA_VISIBLE_DEVICES="${OPENVLA_MANISKILL_VISIBLE_DEVICES_OVERRIDE}"
elif [[ -n "${OPENVLA_MANISKILL_GPU_INDEX:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${OPENVLA_MANISKILL_GPU_INDEX}"
else
  AUTO_GPU_INDEX="$(choose_gpu_index)"
  export CUDA_VISIBLE_DEVICES="${AUTO_GPU_INDEX}"
  export OPENVLA_MANISKILL_GPU_INDEX="${AUTO_GPU_INDEX}"
fi
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cd "${REPO_ROOT}"

log "Step 1/5: setup preflight"
python experiments/robot/maniskill/check_setup.py

log "Step 2/5: runtime estimate"
ESTIMATE_PATH="$(python experiments/robot/maniskill/estimate_runtime.py)"
if [[ ! -f "${ESTIMATE_PATH}" ]]; then
  log "Runtime estimate path missing: ${ESTIMATE_PATH}"
  exit 1
fi
log "Runtime estimate written to: ${ESTIMATE_PATH}"

log "Step 3/5: smoke benchmark"
SMOKE_ARGS=(--mode smoke)
if [[ -n "${OPENVLA_MANISKILL_CHECKPOINT:-}" ]]; then
  SMOKE_ARGS+=(--pretrained_checkpoint "${OPENVLA_MANISKILL_CHECKPOINT}")
fi
python experiments/robot/maniskill/run_maniskill_eval.py "${SMOKE_ARGS[@]}"

log "Step 4/5: full benchmark"
FULL_LOG_PATH="$(mktemp -t openvla-maniskill-full.XXXXXX.log)"
FULL_ARGS=(--mode full)
if [[ -n "${OPENVLA_MANISKILL_CHECKPOINT:-}" ]]; then
  FULL_ARGS+=(--pretrained_checkpoint "${OPENVLA_MANISKILL_CHECKPOINT}")
fi
python experiments/robot/maniskill/run_maniskill_eval.py "${FULL_ARGS[@]}" | tee "${FULL_LOG_PATH}"

SUMMARY_PATH="$(python - "${FULL_LOG_PATH}" <<'PY'
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary = ""
for line in log_path.read_text().splitlines():
    if line.startswith("summary_path="):
        summary = line.split("=", 1)[1].strip()
print(summary)
PY
)"

AVERAGE_RATE="$(python - "${FULL_LOG_PATH}" <<'PY'
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
rate = ""
for line in log_path.read_text().splitlines():
    if line.startswith("average_success_rate="):
        rate = line.split("=", 1)[1].strip()
print(rate)
PY
)"

if [[ -z "${SUMMARY_PATH}" || ! -f "${SUMMARY_PATH}" ]]; then
  log "Unable to resolve summary path from full run output. Log: ${FULL_LOG_PATH}"
  exit 1
fi

RUN_DIR="$(python - "${SUMMARY_PATH}" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve().parent)
PY
)"

log "Step 5/5: exemplar video rebake"
python experiments/robot/maniskill/rebake_videos.py --run_dir "${RUN_DIR}"

log "Benchmark complete"
log "run_dir=${RUN_DIR}"
log "summary_path=${SUMMARY_PATH}"
log "average_success_rate=${AVERAGE_RATE}"
log "full_log_path=${FULL_LOG_PATH}"
