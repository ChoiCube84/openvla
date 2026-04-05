#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash cluster/run_cluster_workflow.sh

Starts the integrated interactive cluster workflow controller.

This wrapper stays attached to the current terminal session and forwards stdin,
stdout, and stderr directly to:
  python3 experiments/robot/interactive_cluster_workflow.py
EOF
  exit 0
fi

cd "${REPO_ROOT}"
exec python3 "experiments/robot/interactive_cluster_workflow.py" "$@"
