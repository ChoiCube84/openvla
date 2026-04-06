#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Deprecated wrapper stub.

The only supported operator entrypoint is:
  python3 experiments/robot/interactive_cluster_workflow.py

This shell script no longer launches the controller or resolves interpreters.
EOF
  exit 0
fi

cat <<'EOF'
UNSUPPORTED_LAUNCHER: cluster/run_cluster_workflow.sh is no longer a supported operator entrypoint.
Launch the controller directly instead:
  python3 experiments/robot/interactive_cluster_workflow.py
EOF
exit 1
