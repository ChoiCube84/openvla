# Cluster Benchmark Workflow

The cluster benchmark surface is now unified behind one shell wrapper:

```bash
bash cluster/run_cluster_workflow.sh
```

That wrapper immediately hands control to:

```bash
python3 experiments/robot/interactive_cluster_workflow.py
```

## Operator guidance

- Use the integrated workflow for cluster execution instead of separate launcher scripts.
- The controller prompts for workload selection, mode, requested parallelism, artifact-root override, and confirmation in one session, and it prints a machine-readable runtime-plan preview before the final confirmation prompt.
- Parent orchestration artifacts are written to `rollouts/cluster_workflow/<session_id>/`.
- ManiSkill and LIBERO child artifacts remain under their runner-native roots unless you override them, and that override is passed through to the actual child dispatch path.

## Supported workloads

- `openvla_maniskill_ft`
- `openpi_maniskill`
- `openvla_libero`
- `openvla_libero_ft`
- `openpi_libero`

## Runtime honesty

- The controller records requested parallelism and runtime planning, but GPU-heavy work still follows the controller's single-GPU policy.
- `openpi_maniskill` depends on a usable OpenPI runtime plus a reachable policy server endpoint.
- `openpi_libero` now uses the same integrated managed OpenPI runtime path as the other pi0 workflow surface; benchmark success still depends on a real OpenPI-capable host/runtime.
- This repository can prepare for those runtime requirements, but this CPU-only sandbox is not evidence that real GPU execution succeeds locally.

## Manual OpenPI policy-server example

```bash
export OPENPI_REPO_ROOT="/path/to/openpi"
export OPENPI_CONDA_ENV="openpi"
cd "${OPENPI_REPO_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${OPENPI_CONDA_ENV}"
python3 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

## Key outputs

- `rollouts/cluster_workflow/<session_id>/workflow_summary.json`
- `rollouts/cluster_workflow/<session_id>/runtime_plan.json`

For broader repo context and current operator-facing instructions, see the top-level `README.md`.
