# Cluster Benchmark Workflow

The supported operator entrypoint is the direct Python controller:

```bash
python3 experiments/robot/interactive_cluster_workflow.py
```

## Operator guidance

- Use the integrated workflow for cluster execution instead of separate launcher scripts.
- The controller prompts for workload selection, mode, artifact-root override, GPU number, and confirmation in one session.
- Blank input accepts the documented default only where one exists: workload=`all`, mode=`full`, artifact root=`rollouts/cluster_workflow`, GPU number=`1`. Confirmation has no default and always requires explicit operator input.
- The controller prints a machine-readable runtime-plan preview after the GPU-choice step and before the final confirmation prompt.
- Workload selection supports `all`, `openvla_maniskill_ft`, `openpi_maniskill`, `openvla_libero`, `openvla_libero_ft`, and `openpi_libero`.
- Parent orchestration artifacts are written to `rollouts/cluster_workflow/<session_id>/`.
- ManiSkill and LIBERO child artifacts remain under their runner-native roots unless you override them, and that override is passed through to the actual child dispatch path.
- The controller runs in the Python process you launched; there is no wrapper-managed interpreter discovery or controller re-exec.

## Supported workloads

- `all`
- `openvla_maniskill_ft`
- `openpi_maniskill`
- `openvla_libero`
- `openvla_libero_ft`
- `openpi_libero`

## Interactive prompt contract

The live operator contract has exactly five prompt stages in this order:

1. workload selection
2. mode
3. artifact root override
4. GPU number
5. confirmation

Blank input resolves to the documented default only for workload (`all`), mode (`full`), artifact root (`rollouts/cluster_workflow`), and GPU number (`1`). Confirmation has no default, so blank confirmation is invalid.

GPU number accepts a single controller-provided selection in the range `0..7`. The prompt contract no longer asks for a GPU count or a list of GPU IDs.

## Runtime honesty

- The controller runs GPU-heavy phases serially and does not expose operator-facing parallel scheduling controls.
- GPU selection is a single manual GPU number. Valid values are `0` through `7`, and blank input defaults to `1`.
- The selected GPU is propagated to child runners through both `CUDA_VISIBLE_DEVICES` and `OPENVLA_MANISKILL_GPU_INDEX`, keeping the prompt preview and runtime env aligned as `selected_gpu` and `selected_cuda_visible_devices`.
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
