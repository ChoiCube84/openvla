# Cluster Benchmark Workflow

The cluster benchmark surface is now unified behind one shell wrapper:

```bash
bash cluster/run_cluster_workflow.sh
```

That wrapper stays attached to the terminal, resolves a torch-capable controller interpreter, reports the chosen interpreter/runtime, and then hands control to this controller entrypoint:

```text
experiments/robot/interactive_cluster_workflow.py
```

## Operator guidance

- Use the integrated workflow for cluster execution instead of separate launcher scripts.
- The controller prompts for workload selection, mode, requested parallelism, artifact-root override, GPU choice, and confirmation in one session.
- Blank input always accepts the recommended/default value for the current step: workload=`openvla_maniskill_ft`, mode=`smoke`, parallelism=`n`, artifact root=`rollouts/cluster_workflow`, GPU choice=`auto`/recommended GPU, confirmation=`n` (cancel).
- The controller prints a machine-readable runtime-plan preview after the GPU-choice step and before the final confirmation prompt.
- Workload selection supports `openvla_maniskill_ft`, `openpi_maniskill`, `openvla_libero`, `openvla_libero_ft`, `openpi_libero`, and `all`.
- Parent orchestration artifacts are written to `rollouts/cluster_workflow/<session_id>/`.
- ManiSkill and LIBERO child artifacts remain under their runner-native roots unless you override them, and that override is passed through to the actual child dispatch path.
- Wrapper/controller interpreter precedence is `OPENVLA_CLUSTER_WORKFLOW_PYTHON` → `OPENVLA_CLUSTER_WORKFLOW_CONDA_ENV` → known conda envs (`openvla`, then `openpi`) → `PATH` `python3`. If startup resolution fails, use those env vars as the remediation knobs.

## Supported workloads

- `openvla_maniskill_ft`
- `openpi_maniskill`
- `openvla_libero`
- `openvla_libero_ft`
- `openpi_libero`

## Runtime honesty

- The controller records requested parallelism and runtime planning, but GPU-heavy work still follows the controller's `single_gpu_v1` policy and serializes GPU-heavy phases instead of pretending concurrent GPU execution happened.
- GPU recommendation prefers the least-busy controller-visible GPU from `nvidia-smi`, otherwise the first `CUDA_VISIBLE_DEVICES` entry, otherwise GPU `0` if torch still reports visible CUDA devices.
- `auto` or blank GPU input uses the recommendation; an explicit visible GPU index overrides it. If the requested GPU is not controller-visible, or no controller-visible GPU exists, preview/runtime stay blocked.
- The selected GPU is propagated to child runners through both `CUDA_VISIBLE_DEVICES` and `OPENVLA_MANISKILL_GPU_INDEX`, keeping the prompt preview and runtime env aligned.
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
