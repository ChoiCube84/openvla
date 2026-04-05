# OpenVLA Cluster Workflow

This repository uses one operator-facing shell entrypoint for cluster runs:

```bash
bash cluster/run_cluster_workflow.sh
```

That wrapper stays attached to the current terminal session and `exec`s:

```bash
python3 experiments/robot/interactive_cluster_workflow.py
```

## What the integrated workflow controls

The controller prompts once for workload selection, mode, requested parallelism, artifact-root override, and confirmation. After the first four inputs, it prints a machine-readable runtime/scheduler preview before you answer the final confirmation prompt. From there it can dispatch the integrated benchmark workloads:

- `openvla_maniskill_ft`
- `openpi_maniskill`
- `openvla_libero`
- `openvla_libero_ft`
- `openpi_libero`

Artifacts from the controller itself are written under:

- `rollouts/cluster_workflow/<session_id>/workflow_summary.json`
- `rollouts/cluster_workflow/<session_id>/runtime_plan.json`

Child runners continue to use their native artifact roots under `rollouts/maniskill/` and `rollouts/libero/` unless you override them during the prompt flow; when you do provide an override, the controller passes that override through to the actual child dispatch command instead of treating it as summary-only metadata.

## Runtime notes

- The controller is the supported way to prepare and launch cluster evaluation flows.
- It uses `python3`, not `python`, for the top-level wrapper.
- GPU-heavy evaluation is scheduled conservatively under the controller's single-GPU policy; requested parallelism is recorded, but the workflow does not pretend concurrent GPU execution succeeded when the host cannot support it.
- `openpi_maniskill` still depends on an OpenPI-capable runtime and a reachable policy server path. The workflow can prepare for that lifecycle, but success still depends on the real host environment.
- `openpi_libero` now uses the same integrated managed OpenPI runtime path as the other pi0 workflow surface; benchmark success still depends on a real OpenPI-capable host/runtime.
- This sandbox is CPU-only / verification-limited, so local documentation here describes the supported workflow surface without claiming benchmark execution succeeds on this host.

## OpenPI policy-server example

If you need to start the supported `pi05_libero` policy server manually from an OpenPI checkout, use the OpenPI environment directly:

```bash
export OPENPI_REPO_ROOT="/path/to/openpi"
export OPENPI_CONDA_ENV="openpi"
cd "${OPENPI_REPO_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${OPENPI_CONDA_ENV}"
python3 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

## Direct child entrypoints

The child runners still exist for implementation/debugging work:

- `experiments/robot/maniskill/run_maniskill_eval.py`
- `experiments/robot/libero/run_libero_eval.py`

Cluster operators should start with `cluster/run_cluster_workflow.sh` so the integrated controller owns prompting, scheduling, runtime planning, and parent-summary capture.

## Upstream reference

For broader OpenVLA training and fine-tuning documentation, refer to upstream:

- https://github.com/openvla/openvla
