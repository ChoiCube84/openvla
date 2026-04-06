# OpenVLA Cluster Workflow

This repository uses one operator-facing shell entrypoint for cluster runs:

```bash
bash cluster/run_cluster_workflow.sh
```

That wrapper stays attached to the current terminal session, resolves a torch-capable controller interpreter, prints the selected interpreter/runtime, and then hands control to this controller entrypoint:

```text
experiments/robot/interactive_cluster_workflow.py
```

## What the integrated workflow controls

The controller prompt flow is:

1. workload selection
2. mode
3. parallelism request
4. artifact-root override
5. GPU choice
6. confirmation

After the first five inputs, it prints a machine-readable runtime/scheduler preview before you answer the final confirmation prompt. Blank input always accepts the recommended/default value for the current step. The current prompt contract is:

- Workload selection: `openvla_maniskill_ft` (recommended/default), `openpi_maniskill`, `openvla_libero`, `openvla_libero_ft`, `openpi_libero`, or `all`; blank selects `openvla_maniskill_ft`.
- Mode: `smoke` (recommended/default) or `full`; blank selects `smoke`.
- Parallelism request: `n` (recommended/default) or `y`; blank selects `n`.
- Artifact root override: blank uses `rollouts/cluster_workflow` (recommended/default); any non-blank path becomes the explicit override passed through to the child workload.
- GPU choice: `auto` resolves to the recommended/default GPU; when the controller can see GPUs it also offers explicit visible GPU indices. Blank selects the recommended/default GPU choice.
- Confirmation: `n` is the recommended/default outcome; blank cancels instead of launching.

From there it can dispatch the integrated benchmark workloads:

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
- Wrapper/controller interpreter precedence is:
  1. `OPENVLA_CLUSTER_WORKFLOW_PYTHON=/absolute/path/to/python3`
  2. `OPENVLA_CLUSTER_WORKFLOW_CONDA_ENV=<env>`
  3. known conda envs: `openvla`, then `openpi`
  4. `PATH` `python3` as the final fallback
- The wrapper does not trust bare `python`; every startup candidate must be torch-capable before the controller launches. If startup resolution fails, the remediation knobs are the same two env vars above.
- GPU-heavy evaluation stays under the controller's honest `single_gpu_v1` policy. Requested parallelism is still recorded, but GPU-heavy phases remain serialized instead of pretending concurrent GPU execution succeeded.
- GPU recommendation policy is deterministic:
  1. pick the least-busy controller-visible GPU from `nvidia-smi` when available
  2. otherwise pick the first device exposed by `CUDA_VISIBLE_DEVICES`
  3. otherwise fall back to GPU `0` if torch still reports visible CUDA devices
- Selecting `auto` (or leaving the GPU prompt blank) uses that recommended GPU. Selecting an explicit visible GPU index overrides the recommendation. If no controller-visible CUDA device exists, preview/runtime stay blocked instead of claiming a GPU choice succeeded.
- The selected GPU is propagated to child runners through both `CUDA_VISIBLE_DEVICES` and `OPENVLA_MANISKILL_GPU_INDEX`, so preview, runtime plan, and child env agree on the chosen device.
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
