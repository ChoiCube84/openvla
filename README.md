# OpenVLA Cluster Workflow

## 1. Repository purpose

This repository contains the OpenVLA evaluation workflow used to launch one supported cluster-operator path for ManiSkill and LIBERO benchmark runs. The operator-facing controller owns prompt collection, runtime planning, GPU selection validation, parent summary capture, child launch attribution, and managed OpenPI lifecycle logging.

## 2. Supported operator entrypoint

The only supported operator launch command is:

```bash
python3 experiments/robot/interactive_cluster_workflow.py
```

Do not start cluster runs from undocumented shell stubs or from the direct child runners. The controller is the supported entrypoint because it owns the prompt contract, artifact planning, scheduler logging, and failure attribution.

## 3. Desktop vs. cluster hosts

- Desktop or CI hosts without controller-visible CUDA devices are for CPU-safe checks only.
- Real benchmark execution is cluster-only because the controller requires explicit visible GPU IDs before it will schedule GPU-heavy phases.
- On this desktop host, the honest outcome is local validation of prompts, compileability, and blocked/no-GPU behavior. It is not valid to claim local GPU benchmark success here.

## 4. Prerequisites

- Linux shell with `python3`
- An environment where the current `python3` can import the workflow's dependencies
- Repository checkout at `/home/user1/openvla`
- Cluster access with controller-visible CUDA devices for real runs
- For `openpi_maniskill` and `openpi_libero`, an OpenPI-capable environment and repo checkout reachable by the controller

## 5. Environment creation and activation

Create and activate the environment you intend to use for the controller before installing repo dependencies. For example, with conda:

```bash
conda create -n openvla python=3.10 -y
conda activate openvla
```

If you run OpenPI workloads, also prepare an OpenPI environment:

```bash
conda create -n openpi python=3.10 -y
conda activate openpi
```

The controller runs in the Python process you launch. There is no controller-side interpreter auto-discovery or re-exec path.

## 6. Dependency installation

From the repo root, install the project and the local dev extras:

```bash
pip install -e .
pip install -e ".[dev]"
```

The primary package dependencies are defined in `pyproject.toml`, including `torch==2.2.0`, `torchvision==0.17.0`, `torchaudio==2.2.0`, `transformers==4.40.1`, `tensorflow==2.15.0`, `wandb`, and the editable OpenVLA package itself.

## 7. Required environment and cluster config

- OpenPI workloads may need:
  - `OPENPI_CONDA_ENV` (defaults to `openpi`)
  - `OPENPI_REPO_ROOT`
  - `OPENPI_POLICY_SERVER_URL` when reusing an existing server
  - `OPENPI_BOOTSTRAP_REPO_URL` and `OPENPI_BOOTSTRAP_REF` only if you need to override the managed bootstrap source
- GPU selection is manual and based on controller-visible devices. If the controller cannot see CUDA devices, GPU-heavy execution is blocked instead of guessed.
- `CUDA_VISIBLE_DEVICES` may restrict which IDs the controller can see. The IDs you enter must match that visible set exactly.

## 8. Supported workloads and artifact outputs

The controller supports one workload per run:

- `openvla_maniskill_ft`
- `openpi_maniskill`
- `openvla_libero`
- `openvla_libero_ft`
- `openpi_libero`

Mode choices are `smoke` and `full`.

After confirmation, the controller creates parent artifacts under:

- `rollouts/cluster_workflow/<session_id>/runtime_plan.json`
- `rollouts/cluster_workflow/<session_id>/workflow_summary.json`
- `rollouts/cluster_workflow/<session_id>/controller.log`

Child workloads keep their native artifact roots unless you override them:

- ManiSkill defaults to `rollouts/maniskill`
- LIBERO defaults to `rollouts/libero`

For OpenPI workloads, the controller owns the managed bootstrap, policy-server startup, health validation, teardown, and lifecycle logging inside the workload session directory.

## 9. Interactive prompts and defaults

The controller prints these six prompt stages in order. The wording below matches the live prompt contract.

### Workload selection

Display:

```text
Workload selection options: openvla_maniskill_ft (recommended/default), openpi_maniskill, openvla_libero, openvla_libero_ft, openpi_libero
```

Prompt:

```text
Select workload:
```

Blank behavior: blank selects `openvla_maniskill_ft`.

### Mode

Display:

```text
Mode options: smoke (recommended/default), full
```

Prompt:

```text
Select mode [smoke/full]:
```

Blank behavior: blank selects `smoke`.

### Artifact root override

Display:

```text
Artifact root behavior: blank -> rollouts/cluster_workflow (recommended/default), <path> -> explicit override
```

Prompt:

```text
Override artifact root (blank for default):
```

Blank behavior: blank selects `rollouts/cluster_workflow`.

### GPU count

Display template:

```text
GPU count options: 1 (recommended/default), 2 | controller-visible GPU count: <n> | controller-visible GPU IDs: <ids-or-none>
```

Prompt:

```text
Select GPU count [1/2]:
```

Blank behavior: blank selects `1`.

### GPU IDs

Display template:

```text
GPU IDs must be explicit and comma-separated in the requested execution order. Controller-visible GPU IDs: <ids-or-none>
```

Prompt:

```text
Enter GPU ID(s) [example: 0 or 0,1]:
```

Blank behavior: blank is invalid; explicit GPU IDs are required.

### Confirmation

Display:

```text
Confirmation semantics: y -> launch workflow, n -> cancel (recommended/default: n)
```

Prompt:

```text
Confirm workflow request? [y/n]:
```

Blank behavior: blank selects `n`, so the workflow cancels.

## 10. Manual GPU count and GPU ID selection

- The controller requires explicit GPU selection from the operator.
- You must enter a GPU count of `1` or `2`, or leave it blank to accept `1`.
- You must enter explicit GPU IDs that match the selected count and the controller-visible device list.
- Duplicate GPU IDs are rejected.
- If `CUDA_VISIBLE_DEVICES` hides all GPUs or if torch/CUDA discovery fails, the controller reports a `GPU_PREFLIGHT_BLOCKED` reason instead of pretending a launch is ready.
- When selection succeeds, the controller records the ordered IDs in `runtime_plan.json` and propagates them through `CUDA_VISIBLE_DEVICES` plus `OPENVLA_MANISKILL_GPU_INDEX` compatibility logging.
- GPU-heavy phases remain serialized under the controller's direct execution model; there is no operator-facing scheduler policy toggle.

## 11. Manual Slack reporting

Slack reporting is manual only.

- There is no Slack integration.
- There is no webhook posting.
- There is no token-based Slack reporting.
- The recommended manual report is to share the selected workload, mode, chosen GPU count/IDs, `workflow_summary.json`, and any relevant failure location from the controller output.

## 12. Local CPU-safe checks

These are the strongest checks that are valid on a non-GPU desktop host:

```bash
python3 -m compileall experiments/robot
python3 experiments/robot/interactive_cluster_workflow.py
printf '\n\n\n\n0\n\n' | python3 experiments/robot/interactive_cluster_workflow.py
```

What these checks prove locally:

- Python sources under `experiments/robot` compile
- The controller prints the expected prompt contract
- Blank GPU count defaults to `1` before the no-GPU host rejects explicit GPU ID `0`
- Blank confirmation cancels the workflow cleanly
- Missing stdin is rejected with `INTERACTIVE_INPUT_REQUIRED`
- A scripted no-GPU run rejects explicit GPU ID `0` with `INVALID_GPU_IDS` when controller-visible GPU IDs are `none`

`make check` is environment-dependent on this host; it is not part of the CPU-safe baseline unless `black` is available on PATH.

What these checks do not prove locally:

- Successful CUDA discovery from the environment that launched the controller
- Successful GPU-backed ManiSkill or LIBERO execution
- Successful managed OpenPI policy-server startup against a real cluster runtime

## 13. Cluster-only checks

Run these only on a host where the controller can see real CUDA devices and the workload dependencies are installed:

- Launch the controller with explicit GPU count and GPU IDs
- Confirm `scheduler_preflight_status=ready`
- Confirm `execution_model=serial_controller_dispatch`
- Confirm `selected_gpu_count`, `selected_gpu_ids`, and `scheduler_selected_cuda_visible_devices` match the requested devices
- Confirm `workflow_runtime_plan_path` points to a real `runtime_plan.json`
- For OpenPI workloads, confirm managed bootstrap success, policy-server health success, workload launch, and teardown entries in the session logs
- Review `workflow_summary.json` for final workload status and child artifact paths

## 14. Example workflow

Example operator session for a single-GPU smoke run:

```bash
python3 experiments/robot/interactive_cluster_workflow.py
```

Example answers:

```text
Select workload: openvla_maniskill_ft
Select mode [smoke/full]: smoke
Override artifact root (blank for default):
Select GPU count [1/2]: 1
Enter GPU ID(s) [example: 0 or 0,1]: 0
Confirm workflow request? [y/n]: y
```

Expected operator follow-up:

1. Watch the controller output for `execution_model`, `scheduler_preflight_status`, `workflow_runtime_plan_path`, and per-workload status lines.
2. Review `rollouts/cluster_workflow/<session_id>/runtime_plan.json` before trusting the run setup.
3. Review `rollouts/cluster_workflow/<session_id>/workflow_summary.json` after completion or failure.
4. Send any Slack update manually with those artifact paths.

## 15. Troubleshooting

- `INTERACTIVE_INPUT_REQUIRED`: launch the controller directly in a terminal and provide stdin; do not run it without input.
- `GPU_PREFLIGHT_BLOCKED`: the controller cannot validate the requested GPU selection on the current host. Fix CUDA visibility, `CUDA_VISIBLE_DEVICES`, or the torch installation before retrying.
- `INVALID_GPU_COUNT`: only `1` or `2` are accepted.
- `INVALID_GPU_IDS`: provide explicit, non-duplicate GPU IDs that match the visible controller GPU list.
- OpenPI startup failures: confirm `OPENPI_REPO_ROOT`, `OPENPI_CONDA_ENV`, policy-server health, and checkpoint access on the cluster host.
- Child dependency import failures such as missing `tensorflow` or `wandb`: complete the repo dependency installation in the active workload environment before claiming benchmark readiness.
