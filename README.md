# OpenVLA Cluster Workflow

## 1. Repository purpose

This repository documents and runs the final supported OpenVLA cluster workflow through one direct Python controller entrypoint. The controller owns prompt collection, request validation, runtime planning, artifact creation, child launch attribution, and managed OpenPI lifecycle logging.

## 2. Supported operator entrypoint

The only supported operator launch command is:

```bash
python3 experiments/robot/interactive_cluster_workflow.py
```

Do not use deleted wrapper paths, undocumented shell stubs, or direct child runners as the operator entrypoint. The final contract is controller-first and direct.

## 3. Host expectations

- Desktop and CI hosts without working CUDA are for CPU-safe validation only.
- Real benchmark execution is cluster-only.
- Manual GPU choice is trusted even on a no-GPU desktop host, so local validation can prove prompt/default/runtime contract behavior without claiming local GPU execution success.

## 4. Prerequisites

- Linux shell with `python3`
- An environment where the launched `python3` can import the workflow dependencies
- Repository checkout at `/home/user1/openvla`
- Cluster access for real benchmark execution
- For `openpi_maniskill` and `openpi_libero`, an OpenPI-capable environment and repo checkout reachable by the controller

## 5. Environment setup

Create and activate the environment you want the controller to run in before installing dependencies. Example with conda:

```bash
conda create -n openvla python=3.10 -y
conda activate openvla
pip install -e .
pip install -e ".[dev]"
```

If you run OpenPI workloads, also prepare an OpenPI environment:

```bash
conda create -n openpi python=3.10 -y
conda activate openpi
```

The controller runs in the Python process you launch. There is no wrapper handoff or controller-side re-exec path.

## 6. Supported workloads

The controller accepts one workload selection input, with `all` expanding to the full supported set:

- `all` (default operator selection)
- `openvla_maniskill_ft`
- `openpi_maniskill`
- `openvla_libero`
- `openvla_libero_ft`
- `openpi_libero`

Mode choices are:

- `smoke`
- `full` (default)

## 7. Artifact outputs

The parent controller artifacts live under `rollouts/cluster_workflow/<session_id>/`:

- `runtime_plan.json`
- `workflow_summary.json`
- `controller.log`

The default parent artifact root remains `rollouts/cluster_workflow`, and the third prompt lets you override it.

Child workloads keep their native artifact roots unless overridden through the controller request:

- ManiSkill defaults to `rollouts/maniskill`
- LIBERO defaults to `rollouts/libero`

Runtime and evidence output now center on:

- `selected_gpu`
- `selected_cuda_visible_devices`

## 8. Interactive prompt order and defaults

The live operator contract has exactly five prompt stages in this order:

1. workload selection
2. mode
3. artifact root override
4. GPU number
5. confirmation

### Workload selection

Display:

```text
Workload selection options: all (recommended/default), openvla_maniskill_ft, openpi_maniskill, openvla_libero, openvla_libero_ft, openpi_libero
```

Prompt:

```text
Select workload:
```

Blank behavior: blank selects `all`.

### Mode

Display:

```text
Mode options: smoke, full (recommended/default)
```

Prompt:

```text
Select mode [smoke/full]:
```

Blank behavior: blank selects `full`.

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

### GPU number

Display:

```text
GPU number options: 0, 1 (recommended/default), 2, 3, 4, 5, 6, 7
```

Prompt:

```text
Select GPU number [0/1/2/3/4/5/6/7]:
```

Blank behavior: blank selects `1`.

### Confirmation

Display:

```text
Confirmation semantics: y -> launch workflow, n -> cancel (no default)
```

Prompt:

```text
Confirm workflow request? [y/n]:
```

Blank behavior: blank is invalid; explicit confirmation is required.

## 9. GPU-number contract

- The operator chooses one manual GPU number.
- Valid GPU numbers are `0` through `7`.
- Blank input defaults to `1`.
- The workflow trusts the operator's manual choice even when the local desktop cannot verify GPU visibility.
- The runtime plan carries the chosen value forward as `selected_gpu` and `selected_cuda_visible_devices`.
- Child execution exports `CUDA_VISIBLE_DEVICES=<selected_gpu>` and `OPENVLA_MANISKILL_GPU_INDEX=<selected_gpu>`.
- Invalid values such as `-1`, `8`, and `abc` are rejected explicitly.

## 10. Local CPU-safe checks

These checks are valid on this desktop host and do not claim real GPU-backed benchmark success:

```bash
python3 -m compileall experiments/robot
python3 experiments/robot/interactive_cluster_workflow.py </dev/null
printf '\n\n\n\n\n' | python3 experiments/robot/interactive_cluster_workflow.py
printf '\n\n\n6\ny\n' | python3 experiments/robot/interactive_cluster_workflow.py
```

What local checks can prove:

- Python sources compile
- The controller prints the five-prompt contract in the documented order
- Blank inputs resolve to `all`, `full`, blank/default artifact root, and `1`
- Blank confirmation is rejected because there is no default
- A manual GPU choice like `6` is accepted into the request/runtime path on the no-GPU desktop without host-side vetoing

What local checks cannot prove:

- Real CUDA availability for benchmark execution
- Successful GPU-backed ManiSkill or LIBERO execution
- Successful managed OpenPI policy-server execution on a real cluster runtime

## 11. Cluster-only checks

Run these only on a cluster host with the required workload dependencies installed:

- Launch the controller and confirm the request reaches `status=ready`
- Review `runtime_plan.json` and confirm `selected_gpu` and `selected_cuda_visible_devices`
- Confirm the child environment uses the chosen `CUDA_VISIBLE_DEVICES`
- Run the selected benchmark workload to completion or controlled failure
- Review `workflow_summary.json` for final status and artifact paths
- For OpenPI workloads, confirm bootstrap, health check, launch, and teardown logging

## 12. Troubleshooting

- `INTERACTIVE_INPUT_REQUIRED`: launch the controller in a real terminal and provide stdin
- `INVALID_GPU_NUMBER`: choose a value in `0..7`
- `INVALID_CONFIRMATION`: reply with `y` or `n`; blank confirmation is intentionally invalid
- Dependency import failures: install the repo and workload dependencies in the active Python environment before claiming cluster readiness
