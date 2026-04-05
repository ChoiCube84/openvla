# OpenVLA ManiSkill Cluster Benchmark

This repository is a cluster-focused fork of [`openvla/openvla`](https://github.com/openvla/openvla) centered on ManiSkill benchmark execution.

## Current Scope

This fork supports three ManiSkill tasks:

- `PickCube-v1`
- `PushCube-v1`
- `StackCube-v1`

and three execution surfaces:

- **dual-model compare-both (primary):** OpenVLA child + pi0 child + parent comparison summary
- single-model OpenVLA (supporting/backward-compatible)
- single-model pi0 (supporting)
- **diagnostics matrix:** structured ManiSkill + LIBERO evidence collection via `cluster/run_benchmark_diagnostics_matrix.sh`

## Primary Entry Point (Dual-Model Compare)

Use this as the primary dual-model benchmark path:

```bash
bash cluster/run_dual_model_maniskill_benchmark.sh
```

The launcher runs two child evals sequentially (`openvla` then `pi0`) and writes one parent summary:

- child 1: `python experiments/robot/maniskill/run_maniskill_eval.py --model_family openvla ...`
- child 2: `python experiments/robot/maniskill/run_maniskill_eval.py --model_family pi0 ...`
- parent: `rollouts/maniskill_comparisons/<compare_id>/comparison_summary.json`

Single-model OpenVLA launcher remains available:

```bash
bash cluster/run_openvla_maniskill_benchmark.sh
```

The cleaned benchmark architecture also includes a separate diagnostics matrix launcher:

```bash
bash cluster/run_benchmark_diagnostics_matrix.sh
```

Use the diagnostics matrix when you want move limit diagnostics and cross-suite evidence instead of a single compare run. The diagnostics matrix keeps ManiSkill baseline-vs-raised move-limit cells separate and writes a machine-readable summary.

## Runtime Environments and OpenPI Requirements

### OpenVLA child runtime

- Conda environment: `openvla`
- Typical deps include ManiSkill/Gymnasium and OpenVLA runtime dependencies.

### pi0 child runtime

The pi0 backend requires OpenPI runtime assumptions that are separate from OpenVLA:

- `OPENPI_CONDA_ENV=openpi` (launcher variable: `OPENVLA_MANISKILL_OPENPI_CONDA_ENV`)
- `OPENPI_REPO_ROOT=/path/to/openpi` (launcher variable: `OPENVLA_MANISKILL_OPENPI_REPO_ROOT`)
- policy server endpoint (launcher variable: `OPENVLA_MANISKILL_PI0_POLICY_SERVER_URL`, default runner value `http://127.0.0.1:8000`)
- default pi0 checkpoint target: `gs://openpi-assets/checkpoints/pi05_libero`

For launcher-driven pi0 paths, the managed OpenPI bootstrap/cache behavior is explicit: when an OpenPI repo root is not provided, the benchmark workflow uses a managed cache under `${XDG_CACHE_HOME:-$HOME/.cache}/openvla/openpi`. That cache is validated before reuse, first bootstrap is recorded as a cache creation event, and later runs reuse the same cache instead of pretending the checkout is ephemeral.

One explicit startup example for the supported `pi05_libero` policy-server path:

```bash
export OPENPI_REPO_ROOT="/path/to/openpi"
export OPENPI_CONDA_ENV="openpi"
cd "${OPENPI_REPO_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${OPENPI_CONDA_ENV}"
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

The current dual launcher forwards OpenPI settings into the pi0 child run. It does **not** claim full automatic OpenPI server lifecycle orchestration; ensure the policy server endpoint you pass is reachable.

The same cache expectation matters for the diagnostics matrix launcher `cluster/run_benchmark_diagnostics_matrix.sh`: diagnostics matrix runs may need the same managed OpenPI cache for pi0-backed cells, but still require a separately reachable policy server.

## OpenVLA Checkpoint Behavior

For OpenVLA runs, default fallback checkpoint behavior remains Juelg-first (`Juelg/openvla-7b-finetuned-maniskill`) with ManiSkill statistics requirements, and local overrides remain supported via explicit checkpoint env/args.

## Artifacts

### Dual-model compare artifacts

Outputs are grouped under:

- `rollouts/maniskill_comparisons/<compare_id>/`
  - `openvla/` (child run tree: `summary.json`, `manifest.json`, `episodes.jsonl`, `frames/`, `videos/`)
  - `pi0/` (child run tree: `summary.json`, `manifest.json`, `episodes.jsonl`, `frames/`, `videos/`)
  - `openvla_child.log`
  - `pi0_child.log`
  - `comparison_summary.json` (parent compare summary)

### Single-model artifacts

- `rollouts/maniskill/runtime_estimate.json`
- `rollouts/maniskill/<run_id>/summary.json`
- `rollouts/maniskill/<run_id>/manifest.json`
- `rollouts/maniskill/<run_id>/episodes.jsonl`
- `rollouts/maniskill/<run_id>/frames/`
- `rollouts/maniskill/<run_id>/videos/`

### Diagnostics matrix artifacts

- `rollouts/diagnostics/experiment_matrix_summary.json`

The diagnostics matrix summary is the discoverable entrypoint for move limit diagnostics and other structured evidence collection.

## Runtime Estimate Guidance (Honest Compare Semantics)

`experiments/robot/maniskill/estimate_runtime.py` currently emits a **single-child** estimate at:

- `rollouts/maniskill/runtime_estimate.json`

There is no dedicated compare-mode estimator file yet. For dual-model compare runs, practical wall-clock should be treated as:

- OpenVLA child runtime
- plus pi0 child runtime
- plus parent orchestration overhead (child process launch, log parse, parent summary write)

Actual runtime still varies with model load time, filesystem speed, GPU contention, rendering throughput, and policy-server responsiveness.

For the diagnostics matrix, wall-clock should be treated honestly as the sum of the launched matrix cells plus summary-writing overhead; it is not the same concept as the single-child runtime estimate.

## Useful Commands

Dual launcher help:

```bash
bash cluster/run_dual_model_maniskill_benchmark.sh --help
```

Diagnostics matrix launcher:

```bash
bash cluster/run_benchmark_diagnostics_matrix.sh
```

Key diagnostics matrix environment variables:

- `OPENVLA_DIAGNOSTICS_PLAN_ONLY=1` — write the diagnostics matrix plan/summary without launching matrix cells
- `OPENVLA_DIAGNOSTICS_SUMMARY_PATH` — override the diagnostics matrix summary output path
- `OPENVLA_DIAGNOSTICS_RAISED_HORIZON` — set the raised ManiSkill move limit used in the baseline-vs-raised comparison
- `OPENVLA_DIAGNOSTICS_OPENVLA_REPO_DEFAULT_CHECKPOINT` — override the diagnostics matrix OpenVLA reference-cell checkpoint (defaults to `openvla/openvla-7b`)
- `OPENVLA_DIAGNOSTICS_OPENVLA_FINETUNED_CHECKPOINT` — override the diagnostics matrix OpenVLA finetuned-cell checkpoint (defaults to `Juelg/openvla-7b-finetuned-maniskill`)
- `OPENVLA_DIAGNOSTICS_PI0_CHECKPOINT` — override the pi0/OpenPI checkpoint used by diagnostics matrix pi0 cells
- `OPENVLA_DIAGNOSTICS_LIBERO_TASK_SUITE` — choose the LIBERO task suite for diagnostics matrix LIBERO cells
- `OPENVLA_DIAGNOSTICS_LIBERO_NUM_TRIALS` — choose the number of trials per LIBERO task in the diagnostics matrix

Single-model setup check:

```bash
python experiments/robot/maniskill/check_setup.py
```

Single-model runtime estimate:

```bash
python experiments/robot/maniskill/estimate_runtime.py
```

Single-model OpenVLA eval:

```bash
python experiments/robot/maniskill/run_maniskill_eval.py --model_family openvla --mode full
```

Single-model pi0 eval example:

```bash
python experiments/robot/maniskill/run_maniskill_eval.py --model_family pi0 --mode full --openpi_conda_env openpi --openpi_repo_root /path/to/openpi
```

## File Guide

- `cluster/run_dual_model_maniskill_benchmark.sh` — dual-model compare launcher
- `cluster/run_openvla_maniskill_benchmark.sh` — single-model OpenVLA launcher
- `cluster/run_benchmark_diagnostics_matrix.sh` — diagnostics matrix launcher for move limit diagnostics and cross-suite evidence
- `experiments/robot/maniskill/run_maniskill_eval.py` — ManiSkill child runner (`openvla` or `pi0`)
- `experiments/robot/maniskill/estimate_runtime.py` — single-child runtime/storage estimation
- `experiments/robot/maniskill/artifacts.py` — child/parent artifact layout helpers
- `experiments/robot/maniskill/CLUSTER_BENCHMARK.md` — operational cluster notes

## Upstream Reference

For broader OpenVLA training/fine-tuning docs, refer to upstream:

- https://github.com/openvla/openvla
