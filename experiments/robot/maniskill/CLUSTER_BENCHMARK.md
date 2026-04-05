# OpenVLA ManiSkill Cluster Benchmark

## 1) Benchmark architecture

Primary compare launcher:

```bash
bash cluster/run_dual_model_maniskill_benchmark.sh
```

What it does:

- runs OpenVLA child eval first
- runs pi0 child eval second
- writes one parent `comparison_summary.json`

Each child uses `experiments/robot/maniskill/run_maniskill_eval.py` with a model-specific `--artifact_root` under the same compare directory.

Single-model launchers/runs remain available as supporting modes, but compare-both is the main dual-model path.

The cleaned architecture also includes a diagnostics matrix launcher:

```bash
bash cluster/run_benchmark_diagnostics_matrix.sh
```

The diagnostics matrix is not the same thing as a compare run. It is the structured path for move limit diagnostics, cross-suite evidence collection, and explicit matrix summaries.

## 2) Runtime assumptions by model family

### OpenVLA child

- expected runtime env: `openvla`
- follows existing OpenVLA checkpoint behavior (Juelg-first fallback unless overridden)

### pi0 child

pi0 has separate runtime requirements and is not just an OpenVLA-mode switch.

Required/expected settings:

- `OPENPI_CONDA_ENV=openpi` (forwarded via `OPENVLA_MANISKILL_OPENPI_CONDA_ENV`)
- `OPENPI_REPO_ROOT=/path/to/openpi` (forwarded via `OPENVLA_MANISKILL_OPENPI_REPO_ROOT`)
- policy server endpoint (forwarded via `OPENVLA_MANISKILL_PI0_POLICY_SERVER_URL`; runner default is `http://127.0.0.1:8000`)
- default pi0 checkpoint target is `gs://openpi-assets/checkpoints/pi05_libero`

Managed bootstrap/cache note:

- when an explicit OpenPI checkout is not provided, launcher-driven pi0 paths use a managed cache
- the managed cache location is `${XDG_CACHE_HOME:-$HOME/.cache}/openvla/openpi`
- the same cache expectation applies to diagnostics matrix pi0 cells
- cache preparation validates the checkout; it does not replace the need for a reachable policy server

Explicit OpenPI policy-server startup example for the supported `pi05_libero` target:

```bash
export OPENPI_REPO_ROOT="/path/to/openpi"
export OPENPI_CONDA_ENV="openpi"
cd "${OPENPI_REPO_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${OPENPI_CONDA_ENV}"
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

Important honesty note: the dual launcher forwards settings to the pi0 child, but does not claim full automatic OpenPI server lifecycle management. Ensure the policy-server endpoint is reachable before launch.

## 3) Dual-launcher environment variables

```text
OPENVLA_MANISKILL_COMPARE_ID
OPENVLA_MANISKILL_COMPARE_ARTIFACT_ROOT (default: rollouts/maniskill_comparisons)
OPENVLA_MANISKILL_COMPARE_MODE (default: full)
OPENVLA_MANISKILL_TASK_IDS
OPENVLA_MANISKILL_EPISODES_PER_TASK
OPENVLA_MANISKILL_MAX_STEPS_PER_EPISODE
OPENVLA_MANISKILL_OPENVLA_CHECKPOINT
OPENVLA_MANISKILL_PI0_CHECKPOINT
OPENVLA_MANISKILL_PI0_POLICY_SERVER_URL
OPENVLA_MANISKILL_OPENPI_CONDA_ENV
OPENVLA_MANISKILL_OPENPI_REPO_ROOT
OPENVLA_MANISKILL_EVAL_ENTRYPOINT
```

Show launcher help:

```bash
bash cluster/run_dual_model_maniskill_benchmark.sh --help
```

Diagnostics matrix launcher:

```bash
bash cluster/run_benchmark_diagnostics_matrix.sh
```

Key diagnostics matrix environment variables:

```text
OPENVLA_DIAGNOSTICS_PLAN_ONLY=1
OPENVLA_DIAGNOSTICS_SUMMARY_PATH
OPENVLA_DIAGNOSTICS_RAISED_HORIZON
OPENVLA_DIAGNOSTICS_OPENVLA_REPO_DEFAULT_CHECKPOINT
OPENVLA_DIAGNOSTICS_OPENVLA_FINETUNED_CHECKPOINT
OPENVLA_DIAGNOSTICS_PI0_CHECKPOINT
OPENVLA_DIAGNOSTICS_LIBERO_TASK_SUITE
OPENVLA_DIAGNOSTICS_LIBERO_NUM_TRIALS
```

- `OPENVLA_DIAGNOSTICS_PLAN_ONLY=1` writes the diagnostics matrix plan/summary without launching cells.
- `OPENVLA_DIAGNOSTICS_SUMMARY_PATH` changes where the diagnostics matrix summary is written.
- `OPENVLA_DIAGNOSTICS_RAISED_HORIZON` controls the raised ManiSkill move limit used in the horizon comparison.
- `OPENVLA_DIAGNOSTICS_OPENVLA_REPO_DEFAULT_CHECKPOINT` controls the checkpoint used by diagnostics matrix OpenVLA reference cells (default: `openvla/openvla-7b`).
- `OPENVLA_DIAGNOSTICS_OPENVLA_FINETUNED_CHECKPOINT` controls the checkpoint used by diagnostics matrix OpenVLA finetuned cells (default: `Juelg/openvla-7b-finetuned-maniskill`).
- `OPENVLA_DIAGNOSTICS_PI0_CHECKPOINT` controls the pi0/OpenPI checkpoint used by diagnostics matrix pi0 cells.
- `OPENVLA_DIAGNOSTICS_LIBERO_TASK_SUITE` controls which LIBERO task suite the diagnostics matrix uses.
- `OPENVLA_DIAGNOSTICS_LIBERO_NUM_TRIALS` controls how many trials per LIBERO task the diagnostics matrix runs.

## 4) Artifact layout (compare mode)

Dual-model compare artifacts are kept together under one compare root:

- `rollouts/maniskill_comparisons/<compare_id>/`
  - `openvla/` child artifacts
  - `pi0/` child artifacts
  - `openvla_child.log`
  - `pi0_child.log`
  - `comparison_summary.json`

Each child artifact tree keeps the normal runner outputs (`summary.json`, `manifest.json`, `episodes.jsonl`, `frames/`, `videos/`).

## 5) Runtime estimate guidance

Current machine-readable estimate file remains single-child:

- `rollouts/maniskill/runtime_estimate.json`

There is no dedicated compare estimate generator yet. For compare mode, treat wall-clock as:

- OpenVLA child runtime
- plus pi0 child runtime
- plus orchestration overhead (sequential launch + child log parse + parent summary write)

Do not assume compare runtime equals one child run.

For the diagnostics matrix, do not assume the matrix summary is a single-child estimate. Diagnostics matrix runtime is the sum of launched cells, move limit comparison pairs, and summary-writing overhead.

## 6) Diagnostics matrix and move limit diagnostics

The diagnostics matrix keeps one-factor comparisons explicit.

- ManiSkill move limit diagnostics compare baseline vs raised `max_steps_per_episode`
- the diagnostics matrix writes a machine-readable summary at `rollouts/diagnostics/experiment_matrix_summary.json`
- `cluster/run_benchmark_diagnostics_matrix.sh` is the launcher to use when you want those diagnostics matrix artifacts
- LIBERO and ManiSkill evidence live in the same diagnostics matrix workflow so cross-suite checks are discoverable from one summary

## 7) Supporting single-model commands

OpenVLA single-model launcher (backward compatibility):

```bash
bash cluster/run_openvla_maniskill_benchmark.sh
```

Direct single-model child eval examples:

```bash
python experiments/robot/maniskill/run_maniskill_eval.py --model_family openvla --mode full
python experiments/robot/maniskill/run_maniskill_eval.py --model_family pi0 --mode full --openpi_conda_env openpi --openpi_repo_root /path/to/openpi
```
