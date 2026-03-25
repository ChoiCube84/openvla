# OpenVLA ManiSkill Cluster Benchmark

This repository is a cluster-focused fork of [`openvla/openvla`](https://github.com/openvla/openvla).

Its current purpose is **not** to document the full upstream OpenVLA training/fine-tuning stack. Instead, it is organized around one practical goal:

- run the **Juelg-first OpenVLA ManiSkill benchmark path** from a GPU cluster
- produce a benchmark score
- save exemplar success/failure videos
- keep raw frame artifacts so videos can be regenerated later

## Current Scope

This fork currently supports a FailSafe-aligned baseline workflow for these 3 ManiSkill tasks:

- `PickCube-v1`
- `PushCube-v1`
- `StackCube-v1`

It does **not** implement the FailSafe dual-model recovery method yet.

## Target Cluster Assumptions

The workflow is designed for a Linux GPU cluster environment with:

- Python 3.10
- Conda environment named `openvla`
- NVIDIA RTX A6000 GPUs (48 GB VRAM)

No server hostnames, usernames, mount paths, or private cluster details are required by the documented workflow.

## Main Entry Point

The standard entry point is:

```bash
bash cluster/run_openvla_maniskill_benchmark.sh
```

This launcher performs, in order:

1. setup preflight
2. runtime estimation
3. smoke benchmark
4. full benchmark
5. exemplar video rebake

The launcher is intentionally **zero-argument** for the default path.

GPU selection is launcher-level:

- if `OPENVLA_MANISKILL_VISIBLE_DEVICES_OVERRIDE` is set, the launcher uses that value directly
- else if `OPENVLA_MANISKILL_GPU_INDEX` is set, the launcher uses that physical GPU index
- else the launcher auto-selects the least-used GPU reported by `nvidia-smi`

## Installation on the Cluster

From the repository root:

```bash
pip install -e .
pip install -r experiments/robot/libero/libero_requirements.txt
pip install mani_skill gymnasium imageio imageio-ffmpeg draccus
pip install packaging ninja psutil
ninja --version
pip install "flash-attn==2.5.5" --no-build-isolation
```

If `flash-attn` fails on the first try, a common retry is:

```bash
pip cache remove flash_attn
MAX_JOBS=4 pip install "flash-attn==2.5.5" --no-build-isolation
```

If your cluster requires a specific PyTorch/CUDA installation path, install the correct Torch build in the `openvla` environment first.

## Default Checkpoint Behavior

The default launcher path uses the fallback checkpoint reference defined in:

- `experiments/robot/maniskill/defaults.py`

Default value:

- `Juelg/openvla-7b-finetuned-maniskill`

This ManiSkill workflow is intentionally **Juelg-first**. Runtime semantics for the default path are:

- base model weights from `openvla/openvla-7b`
- processor/config and ManiSkill normalization stats (`dataset_statistics.json`) from `Juelg/openvla-7b-finetuned-maniskill`

Important: `openvla/openvla-7b` **alone** is not treated as a valid ManiSkill benchmark checkpoint for this workflow because it does not provide the required ManiSkill dataset statistics.

To use a local checkpoint run directory or checkpoint file instead, set an explicit override:

```bash
export OPENVLA_MANISKILL_CHECKPOINT="/path/to/checkpoint-or-run-dir"
```

Accepted local override forms are documented in:

- `experiments/robot/maniskill/CLUSTER_BENCHMARK.md`

## Hugging Face Network/Cache Assumptions

For the default Juelg path, first run requires Hugging Face access to resolve model/config/statistics assets.
After first successful download, subsequent runs can reuse the local HF cache.

## Outputs

The workflow writes outputs under:

- `rollouts/maniskill/runtime_estimate.json`
- `rollouts/maniskill/<run_id>/summary.json`
- `rollouts/maniskill/<run_id>/manifest.json`
- `rollouts/maniskill/<run_id>/episodes.jsonl`
- `rollouts/maniskill/<run_id>/frames/`
- `rollouts/maniskill/<run_id>/videos/`

Videos are generated from saved frames after the benchmark run. Existing MP4s are preserved unless rebake is called with `--overwrite`.

## Runtime Estimate

The current estimator output in this repository reports:

- **Smoke run:** about **5.8 seconds**
- **Full run:** about **78.9 seconds**
- **Estimated raw artifact storage:** about **8.26 GB** for the default full run

Practical total wall-clock expectation for the zero-arg launcher is:

- **about 2 to 5 minutes** on a properly configured cluster node for the currently configured default benchmark size

Why this is larger than the raw 78.9-second full estimate:

- Python startup/import overhead
- setup checks
- runtime probe pass
- smoke run before full run
- log parsing and artifact bookkeeping
- exemplar video rebake after the full run

Important caveat:

- this estimate is based on the current implemented probe method and current benchmark defaults
- actual cluster runtime can vary with model-loading overhead, filesystem speed, GPU contention, and ManiSkill rendering performance
- if you change episode counts, seeds, frame retention, or checkpoint behavior, total runtime will change too

For the latest machine-readable estimate, run:

```bash
python experiments/robot/maniskill/estimate_runtime.py
```

## Useful Commands

Setup only:

```bash
python experiments/robot/maniskill/check_setup.py
```

Runtime estimate only:

```bash
python experiments/robot/maniskill/estimate_runtime.py
```

Smoke benchmark only:

```bash
python experiments/robot/maniskill/run_maniskill_eval.py --mode smoke
```

Full benchmark only:

```bash
python experiments/robot/maniskill/run_maniskill_eval.py --mode full
```

Rebake exemplar videos:

```bash
python experiments/robot/maniskill/rebake_videos.py --run_dir rollouts/maniskill/<run_id>
```

Force overwrite during rebake:

```bash
python experiments/robot/maniskill/rebake_videos.py --run_dir rollouts/maniskill/<run_id> --overwrite
```

QA-only launcher failure path for GPU visibility:

```bash
OPENVLA_MANISKILL_VISIBLE_DEVICES_OVERRIDE='' bash cluster/run_openvla_maniskill_benchmark.sh
```

Pick a specific physical GPU yourself:

```bash
OPENVLA_MANISKILL_GPU_INDEX=1 bash cluster/run_openvla_maniskill_benchmark.sh
```

## File Guide

- `cluster/run_openvla_maniskill_benchmark.sh` — zero-arg cluster launcher
- `experiments/robot/maniskill/defaults.py` — benchmark defaults and assumptions
- `experiments/robot/maniskill/check_setup.py` — preflight validation
- `experiments/robot/maniskill/estimate_runtime.py` — runtime/storage estimation
- `experiments/robot/maniskill/run_maniskill_eval.py` — ManiSkill benchmark runner
- `experiments/robot/maniskill/artifacts.py` — output layout and metadata helpers
- `experiments/robot/maniskill/rebake_videos.py` — video regeneration from saved frames
- `experiments/robot/maniskill/CLUSTER_BENCHMARK.md` — concise operational notes

## Repository Hygiene

Local paper PDFs and cluster notes under `informations/` are intentionally ignored by git and should not be pushed.

## Upstream Reference

If you need the original OpenVLA training, fine-tuning, or broader project documentation, refer to the upstream repository:

- https://github.com/openvla/openvla

This fork is intentionally narrower and currently centered on **cluster execution of the ManiSkill baseline benchmark**.
