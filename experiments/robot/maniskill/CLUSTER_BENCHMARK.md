# OpenVLA ManiSkill Cluster Benchmark (Zero-Arg Launcher)

## 1) Dependencies (cluster node)

From repo root:

```bash
pip install -e .
pip install -r experiments/robot/libero/libero_requirements.txt
pip install mani_skill gymnasium imageio imageio-ffmpeg draccus
pip install packaging ninja psutil
ninja --version
pip install "flash-attn==2.5.5" --no-build-isolation
```

Retry pattern if `flash-attn` build fails:

```bash
pip cache remove flash_attn
MAX_JOBS=4 pip install "flash-attn==2.5.5" --no-build-isolation
```

Use the same environment that runs `python experiments/robot/maniskill/run_maniskill_eval.py`.

## 2) Checkpoint path model (Juelg-first)

Default zero-arg launcher path uses the fallback reference from
`experiments/robot/maniskill/defaults.py`:

- `Juelg/openvla-7b-finetuned-maniskill`

Runtime semantics for this default path are intentionally two-step:

- base model weights from `openvla/openvla-7b`
- processor/config and ManiSkill normalization stats from `Juelg/openvla-7b-finetuned-maniskill`

`openvla/openvla-7b` alone is not a valid ManiSkill benchmark checkpoint for this workflow because it lacks required ManiSkill dataset statistics.

Local checkpoint support remains available as an explicit override using `OPENVLA_MANISKILL_CHECKPOINT`.
Accepted local override forms:

- Run directory containing:
  - `config.json`
  - `dataset_statistics.json`
  - `checkpoints/latest-checkpoint.pt`
- Or an explicit `.pt` file under a `checkpoints/` directory.

Override example:

```bash
export OPENVLA_MANISKILL_CHECKPOINT="/path/to/openvla-run"
```

HF cache/network behavior:

- first run of the default Juelg path requires Hugging Face network access
- later runs can reuse local HF cache when artifacts are already present

## 3) Launch benchmark (setup -> estimate -> smoke -> full -> exemplar rebake)

```bash
bash cluster/run_openvla_maniskill_benchmark.sh
```

Default GPU mapping is launcher-level.

Optional env controls:

- `OPENVLA_MANISKILL_CONDA_ENV` (default: `openvla`)
- `OPENVLA_MANISKILL_SKIP_CONDA_ACTIVATE=1` (skip `conda activate`)
- `OPENVLA_MANISKILL_GPU_INDEX` (pick a specific physical GPU index)
- `OPENVLA_MANISKILL_VISIBLE_DEVICES_OVERRIDE` (QA-only override of `CUDA_VISIBLE_DEVICES`)

Default behavior:

- if `OPENVLA_MANISKILL_VISIBLE_DEVICES_OVERRIDE` is set, the launcher uses it directly
- else if `OPENVLA_MANISKILL_GPU_INDEX` is set, the launcher uses that physical GPU index
- else the launcher auto-selects the least-used GPU from `nvidia-smi`

Choose a specific GPU explicitly:

```bash
OPENVLA_MANISKILL_GPU_INDEX=1 bash cluster/run_openvla_maniskill_benchmark.sh
```

QA failure-path check (forces setup failure before benchmark execution):

```bash
OPENVLA_MANISKILL_VISIBLE_DEVICES_OVERRIDE='' bash cluster/run_openvla_maniskill_benchmark.sh
```

## 4) Outputs

- Runtime estimate: `rollouts/maniskill/runtime_estimate.json`
- Full run artifacts: `rollouts/maniskill/<run_id>/`
  - `summary.json`
  - `manifest.json`
  - `episodes.jsonl`
  - `frames/`
  - `videos/`

Default zero-arg launcher flow now runs `rebake_videos.py` after the full benchmark and writes exemplar MP4s into
`rollouts/maniskill/<run_id>/videos/`; existing MP4s are preserved unless `--overwrite` is explicitly enabled.

Launcher prints `run_dir`, `summary_path`, and `average_success_rate` at completion.

## 5) Rebake exemplar videos from saved frames

```bash
python experiments/robot/maniskill/rebake_videos.py --run_dir rollouts/maniskill/<run_id> [--overwrite]
```
