# OpenVLA ManiSkill Baseline Benchmark on Cluster

## TL;DR
> **Summary**: Add a cluster-ready, zero-extra-argument benchmark flow inside this fork that evaluates a vanilla OpenVLA ManiSkill baseline in the closest practical way to the FailSafe paper, using fixed GPU 3, sampled exemplar videos, and raw frame retention for later video rebaking.
> **Deliverables**:
> - Direct in-repo ManiSkill benchmark runner for `PickCube-v1`, `PushCube-v1`, `StackCube-v1`
> - Preflight checker + runtime estimator + zero-arg cluster launcher
> - Summary/metadata/frame/video artifact pipeline with rebake support
> - Repo hygiene updates so local `informations/` files never enter Git
> **Effort**: Large
> **Parallel**: YES - 2 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Task 6

## Context
### Original Request
- Reproduce the vanilla OpenVLA baseline benchmark on ManiSkill, as a first step toward the FailSafe paper.
- Target the laboratory GPU cluster instead of the local desktop.
- Code must live in this fork and run without requiring extra user arguments.
- Output must include benchmark score plus success/failure videos.
- Estimate runtime before full execution.
- Keep `/informations` out of GitHub.

### Interview Summary
- Scope is benchmark-only for now; no FailSafe dual-model implementation.
- Prefer direct in-repo implementation over an external harness.
- Execution style: single default script with built-in defaults.
- Default flow: smoke/precheck first, then full benchmark automatically.
- Fixed default GPU index: `3`.
- Video policy: save exemplar success/failure videos by default, not every rollout.
- Rebaking requirement: retain raw frame-level artifacts so videos can be regenerated later.
- Formal test infrastructure is deferred; rely on runner-level verification and agent QA.

### Metis Review (gaps addressed)
- Treat this as **closest practical alignment** to FailSafe, not exact paper reproduction, because the paper leaves key fields unspecified.
- Lock the benchmark to exactly three tasks and avoid general benchmark-framework scope creep.
- Use a launcher-level GPU-3 policy instead of scattering device decisions across Python code.
- Define a single artifact contract up front, including raw frames, summary JSON, and exemplar-video selection rules.
- Include explicit preflight checks for GPU visibility, rendering, checkpoint presence, and disk budget before full evaluation.

## Work Objectives
### Core Objective
Create one in-repo, cluster-ready ManiSkill benchmark path that can be launched with no extra arguments and produces a reproducible baseline report for vanilla OpenVLA on the three FailSafe-aligned tasks.

### Deliverables
- ManiSkill benchmark package under `experiments/robot/maniskill/`
- Zero-arg launcher under `cluster/`
- Artifact contract under `rollouts/maniskill/<run_id>/`
- `.gitignore` update for `informations/`
- Assumption ledger embedded in benchmark outputs

### Definition of Done (verifiable conditions with commands)
- `python experiments/robot/maniskill/check_setup.py` exits 0 and prints `SETUP_OK`
- `python experiments/robot/maniskill/estimate_runtime.py` exits 0 and writes a JSON runtime estimate for smoke + full runs
- `bash cluster/run_openvla_maniskill_benchmark.sh` exits 0 on a properly provisioned cluster node and writes summary artifacts
- `rollouts/maniskill/<run_id>/summary.json` contains per-task success rates for all three tasks and `average_success_rate`
- `rollouts/maniskill/<run_id>/frames/` stores raw frame sequences sufficient for later MP4 rebaking
- `rollouts/maniskill/<run_id>/videos/` contains up to 2 success and up to 2 failure exemplar videos per task when such episodes exist

### Must Have
- Exact task set: `PickCube-v1`, `PushCube-v1`, `StackCube-v1`
- Fixed default launcher GPU policy: physical GPU 3 via launcher environment setup
- Zero-arg user entrypoint
- Smoke-first execution before full benchmark
- Raw frame retention for rebaking later
- Summary JSON with explicit assumptions

### Must NOT Have
- No FailSafe recovery model work
- No generalized benchmark framework beyond the 3 target tasks
- No formal pytest/CI setup in this phase
- No dependence on `vla-evaluation-harness` for the default path
- No silent fallback to CPU for the full benchmark
- No Git tracking of `informations/`

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: none for formal framework; runner-level smoke checks + scripted validations
- QA policy: Every task includes happy-path and failure-path agent-executed scenarios
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.

Wave 1: contract/defaults, preflight/runtime estimator, artifact pipeline

Wave 2: core runner, rebake utility, zero-arg launcher + docs

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 2-6
- Task 2 blocks Tasks 4 and 6
- Task 3 blocks Tasks 4 and 5
- Task 4 blocks Task 6
- Task 5 is blocked by Task 3 and can run in parallel with Task 4
- Task 6 depends on Tasks 2, 4, 5

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 3 tasks → `quick`, `unspecified-low`
- Wave 2 → 3 tasks → `unspecified-high`, `quick`
- Final Verification → 4 tasks → `oracle`, `unspecified-high`, `deep`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Lock benchmark contract, assumptions, and ignore rules

  **What to do**: Add a dedicated ManiSkill benchmark config/defaults module and contract notes that hard-code this phase to `PickCube-v1`, `PushCube-v1`, and `StackCube-v1`; define default smoke and full-run episode counts; define canonical checkpoint path/source rules; define summary JSON schema; define exemplar-video policy (`max 2 success + max 2 failure per task`); define raw-frame retention layout; and update `.gitignore` to ignore `informations/`. The assumption ledger must explicitly mark paper-unknowns: exact ManiSkill version, exact eval episode count in FailSafe, exact seed set, and exact checkpoint provenance.
  **Must NOT do**: Do not add generalized benchmark registries for arbitrary ManiSkill suites; do not silently treat base `openvla/openvla-7b` as paper-equivalent without labeling it as an assumption; do not place artifacts under `informations/`.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: concentrated config and repo-hygiene changes with limited surface area
  - Skills: `[]` — No special skill needed
  - Omitted: `['git-master']` — No commit operation in this task itself

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [2, 3, 4, 5, 6] | Blocked By: []

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `experiments/robot/libero/run_libero_eval.py:54-86` — canonical draccus dataclass evaluation defaults and seed handling
  - Pattern: `experiments/robot/libero/run_libero_eval.py:119-126` — local log file initialization convention
  - Pattern: `experiments/robot/libero/libero_utils.py:61-74` — rollout MP4 naming and save convention
  - Pattern: `.gitignore:147-153` — existing ignored artifact roots (`data/`, `rollouts/`, `wandb/`) to extend with `informations/`
  - Evidence: `README.md:583-593` — OpenVLA eval docs pin package versions and note GPU nondeterminism
  - Evidence: `informations/failsafe.pdf` — paper-aligned target tasks and metric, but with unspecified seeds/version counts

  **Acceptance Criteria** (agent-executable only):
  - [ ] A single benchmark defaults/config module exists under `experiments/robot/maniskill/` and defines exactly the 3 supported tasks, smoke defaults, full-run defaults, artifact root, checkpoint policy, seed policy, and exemplar-video limits.
  - [ ] `.gitignore` contains an explicit `informations/` entry.
  - [ ] A machine-readable contract/assumption artifact path is defined and referenced by downstream scripts.
  - [ ] Default full-run assumption values are documented in code comments and output schema, not hidden.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```
  Scenario: Contract defaults are locked correctly
    Tool: Bash
    Steps: python - <<'PY'
from experiments.robot.maniskill.defaults import TASK_IDS, EXEMPLAR_LIMITS, DEFAULT_GPU_INDEX
assert TASK_IDS == ["PickCube-v1", "PushCube-v1", "StackCube-v1"]
assert EXEMPLAR_LIMITS == {"success": 2, "failure": 2}
assert DEFAULT_GPU_INDEX == 3
print("DEFAULTS_OK")
PY
    Expected: Command prints `DEFAULTS_OK` and exits 0
    Evidence: .sisyphus/evidence/task-1-contract.txt

  Scenario: informations is git-ignored
    Tool: Bash
    Steps: git check-ignore -v informations/failsafe.pdf
    Expected: Output shows `.gitignore` rule for `informations/`
    Evidence: .sisyphus/evidence/task-1-ignore.txt
  ```

  **Commit**: YES | Message: `chore(maniskill): add benchmark defaults and artifact contract` | Files: `.gitignore`, `experiments/robot/maniskill/*`

- [x] 2. Add cluster preflight checks and runtime estimator

  **What to do**: Implement a setup checker and runtime estimator that validate conda/env dependencies, ManiSkill importability, renderer availability, ffmpeg/video writing availability, checkpoint existence, disk budget for raw frames, and GPU-3 visibility before full evaluation. The estimator must run a tiny probe on all three tasks, measure episode time, and write JSON with smoke/full estimated durations plus storage estimates for frame retention.
  **Must NOT do**: Do not start the full benchmark inside the estimator; do not auto-fallback to CPU if GPU 3 is missing; do not produce runtime estimates without writing them to a structured JSON file.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` — Reason: medium-complexity scripting and environment validation
  - Skills: `[]`
  - Omitted: `['playwright']` — no browser work

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [4, 6] | Blocked By: [1]

  **References**:
  - Pattern: `README.md:517-529` — external environment installation pattern for simulation evaluation dependencies
  - Pattern: `README.md:590-593` — pinned Python/PyTorch/transformers/flash-attn versions for reproducibility notes
  - Pattern: `experiments/robot/robot_utils.py:29-37` — seed control and reproducibility helpers
  - Pattern: `experiments/robot/openvla_utils.py:31-72` — checkpoint loading path and dataset statistics warning behavior
  - External: `informations/server_usage.pdf` — cluster GPU context motivating preflight and duration reporting

  **Acceptance Criteria**:
  - [ ] `python experiments/robot/maniskill/check_setup.py` exits 0 and prints `SETUP_OK` on a valid cluster node.
  - [ ] The setup checker exits non-zero with precise messages for: missing GPU 3, missing checkpoint, missing ManiSkill dependency, and insufficient free disk for configured raw-frame retention.
  - [ ] `python experiments/robot/maniskill/estimate_runtime.py` writes `rollouts/maniskill/runtime_estimate.json` with `estimated_total_seconds`, `estimated_storage_bytes`, `per_task_estimates`, and `assumptions`.
  - [ ] The estimator uses measured probe timings rather than hard-coded constants alone.

  **QA Scenarios**:
  ```
  Scenario: Valid setup passes preflight
    Tool: Bash
    Steps: python experiments/robot/maniskill/check_setup.py
    Expected: Prints `SETUP_OK` and exits 0
    Evidence: .sisyphus/evidence/task-2-setup-pass.txt

  Scenario: Missing checkpoint fails clearly
    Tool: Bash
    Steps: OPENVLA_MANISKILL_CHECKPOINT=/tmp/does-not-exist python experiments/robot/maniskill/check_setup.py
    Expected: Exits non-zero and prints `CHECKPOINT_MISSING:` followed by the resolved path
    Evidence: .sisyphus/evidence/task-2-setup-missing-checkpoint.txt
  ```

  **Commit**: YES | Message: `feat(maniskill): add setup checks and runtime estimator` | Files: `experiments/robot/maniskill/check_setup.py`, `experiments/robot/maniskill/estimate_runtime.py`, supporting helpers

- [x] 3. Build ManiSkill artifact pipeline for raw frames, metadata, and exemplars

  **What to do**: Implement artifact helpers that create `rollouts/maniskill/<run_id>/` with `summary.json`, `manifest.json`, `episodes.jsonl`, `frames/<task>/<episode>/frame_XXXXXX.png`, and `videos/<task>/`. Capture enough per-episode metadata to regenerate videos later, including task id, episode index, success flag, seed, checkpoint id/path, timing, and exact frame directory. Exemplar selection must be deterministic: first 2 successful episodes and first 2 failed episodes per task in episode order. If a class is absent, record zero count and skip MP4 generation for that class.
  **Must NOT do**: Do not store raw artifacts under `informations/`; do not save every rollout as an MP4 by default; do not make exemplar selection random.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` — Reason: focused utility work with file-layout and serialization decisions
  - Skills: `[]`
  - Omitted: `['git-master']` — not a git-specific task

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [4, 5] | Blocked By: [1]

  **References**:
  - Pattern: `experiments/robot/libero/libero_utils.py:61-74` — video save function and naming pattern
  - Pattern: `experiments/robot/libero/run_libero_eval.py:243-255` — episode-level save + progress logging timing
  - Pattern: `experiments/robot/robot_utils.py:17-20` — date/date-time naming helpers used elsewhere in eval code
  - Convention: `.gitignore:151-153` — `rollouts/` is already intended as generated output storage

  **Acceptance Criteria**:
  - [ ] Artifact helpers create the full run directory tree and write valid JSON/JSONL records.
  - [ ] Raw frames are sufficient to rebuild exemplar MP4s without rerunning inference.
  - [ ] `summary.json` includes keys: `tasks`, `per_task_success_rate`, `average_success_rate`, `checkpoint`, `maniskill_version`, `seed_config`, `episode_count_per_task`, `artifact_paths`, `assumptions`.
  - [ ] Absent-success or absent-failure cases are represented explicitly in metadata.

  **QA Scenarios**:
  ```
  Scenario: Artifact contract is written correctly
    Tool: Bash
    Steps: python - <<'PY'
from experiments.robot.maniskill.artifacts import create_run_layout, write_summary
run_dir = create_run_layout("contract_test")
write_summary(run_dir, {
  "tasks": ["PickCube-v1", "PushCube-v1", "StackCube-v1"],
  "per_task_success_rate": {"PickCube-v1": 0.0, "PushCube-v1": 0.0, "StackCube-v1": 0.0},
  "average_success_rate": 0.0,
  "checkpoint": "dummy",
  "maniskill_version": "dummy",
  "seed_config": {"full": [7]},
  "episode_count_per_task": 50,
  "artifact_paths": {},
  "assumptions": ["test"]
})
print("ARTIFACTS_OK")
PY
    Expected: Prints `ARTIFACTS_OK`, exits 0, and creates summary/manifest/frame/video directories
    Evidence: .sisyphus/evidence/task-3-artifacts.txt

  Scenario: Missing exemplar class is handled gracefully
    Tool: Bash
    Steps: python - <<'PY'
from experiments.robot.maniskill.artifacts import select_exemplars
episodes = [{"episode_index": i, "success": True} for i in range(3)]
selected = select_exemplars(episodes)
assert len(selected["success"]) == 2
assert selected["failure"] == []
print("EXEMPLARS_OK")
PY
    Expected: Prints `EXEMPLARS_OK` and exits 0
    Evidence: .sisyphus/evidence/task-3-exemplars.txt
  ```

  **Commit**: YES | Message: `feat(maniskill): add artifact capture and exemplar selection` | Files: `experiments/robot/maniskill/artifacts.py`, related helpers

- [x] 4. Implement the ManiSkill benchmark runner using repo evaluation conventions

  **What to do**: Add `experiments/robot/maniskill/run_maniskill_eval.py` and supporting utilities under `experiments/robot/maniskill/` that mirror the structure of the LIBERO evaluator: draccus config, model load via shared OpenVLA helpers, deterministic task loop over the three target tasks, per-episode reset/rollout logic, progress logging, per-task success aggregation, and summary JSON emission. Use launcher-provided GPU policy; use the assumption ledger to stamp unresolved paper fields into outputs. Default full benchmark should be 50 episodes/task with seed `[7]` unless a stronger in-repo source is discovered during implementation.
  **Must NOT do**: Do not broaden supported tasks beyond the three target environments; do not bury hard-coded `cuda:3` logic inside low-level OpenVLA helpers; do not save all episodes as MP4s.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: core benchmark implementation with environment integration and model inference
  - Skills: `[]`
  - Omitted: `['playwright']` — CLI/simulation task only

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [6] | Blocked By: [1, 2, 3]

  **References**:
  - Pattern: `experiments/robot/libero/run_libero_eval.py:54-91` — draccus config + wrapped eval entrypoint
  - Pattern: `experiments/robot/libero/run_libero_eval.py:97-145` — seed/model/processor/log initialization flow
  - Pattern: `experiments/robot/libero/run_libero_eval.py:147-260` — per-task/per-episode evaluation and success logging shape
  - API/Type: `experiments/robot/robot_utils.py:40-72` — shared `get_model`, `get_image_resize_size`, `get_action`
  - API/Type: `experiments/robot/openvla_utils.py:31-78` — model/processor loading expectations
  - Evidence: `README.md:583-593` — 50-trials-per-task precedent and reproducibility package pins
  - Evidence: `informations/failsafe.pdf` — target tasks/metric and paper baseline numbers for comparison in summary output

  **Acceptance Criteria**:
  - [ ] `python experiments/robot/maniskill/run_maniskill_eval.py --mode smoke` completes a smoke pass across all three tasks and writes artifacts.
  - [ ] Full-run mode writes `summary.json` with all three task rates and `average_success_rate`.
  - [ ] The runner writes an assumption ledger into output metadata covering checkpoint choice, ManiSkill version, seed policy, and episode-count policy.
  - [ ] Progress logging follows repo convention with local text logs under `./experiments/logs`.

  **QA Scenarios**:
  ```
  Scenario: Smoke benchmark runs end-to-end
    Tool: Bash
    Steps: python experiments/robot/maniskill/run_maniskill_eval.py --mode smoke
    Expected: Exits 0 and writes a run directory with summary.json plus at least one episode record per target task
    Evidence: .sisyphus/evidence/task-4-smoke.txt

  Scenario: Invalid task id is rejected
    Tool: Bash
    Steps: python experiments/robot/maniskill/run_maniskill_eval.py --mode smoke --task_ids PickCube-v1,InvalidTask
    Expected: Exits non-zero and prints `UNSUPPORTED_TASK:` with the invalid id
    Evidence: .sisyphus/evidence/task-4-invalid-task.txt
  ```

  **Commit**: YES | Message: `feat(maniskill): add benchmark runner` | Files: `experiments/robot/maniskill/run_maniskill_eval.py`, `experiments/robot/maniskill/maniskill_utils.py`, supporting modules

- [x] 5. Add video rebake utility from saved raw frames

  **What to do**: Implement a deterministic rebake script that reconstructs MP4s from saved frame directories and metadata without rerunning model inference. It must support rebaking the default exemplar set and a targeted task/episode selection mode for future use, while keeping the default user path zero-arg through the main launcher. The tool should read `episodes.jsonl`/`manifest.json`, locate frame folders, and render MP4s into `videos/` with stable naming.
  **Must NOT do**: Do not require the benchmark to rerun to regenerate videos; do not make video selection depend on manual browsing; do not overwrite existing exemplar videos unless explicitly requested by code-level flag.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: bounded utility built on top of the artifact contract
  - Skills: `[]`
  - Omitted: `['playwright']` — no browser/UI work

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [6] | Blocked By: [1, 3]

  **References**:
  - Pattern: `experiments/robot/libero/libero_utils.py:61-74` — imageio MP4 writing pattern
  - Pattern: `experiments/robot/libero/run_libero_eval.py:243-246` — episode video save timing in the eval lifecycle
  - Contract: Task 3 artifact schema — rebake must consume the exact saved frame and metadata layout

  **Acceptance Criteria**:
  - [ ] `python experiments/robot/maniskill/rebake_videos.py --run_dir <path>` regenerates exemplar MP4s from saved frames with no model load.
  - [ ] Targeted rebake mode can regenerate a specific task/episode pair from metadata alone.
  - [ ] The rebake utility exits non-zero with a clear error if frame directories referenced in metadata are missing.

  **QA Scenarios**:
  ```
  Scenario: Rebuild exemplar videos from saved frames
    Tool: Bash
    Steps: python - <<'PY'
from pathlib import Path
from PIL import Image
import json
run_dir = Path('rollouts/maniskill/rebake_fixture')
(run_dir / 'frames' / 'PickCube-v1' / 'episode_0001').mkdir(parents=True, exist_ok=True)
(run_dir / 'videos' / 'PickCube-v1').mkdir(parents=True, exist_ok=True)
for i in range(3):
    Image.new('RGB', (32, 32), (i * 40, 0, 0)).save(run_dir / 'frames' / 'PickCube-v1' / 'episode_0001' / f'frame_{i:06d}.png')
with open(run_dir / 'episodes.jsonl', 'w') as f:
    f.write(json.dumps({"task_id": "PickCube-v1", "episode_index": 1, "success": True, "frame_dir": str(run_dir / 'frames' / 'PickCube-v1' / 'episode_0001')}) + '\n')
with open(run_dir / 'manifest.json', 'w') as f:
    json.dump({"run_id": "rebake_fixture"}, f)
print(run_dir)
PY
    python experiments/robot/maniskill/rebake_videos.py --run_dir rollouts/maniskill/rebake_fixture
    Expected: Exits 0 and creates/refreshes MP4s under `rollouts/maniskill/rebake_fixture/videos/`
    Evidence: .sisyphus/evidence/task-5-rebake.txt

  Scenario: Missing frame directory fails loudly
    Tool: Bash
    Steps: python experiments/robot/maniskill/rebake_videos.py --run_dir /tmp/nonexistent-run
    Expected: Exits non-zero and prints `FRAME_DIR_MISSING:` or `RUN_DIR_MISSING:`
    Evidence: .sisyphus/evidence/task-5-missing-frames.txt
  ```

  **Commit**: YES | Message: `feat(maniskill): add video rebake utility` | Files: `experiments/robot/maniskill/rebake_videos.py`, related helpers

- [x] 6. Add zero-argument cluster launcher and execution notes

  **What to do**: Create a single cluster launcher script that activates the intended conda env, exports `CUDA_VISIBLE_DEVICES=3`, runs setup checks, runs the runtime estimator, performs the smoke pass, then runs the full benchmark, and finally prints the summary path and headline rates. Keep the user entrypoint zero-arg. Include concise repo-local execution notes describing required checkpoint placement/path conventions, dependency install steps on the cluster, and how to rebake videos after a run.
  **Must NOT do**: Do not require the user to pass GPU/task/checkpoint args for the default path; do not silently continue after failed preflight; do not place execution instructions in `informations/`; do not make GPU-failure QA impossible to trigger.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: bounded shell orchestration and minimal documentation update
  - Skills: `[]`
  - Omitted: `['git-master']` — commit handled after implementation wave, not inside task logic

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [F1, F2, F3, F4] | Blocked By: [1, 2, 4, 5]

  **References**:
  - Pattern: `README.md:517-529` — repo style for environment setup notes
  - Pattern: `README.md:551-580` — explicit runnable evaluation command examples
  - Pattern: `experiments/robot/libero/run_libero_eval.py:119-126` — log file convention the launcher should surface to the user
  - Constraint: `experiments/robot/robot_utils.py:19` and `experiments/robot/openvla_utils.py:21` default to `cuda:0`; the launcher must enforce GPU-3 visibility from the outside instead of editing shared low-level helpers for all eval flows

  **Acceptance Criteria**:
  - [ ] `bash cluster/run_openvla_maniskill_benchmark.sh` is the documented zero-arg entrypoint.
  - [ ] The launcher aborts immediately if setup check or runtime estimation fails.
  - [ ] The launcher prints the resolved run directory, summary path, and average success rate at completion.
  - [ ] Repo-local notes explain checkpoint placement, dependency install, benchmark launch, and video rebake commands.
  - [ ] The launcher exposes one explicit QA-only override (for example `OPENVLA_MANISKILL_VISIBLE_DEVICES_OVERRIDE`) so GPU-availability failure can be tested deterministically without changing the default zero-arg behavior.

  **QA Scenarios**:
  ```
  Scenario: Zero-arg launcher orchestrates the full flow
    Tool: Bash
    Steps: bash cluster/run_openvla_maniskill_benchmark.sh
    Expected: Runs setup -> estimate -> smoke -> full benchmark in order and exits 0 on a valid node
    Evidence: .sisyphus/evidence/task-6-launcher.txt

  Scenario: Missing GPU 3 aborts before smoke run
    Tool: Bash
    Steps: OPENVLA_MANISKILL_VISIBLE_DEVICES_OVERRIDE='' bash cluster/run_openvla_maniskill_benchmark.sh
    Expected: Exits non-zero before benchmark execution and prints `GPU_3_UNAVAILABLE:` or equivalent setup error
    Evidence: .sisyphus/evidence/task-6-no-gpu.txt
  ```

  **Commit**: YES | Message: `feat(maniskill): add cluster launcher and usage notes` | Files: `cluster/run_openvla_maniskill_benchmark.sh`, repo-local usage documentation

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [x] F1. Plan Compliance Audit — oracle

  **What to do**: Compare the finished implementation against this plan task-by-task and verify that all required deliverables, assumptions, artifact contracts, and launcher behaviors are present.
  **Parallelization**: Can Parallel: YES | Final Wave | Blocks: [] | Blocked By: [1, 2, 3, 4, 5, 6]
  **Acceptance Criteria**:
  - [ ] Oracle confirms every required deliverable exists or returns a concrete defect list.
  - [ ] Oracle verifies the implementation still targets only the 3 ManiSkill tasks.
  **QA Scenarios**:
  ```
  Scenario: Plan compliance review
    Tool: task(oracle)
    Steps: Review code and artifacts against `.sisyphus/plans/openvla-maniskill-baseline-cluster.md`; enumerate deviations or respond `APPROVED`
    Expected: Returns `APPROVED` or a concrete actionable defect list
    Evidence: .sisyphus/evidence/f1-plan-compliance.md
  ```

- [x] F2. Code Quality Review — unspecified-high

  **What to do**: Review implementation quality, failure handling, naming consistency, and maintainability of the new ManiSkill benchmark code.
  **Parallelization**: Can Parallel: YES | Final Wave | Blocks: [] | Blocked By: [1, 2, 3, 4, 5, 6]
  **Acceptance Criteria**:
  - [ ] Reviewer approves code structure or returns a concrete defect list.
  - [ ] Any defect list cites exact files and failure modes.
  **QA Scenarios**:
  ```
  Scenario: Code quality review
    Tool: task(unspecified-high)
    Steps: Review the new ManiSkill benchmark implementation for robustness, duplication, error handling, and config clarity; respond `APPROVED` or list defects with file paths
    Expected: Returns `APPROVED` or a concrete actionable defect list
    Evidence: .sisyphus/evidence/f2-code-quality.md
  ```

- [x] F3. Real Manual QA — unspecified-high (+ playwright if UI)

  **What to do**: Execute the delivered commands on the target environment and verify that setup, estimation, smoke, full benchmark, and rebake behaviors match the plan.
  **Parallelization**: Can Parallel: YES | Final Wave | Blocks: [] | Blocked By: [1, 2, 3, 4, 5, 6]
  **Acceptance Criteria**:
  - [ ] Reviewer confirms command-level behavior matches plan outputs.
  - [ ] Reviewer captures exact pass/fail evidence paths for setup, runtime estimate, full run, and rebake.
  **QA Scenarios**:
  ```
  Scenario: End-to-end execution review
    Tool: task(unspecified-high)
    Steps: Run the documented benchmark commands and inspect produced artifacts/logs; respond `APPROVED` or list mismatches with exact commands and outputs
    Expected: Returns `APPROVED` or a concrete actionable defect list
    Evidence: .sisyphus/evidence/f3-manual-qa.md
  ```

- [x] F4. Scope Fidelity Check — deep

  **What to do**: Audit whether the delivered work stayed within benchmark-only scope and avoided accidental drift into FailSafe recovery-model work, generalized harnessing, or test-infra setup.
  **Parallelization**: Can Parallel: YES | Final Wave | Blocks: [] | Blocked By: [1, 2, 3, 4, 5, 6]
  **Acceptance Criteria**:
  - [ ] Reviewer confirms no out-of-scope systems were added.
  - [ ] Reviewer flags any unnecessary framework or unrelated cleanup additions.
  **QA Scenarios**:
  ```
  Scenario: Scope fidelity review
    Tool: task(deep)
    Steps: Compare delivered changes against the benchmark-only scope in the plan; respond `APPROVED` or list out-of-scope additions
    Expected: Returns `APPROVED` or a concrete actionable defect list
    Evidence: .sisyphus/evidence/f4-scope-fidelity.md
  ```

## Commit Strategy
- Commit 1: `chore(maniskill): add benchmark defaults and artifact contract`
- Commit 2: `feat(maniskill): add setup checks and runtime estimator`
- Commit 3: `feat(maniskill): add benchmark runner and artifact capture`
- Commit 4: `feat(maniskill): add rebake and cluster launcher`

## Success Criteria
- One command on the cluster performs smoke validation then full benchmark with no extra args
- Outputs are sufficient to report score immediately and regenerate videos later
- Assumptions are explicit anywhere the paper is underspecified
- Runtime estimate is produced before the heavy run begins
