TASK_IDS = ["PickCube-v1", "PushCube-v1", "StackCube-v1"]

DEFAULT_GPU_INDEX_ENV_KEY = "OPENVLA_MANISKILL_GPU_INDEX"
DEFAULT_GPU_INDEX = None
EXEMPLAR_LIMITS = {"success": 2, "failure": 2}
ARTIFACT_ROOT = "rollouts/maniskill"

SMOKE_EPISODE_COUNT_PER_TASK = 2
FULL_EPISODE_COUNT_PER_TASK = 50
EPISODE_COUNT_DEFAULTS = {
    "smoke": SMOKE_EPISODE_COUNT_PER_TASK,
    "full": FULL_EPISODE_COUNT_PER_TASK,
}

SEED_POLICY = {
    "mode": "fixed_list",
    "smoke": [7],
    "full": [7],
}

CHECKPOINT_POLICY = {
    "selection": "user_provided_or_repo_default",
    "paper_identical_provenance_proven": False,
    "fallback_reference": "Juelg/openvla-7b-finetuned-maniskill",
}

SUMMARY_SCHEMA_KEYS = [
    "tasks",
    "per_task_success_rate",
    "average_success_rate",
    "checkpoint",
    "maniskill_version",
    "seed_config",
    "episode_count_per_task",
    "artifact_paths",
    "assumptions",
]

RAW_FRAME_LAYOUT_CONTRACT = {
    "root": f"{ARTIFACT_ROOT}/frames",
    "task_dir": "{task_id}",
    "seed_dir": "seed_{seed}",
    "episode_dir": "episode_{episode_idx:04d}",
    "frame_filename": "frame_{frame_idx:06d}.png",
    "retention": "retain_all_raw_frames",
}

ASSUMPTION_LEDGER_PATH_TEMPLATE = f"{ARTIFACT_ROOT}/{{run_id}}/assumptions.json"

ASSUMPTIONS = [
    {
        "key": "maniskill_version_unspecified",
        "label": "assumption",
        "value": "Exact ManiSkill version used by FailSafe is unspecified.",
    },
    {
        "key": "failsafe_eval_episode_count_unspecified",
        "label": "assumption",
        "value": "Exact FailSafe evaluation episode count is unspecified.",
    },
    {
        "key": "failsafe_seed_set_unspecified",
        "label": "assumption",
        "value": "Exact FailSafe seed set is unspecified.",
    },
    {
        "key": "checkpoint_provenance_assumed",
        "label": "assumption",
        "value": "Checkpoint provenance for this benchmark is assumed/defaulted and not proven paper-identical.",
    },
]
