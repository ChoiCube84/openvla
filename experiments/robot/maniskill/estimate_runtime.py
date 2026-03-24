from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.robot.maniskill.defaults import (
    ARTIFACT_ROOT,
    ASSUMPTIONS,
    FULL_EPISODE_COUNT_PER_TASK,
    RAW_FRAME_LAYOUT_CONTRACT,
    SEED_POLICY,
    SMOKE_EPISODE_COUNT_PER_TASK,
    TASK_IDS,
)


PROBE_RENDER_MODE = "rgb_array"
DEFAULT_STEPS_PER_EPISODE = 200
DEFAULT_FRAME_SHAPE = (480, 640, 3)
DEFAULT_PNG_COMPRESSION_RATIO = 0.35
FALLBACK_STEP_SECONDS = 0.08
FALLBACK_RESET_SECONDS = 0.3


def _timed_import(module_name: str) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        module = importlib.import_module(module_name)
        return {
            "ok": True,
            "seconds": time.perf_counter() - start,
            "module": module,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "seconds": time.perf_counter() - start,
            "module": None,
            "error": str(exc),
        }


def _probe_env() -> Dict[str, Any]:
    mani_skill_import = _timed_import("mani_skill")
    gym_import = _timed_import("gymnasium")

    probe: Dict[str, Any] = {
        "imports": {
            "mani_skill": {"ok": mani_skill_import["ok"], "seconds": mani_skill_import["seconds"], "error": mani_skill_import["error"]},
            "gymnasium": {"ok": gym_import["ok"], "seconds": gym_import["seconds"], "error": gym_import["error"]},
        },
        "env_probes_by_task": {},
        "env_probe": {
            "ok": False,
            "task_id": TASK_IDS[0],
            "render_mode": PROBE_RENDER_MODE,
            "reset_seconds": None,
            "step_seconds": None,
            "frame_bytes": None,
            "error": None,
        },
    }

    if not (mani_skill_import["ok"] and gym_import["ok"]):
        import_error = mani_skill_import["error"] or gym_import["error"] or "import_failure"
        for task_id in TASK_IDS:
            probe["env_probes_by_task"][task_id] = {
                "ok": False,
                "task_id": task_id,
                "render_mode": PROBE_RENDER_MODE,
                "reset_seconds": None,
                "step_seconds": None,
                "frame_bytes": None,
                "error": import_error,
            }
        probe["env_probe"] = dict(probe["env_probes_by_task"][TASK_IDS[0]])
        return probe

    gymnasium = gym_import["module"]
    seed = SEED_POLICY.get("full", [7])[0]
    for task_id in TASK_IDS:
        task_probe: Dict[str, Any] = {
            "ok": False,
            "task_id": task_id,
            "render_mode": PROBE_RENDER_MODE,
            "reset_seconds": None,
            "step_seconds": None,
            "frame_bytes": None,
            "error": None,
        }
        env = None
        try:
            env = gymnasium.make(task_id, render_mode=PROBE_RENDER_MODE)

            reset_start = time.perf_counter()
            env.reset(seed=seed)
            reset_seconds = time.perf_counter() - reset_start

            step_start = time.perf_counter()
            action = env.action_space.sample()
            env.step(action)
            step_seconds = time.perf_counter() - step_start

            frame = env.render()
            frame_bytes = int(getattr(frame, "nbytes", 0))

            task_probe.update(
                {
                    "ok": True,
                    "reset_seconds": reset_seconds,
                    "step_seconds": step_seconds,
                    "frame_bytes": frame_bytes,
                }
            )
        except Exception as exc:
            task_probe["error"] = str(exc)
        finally:
            if env is not None:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

        probe["env_probes_by_task"][task_id] = task_probe

    probe["env_probe"] = dict(probe["env_probes_by_task"].get(TASK_IDS[0], probe["env_probe"]))

    return probe


def _build_assumptions(probe: Dict[str, Any]) -> List[Dict[str, Any]]:
    assumptions: List[Dict[str, Any]] = [dict(item) for item in ASSUMPTIONS]
    assumptions.append(
        {
            "key": "runtime_probe_method",
            "label": "assumption",
            "value": "Estimator uses import + one-env reset/step/render probe when available.",
        }
    )
    assumptions.append(
        {
            "key": "runtime_probe_task",
            "label": "assumption",
            "value": f"Probe tasks are `{', '.join(TASK_IDS)}` with `render_mode={PROBE_RENDER_MODE}`.",
        }
    )

    assumptions.append(
        {
            "key": "runtime_profiles",
            "label": "assumption",
            "value": "Runtime estimate emits explicit `smoke` and `full` profiles; legacy top-level totals remain the full profile.",
        }
    )

    if not any(bool(task_probe.get("ok")) for task_probe in probe.get("env_probes_by_task", {}).values()):
        assumptions.append(
            {
                "key": "runtime_probe_fallback",
                "label": "assumption",
                "value": "Environment probe unavailable; per-step/per-reset timings fall back to conservative constants.",
            }
        )

    return assumptions


def _estimate(probe: Dict[str, Any]) -> Dict[str, Any]:
    default_frame_bytes = DEFAULT_FRAME_SHAPE[0] * DEFAULT_FRAME_SHAPE[1] * DEFAULT_FRAME_SHAPE[2]
    import_seconds = sum(entry["seconds"] for entry in probe["imports"].values())

    env_probes_by_task = probe.get("env_probes_by_task", {})

    def _profile(episode_count_per_task: int, seed_mode_key: str) -> Dict[str, Any]:
        per_task_estimates: Dict[str, Dict[str, Any]] = {}
        for task_id in TASK_IDS:
            task_probe = env_probes_by_task.get(task_id, {})
            task_ok = bool(task_probe.get("ok"))
            step_seconds = task_probe.get("step_seconds") if task_ok else FALLBACK_STEP_SECONDS
            reset_seconds = task_probe.get("reset_seconds") if task_ok else FALLBACK_RESET_SECONDS
            frame_bytes = (
                int(task_probe.get("frame_bytes"))
                if task_ok and task_probe.get("frame_bytes")
                else default_frame_bytes
            )

            per_episode_seconds = float(reset_seconds + (step_seconds * DEFAULT_STEPS_PER_EPISODE))
            per_episode_storage_bytes = int(frame_bytes * DEFAULT_STEPS_PER_EPISODE * DEFAULT_PNG_COMPRESSION_RATIO)
            task_seconds = float(per_episode_seconds * episode_count_per_task)
            task_storage = int(per_episode_storage_bytes * episode_count_per_task)

            per_task_estimates[task_id] = {
                "episodes": episode_count_per_task,
                "estimated_seconds": task_seconds,
                "estimated_storage_bytes": task_storage,
                "steps_per_episode": DEFAULT_STEPS_PER_EPISODE,
                "seed_count": len(SEED_POLICY.get(seed_mode_key, [])) or 1,
                "raw_frame_retention": RAW_FRAME_LAYOUT_CONTRACT.get("retention"),
            }

        estimated_total_seconds = float(import_seconds + sum(item["estimated_seconds"] for item in per_task_estimates.values()))
        estimated_storage_bytes = int(sum(item["estimated_storage_bytes"] for item in per_task_estimates.values()))
        return {
            "estimated_total_seconds": estimated_total_seconds,
            "estimated_storage_bytes": estimated_storage_bytes,
            "per_task_estimates": per_task_estimates,
        }

    smoke_profile = _profile(episode_count_per_task=SMOKE_EPISODE_COUNT_PER_TASK, seed_mode_key="smoke")
    full_profile = _profile(episode_count_per_task=FULL_EPISODE_COUNT_PER_TASK, seed_mode_key="full")

    return {
        "estimated_total_seconds": full_profile["estimated_total_seconds"],
        "estimated_storage_bytes": full_profile["estimated_storage_bytes"],
        "per_task_estimates": full_profile["per_task_estimates"],
        "mode_estimates": {
            "smoke": smoke_profile,
            "full": full_profile,
        },
        "assumptions": _build_assumptions(probe),
        "probe": {
            "imports": probe["imports"],
            "env_probe": probe["env_probe"],
            "env_probes_by_task": probe.get("env_probes_by_task", {}),
        },
    }


def main() -> None:
    probe = _probe_env()
    estimate_payload = _estimate(probe)

    artifact_root = Path(ARTIFACT_ROOT)
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_path = artifact_root / "runtime_estimate.json"
    with output_path.open("w") as f:
        json.dump(estimate_payload, f, indent=2)

    print(str(output_path))


if __name__ == "__main__":
    main()
