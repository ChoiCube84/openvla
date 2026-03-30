from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request

import numpy as np

from experiments.robot.maniskill.backends.base import (
    ManiSkillBackendMetadata,
    canonicalize_action,
)


DEFAULT_OPENPI_CONDA_ENV = "openpi"
DEFAULT_OPENPI_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
DEFAULT_OPENPI_POLICY_SERVER_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class Pi0RuntimeConfig:
    policy_server_url: str
    openpi_conda_env: str
    openpi_repo_root: str
    checkpoint: str


def _read_cfg_value(cfg: Any, *field_names: str, default: str = "") -> str:
    for field_name in field_names:
        value = getattr(cfg, field_name, None)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return default


def _resolve_runtime_config(cfg: Any) -> Pi0RuntimeConfig:
    policy_server_url = _read_cfg_value(
        cfg,
        "pi0_policy_server_url",
        "policy_server_url",
        default=os.getenv("OPENPI_POLICY_SERVER_URL", DEFAULT_OPENPI_POLICY_SERVER_URL),
    )
    openpi_conda_env = _read_cfg_value(
        cfg,
        "openpi_conda_env",
        default=os.getenv("OPENPI_CONDA_ENV", DEFAULT_OPENPI_CONDA_ENV),
    )
    openpi_repo_root = _read_cfg_value(
        cfg,
        "openpi_repo_root",
        default=os.getenv("OPENPI_REPO_ROOT", ""),
    )
    checkpoint = _read_cfg_value(
        cfg,
        "openpi_checkpoint",
        "pi0_checkpoint",
        "pretrained_checkpoint",
        default=os.getenv("OPENPI_CHECKPOINT", DEFAULT_OPENPI_CHECKPOINT),
    )
    return Pi0RuntimeConfig(
        policy_server_url=policy_server_url,
        openpi_conda_env=openpi_conda_env,
        openpi_repo_root=openpi_repo_root,
        checkpoint=checkpoint,
    )


def _http_json(url: str, payload: dict[str, Any], timeout_s: float = 3.0) -> dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=encoded, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
        if not body.strip():
            return {}
        parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError(f"PI0_SERVER_PROTOCOL_ERROR: expected JSON object response from `{url}`.")
    return parsed


def _extract_chunk(response: dict[str, Any]) -> list[np.ndarray]:
    candidate = None
    for key in ("actions", "action_chunk", "chunk", "action"):
        if key in response:
            candidate = response[key]
            break

    if candidate is None:
        raise ValueError("PI0_SERVER_PROTOCOL_ERROR: policy response missing action payload.")

    if isinstance(candidate, dict):
        candidate = candidate.get("actions") or candidate.get("action")

    if candidate is None:
        raise ValueError("PI0_SERVER_PROTOCOL_ERROR: action payload is empty.")

    if isinstance(candidate, (list, tuple)):
        if not candidate:
            raise ValueError("PI0_SERVER_PROTOCOL_ERROR: action chunk is empty.")

        if np.asarray(candidate).ndim == 1:
            return [np.asarray(candidate, dtype=np.float32)]

        return [np.asarray(step_action, dtype=np.float32) for step_action in candidate]

    return [np.asarray(candidate, dtype=np.float32)]


class Pi0Backend:
    metadata = ManiSkillBackendMetadata(
        backend_id="maniskill.pi0",
        model_family="pi0",
    )

    def __init__(self) -> None:
        self._runtime: Pi0RuntimeConfig | None = None
        self._cached_chunk: list[np.ndarray] = []
        self._active_task_label: str | None = None

    def _reset_chunk_cache(self) -> None:
        self._cached_chunk = []

    def reset_rollout_state(self, *, task_label: str | None = None, reason: str = "") -> None:
        self._active_task_label = task_label
        self._reset_chunk_cache()

    def get_model(self, cfg: Any):
        runtime = _resolve_runtime_config(cfg)
        self._runtime = runtime
        self._reset_chunk_cache()
        self._active_task_label = None

        health_payload = {
            "ping": "pi0",
            "checkpoint": runtime.checkpoint,
            "runtime": {
                "OPENPI_CONDA_ENV": runtime.openpi_conda_env,
                "OPENPI_REPO_ROOT": runtime.openpi_repo_root,
            },
        }
        health_endpoints = ("/healthz", "/health", "/")
        errors: list[str] = []

        for endpoint in health_endpoints:
            health_url = f"{runtime.policy_server_url.rstrip('/')}{endpoint}"
            try:
                response = _http_json(health_url, health_payload)
                status = str(response.get("status", "ok")).strip().lower()
                if status not in ("ok", "healthy", "ready"):
                    raise ValueError(f"unexpected status `{status}`")
                return runtime
            except (error.URLError, error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{health_url}: {exc}")

        details = " | ".join(errors)
        raise RuntimeError(f"PI0_HEALTHCHECK_ERROR: unable to reach a healthy OpenPI policy server. details={details}")

    def get_processor(self, cfg: Any):
        return None

    def get_image_resize_size(self, cfg: Any) -> int:
        return 224

    def _next_cached_action(self, task_label: str) -> np.ndarray | None:
        if self._active_task_label != task_label:
            self._active_task_label = task_label
            self._reset_chunk_cache()
        if not self._cached_chunk:
            return None
        action = self._cached_chunk.pop(0)
        return canonicalize_action(action, backend_id=self.metadata.backend_id)

    def get_action(self, cfg: Any, model: Any, obs: dict[str, Any], task_label: str, processor: Any = None):
        cached = self._next_cached_action(task_label)
        if cached is not None:
            return cached

        runtime = self._runtime if self._runtime is not None else _resolve_runtime_config(cfg)
        image = np.asarray(obs.get("full_image", []), dtype=np.uint8)
        payload = {
            "task": str(task_label),
            "checkpoint": runtime.checkpoint,
            "observation": {
                "full_image": image.tolist(),
                "dtype": str(image.dtype),
                "shape": list(image.shape),
            },
            "runtime": {
                "OPENPI_CONDA_ENV": runtime.openpi_conda_env,
                "OPENPI_REPO_ROOT": runtime.openpi_repo_root,
            },
        }

        infer_url = f"{runtime.policy_server_url.rstrip('/')}/act"
        try:
            response = _http_json(infer_url, payload, timeout_s=8.0)
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"PI0_POLICY_SERVER_ERROR: request to `{infer_url}` failed: {exc}") from exc

        chunk = _extract_chunk(response)
        if len(chunk) > 1:
            self._cached_chunk.extend(chunk[1:])
        return canonicalize_action(chunk[0], backend_id=self.metadata.backend_id)
