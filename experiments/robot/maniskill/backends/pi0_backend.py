from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.robot.maniskill.backends.base import (
    ManiSkillBackendMetadata,
    canonicalize_action,
)

DEFAULT_OPENPI_CONDA_ENV = "openpi"
DEFAULT_OPENPI_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
DEFAULT_OPENPI_POLICY_SERVER_URL = "http://127.0.0.1:8001"


@dataclass(frozen=True)
class Pi0RuntimeConfig:
    host: str
    port: int
    checkpoint: str


def _resolve_runtime_config(cfg: Any) -> Pi0RuntimeConfig:
    url = str(getattr(cfg, "pi0_policy_server_url", None)
              or getattr(cfg, "policy_server_url", None)
              or os.getenv("OPENPI_POLICY_SERVER_URL", DEFAULT_OPENPI_POLICY_SERVER_URL)).strip()
    # url: http://127.0.0.1:8001
    host = url.split("//")[-1].split(":")[0]
    port = int(url.split(":")[-1])
    checkpoint = str(
        getattr(cfg, "openpi_checkpoint", None)
        or getattr(cfg, "pi0_checkpoint", None)
        or getattr(cfg, "pretrained_checkpoint", None)
        or os.getenv("OPENPI_CHECKPOINT", DEFAULT_OPENPI_CHECKPOINT)
    ).strip()
    return Pi0RuntimeConfig(host=host, port=port, checkpoint=checkpoint)


class Pi0Backend:
    metadata = ManiSkillBackendMetadata(
        backend_id="maniskill.pi0",
        model_family="pi0",
    )

    def __init__(self) -> None:
        self._runtime: Pi0RuntimeConfig | None = None
        self._client = None
        self._cached_chunk: list[np.ndarray] = []
        self._active_task_label: str | None = None

    def _reset_chunk_cache(self) -> None:
        self._cached_chunk = []

    def reset_rollout_state(self, *, task_label: str | None = None, reason: str = "") -> None:
        self._active_task_label = task_label
        self._reset_chunk_cache()
        if self._client is not None:
            try:
                self._client.reset()
            except Exception:
                pass

    def get_model(self, cfg: Any):
        runtime = _resolve_runtime_config(cfg)
        self._runtime = runtime
        self._reset_chunk_cache()
        self._active_task_label = None
        self._client = None  # 연결 안 함, get_action에서 lazy 연결
        return runtime  # client 대신 runtime 반환

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
        import pathlib
        pathlib.Path("/home/jwchoi84/openvla/env_debug.log").open("a").write(
            "\n".join(f"{k}={v}" for k, v in os.environ.items() if any(x in k.lower() for x in ["proxy", "http", "ws", "cuda", "openpi"]))
            + "\n---\n"
        )
        
        cached = self._next_cached_action(task_label)
        if cached is not None:
            return cached

        if self._client is None:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            runtime = self._runtime or _resolve_runtime_config(cfg)
            self._client = WebsocketClientPolicy(host=runtime.host, port=runtime.port)

        img = np.asarray(obs["full_image"], dtype=np.uint8)
        wrist_raw = obs.get("wrist_image", np.zeros((224, 224, 3), dtype=np.uint8))
        wrist_img = np.asarray(wrist_raw, dtype=np.uint8)

        state = np.asarray(obs.get("state", np.zeros(8)), dtype=np.float32)

        obs_payload = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": str(task_label),
        }

        state = np.asarray(obs.get("state", np.zeros(8)), dtype=np.float32)

        obs_payload = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": str(task_label),
        }

        REPLAN_STEPS = 5  # openpi 예제랑 맞춤

        result = client.infer(obs_payload)
        actions = np.asarray(result["actions"], dtype=np.float32)

        # 디버그 로그
        import pathlib
        pathlib.Path("/home/jwchoi84/openvla/pi0_debug.log").open("a").write(
            f"actions[0]={actions[0].tolist()}, shape={list(actions.shape)}\n"
        )

        chunk = [actions[i] for i in range(min(REPLAN_STEPS, len(actions)))]
        if len(chunk) > 1:
            self._cached_chunk.extend(chunk[1:])
        return canonicalize_action(chunk[0], backend_id=self.metadata.backend_id)