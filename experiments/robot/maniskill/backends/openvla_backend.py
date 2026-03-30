from __future__ import annotations

from typing import Any

from experiments.robot.maniskill.backends.base import (
    ManiSkillBackendMetadata,
    canonicalize_action,
)
from experiments.robot.openvla_utils import get_processor, get_vla, get_vla_action


class OpenVLABackend:
    metadata = ManiSkillBackendMetadata(
        backend_id="maniskill.openvla",
        model_family="openvla",
    )

    def get_model(self, cfg: Any):
        return get_vla(cfg)

    def get_processor(self, cfg: Any):
        return get_processor(cfg)

    def get_image_resize_size(self, cfg: Any) -> int:
        return 224

    def reset_rollout_state(self, *, task_label: str | None = None, reason: str = "") -> None:
        return None

    def get_action(self, cfg: Any, model: Any, obs: dict[str, Any], task_label: str, processor: Any = None):
        action = get_vla_action(
            model,
            processor,
            cfg.pretrained_checkpoint,
            obs,
            task_label,
            cfg.unnorm_key,
            center_crop=cfg.center_crop,
        )
        return canonicalize_action(action, backend_id=self.metadata.backend_id)
