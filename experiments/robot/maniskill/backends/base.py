from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np


ACTION_DIM = 7


@dataclass(frozen=True)
class ManiSkillBackendMetadata:
    backend_id: str
    model_family: str
    action_dim: int = ACTION_DIM
    action_dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ManiSkillBackend(Protocol):
    @property
    def metadata(self) -> ManiSkillBackendMetadata:
        ...

    def get_model(self, cfg: Any):
        ...

    def get_processor(self, cfg: Any):
        ...

    def get_image_resize_size(self, cfg: Any) -> int:
        ...

    def reset_rollout_state(self, *, task_label: str | None = None, reason: str = "") -> None:
        ...

    def get_action(self, cfg: Any, model: Any, obs: dict[str, Any], task_label: str, processor: Any = None) -> np.ndarray:
        ...


def canonicalize_action(action: Any, *, backend_id: str) -> np.ndarray:
    canonical = np.asarray(action, dtype=np.float32)
    if canonical.shape != (ACTION_DIM,):
        raise ValueError(
            f"INVALID_BACKEND_ACTION: backend `{backend_id}` produced shape={canonical.shape}, expected={(ACTION_DIM,)}"
        )
    if not np.all(np.isfinite(canonical)):
        raise ValueError(f"INVALID_BACKEND_ACTION: backend `{backend_id}` produced non-finite values.")
    return canonical
