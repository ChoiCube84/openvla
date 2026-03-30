from __future__ import annotations

from experiments.robot.maniskill.backends.base import ManiSkillBackendMetadata


def _build_openvla_backend():
    from experiments.robot.maniskill.backends.openvla_backend import OpenVLABackend

    return OpenVLABackend()


def _build_pi0_backend():
    from experiments.robot.maniskill.backends.pi0_backend import Pi0Backend

    return Pi0Backend()


_BACKEND_FACTORIES = {
    "openvla": _build_openvla_backend,
    "pi0": _build_pi0_backend,
}
_BACKENDS = {}


def get_backend(model_family: str):
    normalized_model_family = str(model_family).strip().lower()
    if normalized_model_family not in _BACKEND_FACTORIES:
        supported = ", ".join(sorted(_BACKEND_FACTORIES.keys()))
        raise ValueError(f"INVALID_MODEL_FAMILY: `{model_family}`. Supported model families: {supported}")
    backend = _BACKENDS.get(normalized_model_family)
    if backend is None:
        backend = _BACKEND_FACTORIES[normalized_model_family]()
        _BACKENDS[normalized_model_family] = backend
    return backend


def get_backend_metadata(model_family: str) -> ManiSkillBackendMetadata:
    return get_backend(model_family).metadata
