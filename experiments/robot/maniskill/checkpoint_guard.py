from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, cast

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
)

HFContractValidator = Callable[[str | Path], dict[str, Any]]
CheckpointSourceType = Literal["hf_id", "local_dir", "local_file"]
CheckpointSourceInput = Literal["auto", "hf_id", "local_dir", "local_file"]


@dataclass(frozen=True)
class CheckpointValidationResult:
    source_type: CheckpointSourceType
    checkpoint_reference: str
    run_dir: Path | None
    checkpoint_file: Path | None


class CheckpointValidationError(ValueError):
    tag: str
    message: str

    def __init__(self, tag: str, message: str):
        super().__init__(message)
        self.tag = tag
        self.message = message


def resolve_checkpoint_target(checkpoint_override: str | None, fallback_reference: str) -> str:
    override = (checkpoint_override or "").strip()
    if override:
        return override
    return str(fallback_reference)


def _looks_like_local_reference(reference: str) -> bool:
    if reference.startswith(".") or reference.startswith("~"):
        return True
    if Path(reference).is_absolute():
        return True
    return reference.count("/") != 1


def _classify_source(reference: str) -> CheckpointSourceType:
    reference_path = Path(reference).expanduser()
    if reference_path.exists():
        if reference_path.is_dir():
            return "local_dir"
        return "local_file"
    if _looks_like_local_reference(reference):
        return "local_dir"
    return "hf_id"


def _validate_local_checkpoint_layout(checkpoint_target: str, source_type: CheckpointSourceType) -> CheckpointValidationResult:
    checkpoint_path = Path(checkpoint_target).expanduser()
    if not checkpoint_path.exists():
        raise CheckpointValidationError("CHECKPOINT_MISSING", f"Path does not exist: `{checkpoint_path}`")

    if source_type == "local_file":
        if not checkpoint_path.is_file():
            raise CheckpointValidationError("CHECKPOINT_MISSING", f"Expected file checkpoint path, got `{checkpoint_path}`")
        if checkpoint_path.suffix != ".pt" or checkpoint_path.parent.name != "checkpoints":
            raise CheckpointValidationError(
                "CHECKPOINT_MISSING",
                f"Expected a `.pt` file under a `checkpoints/` directory but got `{checkpoint_path}`.",
            )
        run_dir = checkpoint_path.parents[1]
    else:
        if not checkpoint_path.is_dir():
            raise CheckpointValidationError("CHECKPOINT_MISSING", f"Expected directory checkpoint path, got `{checkpoint_path}`")
        run_dir = checkpoint_path
        checkpoint_path = run_dir / "checkpoints" / "latest-checkpoint.pt"
        if not checkpoint_path.exists():
            raise CheckpointValidationError(
                "CHECKPOINT_MISSING",
                f"Missing `checkpoints/latest-checkpoint.pt` under `{run_dir}`.",
            )

    config_json = run_dir / "config.json"
    dataset_statistics_json = run_dir / "dataset_statistics.json"
    if not config_json.exists():
        raise CheckpointValidationError("CHECKPOINT_MISSING", f"Missing `config.json` for `{run_dir}`")
    if not dataset_statistics_json.exists():
        raise CheckpointValidationError("CHECKPOINT_MISSING", f"Missing `dataset_statistics.json` for `{run_dir}`")

    return CheckpointValidationResult(
        source_type=source_type,
        checkpoint_reference=checkpoint_target,
        run_dir=run_dir,
        checkpoint_file=checkpoint_path,
    )


def _load_hf_contract_validator() -> HFContractValidator:
    try:
        module = importlib.import_module("experiments.robot.openvla_utils")
        return getattr(module, "validate_hf_checkpoint_contract")
    except Exception:
        openvla_utils_path = Path(__file__).resolve().parents[1] / "openvla_utils.py"
        source = openvla_utils_path.read_text()
        tree = ast.parse(source)
        selected_nodes: list[ast.stmt] = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_looks_like_local_checkpoint", "validate_hf_checkpoint_contract"}
        ]
        if len(selected_nodes) != 2:
            raise CheckpointValidationError(
                "HF_CHECKPOINT_CONTRACT_UNAVAILABLE",
                f"Unable to load HF checkpoint contract validator from `{openvla_utils_path}`.",
            )
        module_ast = ast.Module(body=selected_nodes, type_ignores=[])
        namespace: dict[str, Any] = {}
        exec(
            "import os\nfrom pathlib import Path\nfrom typing import Any\n"
            "from huggingface_hub import HfApi\n"
            "from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError",
            namespace,
        )
        exec(compile(module_ast, str(openvla_utils_path), "exec"), namespace)
        validator_any = namespace.get("validate_hf_checkpoint_contract")
        if not callable(validator_any):
            raise CheckpointValidationError(
                "HF_CHECKPOINT_CONTRACT_UNAVAILABLE",
                f"HF checkpoint contract validator missing in `{openvla_utils_path}`.",
            )
        return cast(HFContractValidator, validator_any)


def _validate_hf_checkpoint(checkpoint_id: str, require_dataset_statistics: bool) -> CheckpointValidationResult:
    validate_hf_checkpoint_contract = _load_hf_contract_validator()
    try:
        _ = validate_hf_checkpoint_contract(checkpoint_id)
    except CheckpointValidationError:
        raise
    except ValueError as exc:
        message = str(exc)
        if "failed to inspect HF model repo" in message:
            tag = "HF_CHECKPOINT_NETWORK_UNAVAILABLE"
        elif message.startswith("HF_CHECKPOINT_CONTRACT_INVALID"):
            tag = "HF_CHECKPOINT_CONTRACT_INVALID"
        else:
            tag = "HF_CHECKPOINT_INVALID"
        raise CheckpointValidationError(tag, message) from exc
    except OfflineModeIsEnabled as exc:
        raise CheckpointValidationError(
            "HF_CHECKPOINT_NETWORK_UNAVAILABLE",
            f"Unable to validate HF checkpoint contract for `{checkpoint_id}`: {exc}",
        ) from exc
    except Exception as exc:
        raise CheckpointValidationError(
            "HF_CHECKPOINT_CONTRACT_UNAVAILABLE",
            f"Unable to validate HF checkpoint contract for `{checkpoint_id}`: {exc}",
        ) from exc

    if require_dataset_statistics:
        try:
            _ = hf_hub_download(
                repo_id=checkpoint_id,
                filename="dataset_statistics.json",
                repo_type="model",
            )
        except LocalEntryNotFoundError as exc:
            raise CheckpointValidationError(
                "HF_CHECKPOINT_CACHE_MISS",
                f"HF checkpoint `{checkpoint_id}` is not available in local cache and could not be resolved from remote.",
            ) from exc
        except EntryNotFoundError as exc:
            raise CheckpointValidationError(
                "HF_CHECKPOINT_DATASET_STATS_MISSING",
                f"HF checkpoint `{checkpoint_id}` is missing required `dataset_statistics.json`.",
            ) from exc
        except RepositoryNotFoundError as exc:
            raise CheckpointValidationError(
                "HF_CHECKPOINT_CONTRACT_INVALID",
                f"HF model repo `{checkpoint_id}` was not found.",
            ) from exc
        except HfHubHTTPError as exc:
            raise CheckpointValidationError(
                "HF_CHECKPOINT_NETWORK_UNAVAILABLE",
                f"Unable to fetch `dataset_statistics.json` for `{checkpoint_id}`: {exc}",
            ) from exc
        except Exception as exc:
            raise CheckpointValidationError(
                "HF_CHECKPOINT_DATASET_STATS_UNAVAILABLE",
                f"Unable to resolve `dataset_statistics.json` for `{checkpoint_id}`: {exc}",
            ) from exc

    return CheckpointValidationResult(
        source_type="hf_id",
        checkpoint_reference=checkpoint_id,
        run_dir=None,
        checkpoint_file=None,
    )


def validate_checkpoint_reference(
    checkpoint_reference: str,
    source_type: CheckpointSourceInput = "auto",
    require_dataset_statistics: bool = True,
) -> CheckpointValidationResult:
    normalized_reference = checkpoint_reference.strip()
    if not normalized_reference:
        raise CheckpointValidationError("CHECKPOINT_MISSING", "Checkpoint reference is empty.")

    if source_type == "auto":
        resolved_source = _classify_source(normalized_reference)
    else:
        resolved_source = source_type

    if resolved_source == "hf_id":
        return _validate_hf_checkpoint(normalized_reference, require_dataset_statistics=require_dataset_statistics)

    return _validate_local_checkpoint_layout(normalized_reference, source_type=resolved_source)
