from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.robot.maniskill.checkpoint_guard import (
    CheckpointValidationError,
    CheckpointValidationResult,
    validate_checkpoint_reference,
)


def fail(tag: str, message: str, code: int = 1) -> NoReturn:
    print(f"{tag}: {message}")
    raise SystemExit(code)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate ManiSkill checkpoint reference for local or HF sources.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint_id", type=str, help="Hugging Face model id (e.g. openvla/openvla-7b).")
    group.add_argument("--checkpoint_dir", type=str, help="Local run directory containing checkpoints/latest-checkpoint.pt.")
    group.add_argument("--checkpoint_file", type=str, help="Local .pt checkpoint under a checkpoints/ directory.")
    parser.add_argument(
        "--skip_dataset_statistics",
        action="store_true",
        help="Skip dataset_statistics.json availability checks.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    return parser


def _result_payload(result: CheckpointValidationResult) -> dict[str, str | None]:
    return {
        "source_type": result.source_type,
        "checkpoint_reference": result.checkpoint_reference,
        "run_dir": str(result.run_dir) if result.run_dir else None,
        "checkpoint_file": str(result.checkpoint_file) if result.checkpoint_file else None,
    }


def main() -> None:
    args = _build_parser().parse_args()

    if args.checkpoint_id:
        source_type = "hf_id"
        checkpoint_reference = args.checkpoint_id
    elif args.checkpoint_dir:
        source_type = "local_dir"
        checkpoint_reference = args.checkpoint_dir
    else:
        source_type = "local_file"
        checkpoint_reference = args.checkpoint_file

    try:
        result = validate_checkpoint_reference(
            checkpoint_reference,
            source_type=source_type,
            require_dataset_statistics=not args.skip_dataset_statistics,
        )
    except CheckpointValidationError as exc:
        fail(exc.tag, exc.message)

    payload = _result_payload(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "CHECKPOINT_VALID: "
            f"source_type={payload['source_type']} checkpoint_reference={payload['checkpoint_reference']}"
        )


if __name__ == "__main__":
    main()
