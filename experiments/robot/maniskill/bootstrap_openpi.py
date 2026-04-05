from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error
from urllib import request

DEFAULT_OPENPI_CONDA_ENV = "openpi"
DEFAULT_OPENPI_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
DEFAULT_OPENPI_POLICY_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_OPENPI_REPO_URL = "https://github.com/Physical-Intelligence/openpi.git"
DEFAULT_OPENPI_BOOTSTRAP_REF = "fdc03f527881cdfc8ae1a168ed6a20c60edbbbcc"
BOOTSTRAP_MARKER_NAME = ".openvla_openpi_bootstrap.json"
BOOTSTRAP_SCHEMA_VERSION = 2
EXPECTED_REPO_PATHS = (
    "pyproject.toml",
    "scripts/serve_policy.py",
    "src/openpi",
)


class OpenPIRuntimeError(RuntimeError):
    def __init__(self, state: str, message: str) -> None:
        super().__init__(message)
        self.state = state


@dataclass(frozen=True)
class BootstrapMarker:
    schema_version: int
    bootstrap_source_url: str
    bootstrap_ref: str
    git_revision: str
    created_at_unix: float
    expected_repo_paths: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeMetadata:
    status: str
    cache_state: str
    bootstrap_action: str
    repo_root: str
    managed_repo_root: str
    bootstrap_marker_path: str
    bootstrap_source_url: str
    bootstrap_ref: str
    git_revision: str
    policy_server_status: str
    policy_server_url: str
    policy_server_python: str
    policy_server_entrypoint: str
    policy_server_launch_prefix: str
    checkpoint: str
    openpi_conda_env: str


def _cache_parent_root() -> Path:
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        return Path(xdg_cache).expanduser().resolve() / "openvla"
    return Path.home().resolve() / ".cache" / "openvla"


def _managed_repo_root() -> Path:
    return _cache_parent_root() / "openpi"


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or f"git exited with code {result.returncode}"
        raise OpenPIRuntimeError("bootstrap_failed", f"OPENPI_BOOTSTRAP_ERROR: git {' '.join(args)} failed: {stderr}")
    return result.stdout.strip()


def _resolve_head_revision(repo_root: Path) -> str:
    return _run_git(["rev-parse", "HEAD"], cwd=repo_root)


def _validate_checked_out_revision(repo_root: Path, *, expected_git_revision: str) -> str:
    actual_git_revision = _resolve_head_revision(repo_root)
    if actual_git_revision != expected_git_revision:
        raise OpenPIRuntimeError(
            "cache_invalid",
            "OPENPI_CACHE_INVALID: checked-out OpenPI revision "
            f"`{actual_git_revision}` at `{repo_root}` does not match expected `{expected_git_revision}`.",
        )
    return actual_git_revision


def _write_marker(repo_root: Path, marker: BootstrapMarker) -> Path:
    marker_path = repo_root / BOOTSTRAP_MARKER_NAME
    marker_path.write_text(f"{json.dumps(asdict(marker), indent=2, sort_keys=True)}\n")
    return marker_path


def _parse_marker(marker_path: Path) -> BootstrapMarker:
    try:
        payload = json.loads(marker_path.read_text())
    except FileNotFoundError as exc:
        raise OpenPIRuntimeError(
            "cache_invalid",
            f"OPENPI_CACHE_INVALID: bootstrap marker missing at `{marker_path}`.",
        ) from exc
    except json.JSONDecodeError as exc:
        raise OpenPIRuntimeError(
            "cache_invalid",
            f"OPENPI_CACHE_INVALID: bootstrap marker at `{marker_path}` is not valid JSON: {exc}",
        ) from exc

    try:
        schema_version = int(payload["schema_version"])
        bootstrap_source_url = str(payload["bootstrap_source_url"]).strip()
        bootstrap_ref = str(payload.get("bootstrap_ref", "")).strip()
        git_revision = str(payload["git_revision"]).strip()
        created_at_unix = float(payload["created_at_unix"])
        expected_repo_paths = tuple(str(item) for item in payload["expected_repo_paths"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OpenPIRuntimeError(
            "cache_invalid",
            f"OPENPI_CACHE_INVALID: bootstrap marker at `{marker_path}` is malformed: {exc}",
        ) from exc

    return BootstrapMarker(
        schema_version=schema_version,
        bootstrap_source_url=bootstrap_source_url,
        bootstrap_ref=bootstrap_ref,
        git_revision=git_revision,
        created_at_unix=created_at_unix,
        expected_repo_paths=expected_repo_paths,
    )


def _validate_expected_repo_paths(repo_root: Path, expected_paths: tuple[str, ...]) -> None:
    missing = [relative_path for relative_path in expected_paths if not (repo_root / relative_path).exists()]
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise OpenPIRuntimeError(
            "cache_invalid",
            f"OPENPI_CACHE_INVALID: `{repo_root}` is missing expected OpenPI paths: {missing_csv}",
        )


def _validate_repo_root(
    repo_root: Path,
    *,
    bootstrap_source_url: str,
    bootstrap_ref: str = "",
    require_marker: bool,
) -> tuple[BootstrapMarker | None, str]:
    if not repo_root.exists():
        raise OpenPIRuntimeError(
            "cache_missing",
            f"OPENPI_CACHE_MISSING: expected OpenPI checkout at `{repo_root}`.",
        )
    if not repo_root.is_dir():
        raise OpenPIRuntimeError(
            "cache_invalid",
            f"OPENPI_CACHE_INVALID: `{repo_root}` exists but is not a directory.",
        )

    marker_path = repo_root / BOOTSTRAP_MARKER_NAME
    marker: BootstrapMarker | None = None
    if require_marker:
        marker = _parse_marker(marker_path)
        if marker.schema_version != BOOTSTRAP_SCHEMA_VERSION:
            raise OpenPIRuntimeError(
                "cache_stale",
                "OPENPI_CACHE_STALE: managed cache schema version "
                f"`{marker.schema_version}` at `{repo_root}` does not match expected `{BOOTSTRAP_SCHEMA_VERSION}`.",
            )
        if marker.bootstrap_source_url != bootstrap_source_url:
            raise OpenPIRuntimeError(
                "cache_stale",
                "OPENPI_CACHE_STALE: managed cache bootstrap source "
                f"`{marker.bootstrap_source_url}` at `{repo_root}` does not match configured `{bootstrap_source_url}`.",
            )
        if marker.bootstrap_ref != bootstrap_ref:
            raise OpenPIRuntimeError(
                "cache_stale",
                "OPENPI_CACHE_STALE: managed cache bootstrap ref "
                f"`{marker.bootstrap_ref}` at `{repo_root}` does not match configured `{bootstrap_ref}`.",
            )
        _validate_expected_repo_paths(repo_root, marker.expected_repo_paths)
        actual_git_revision = _validate_checked_out_revision(repo_root, expected_git_revision=marker.git_revision)
        return marker, actual_git_revision

    _validate_expected_repo_paths(repo_root, EXPECTED_REPO_PATHS)
    revision = ""
    git_dir = repo_root / ".git"
    if git_dir.exists():
        revision = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
    return None, revision


def _checkout_bootstrap_ref(repo_root: Path, *, bootstrap_ref: str) -> str:
    _run_git(["fetch", "--depth", "1", "origin", bootstrap_ref], cwd=repo_root)
    expected_git_revision = _run_git(["rev-parse", "FETCH_HEAD^{commit}"], cwd=repo_root)
    _run_git(["checkout", "--detach", expected_git_revision], cwd=repo_root)
    _run_git(["submodule", "update", "--init", "--recursive"], cwd=repo_root)
    return _validate_checked_out_revision(repo_root, expected_git_revision=expected_git_revision)


def _bootstrap_managed_repo(repo_root: Path, *, bootstrap_source_url: str, bootstrap_ref: str) -> tuple[BootstrapMarker, str]:
    parent_root = repo_root.parent
    parent_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix="openpi.bootstrap.", dir=parent_root))
    staging_repo_root = staging_root / "openpi"
    try:
        _run_git(["clone", "--depth", "1", bootstrap_source_url, str(staging_repo_root)])
        git_revision = _checkout_bootstrap_ref(staging_repo_root, bootstrap_ref=bootstrap_ref)
        marker = BootstrapMarker(
            schema_version=BOOTSTRAP_SCHEMA_VERSION,
            bootstrap_source_url=bootstrap_source_url,
            bootstrap_ref=bootstrap_ref,
            git_revision=git_revision,
            created_at_unix=time.time(),
            expected_repo_paths=EXPECTED_REPO_PATHS,
        )
        _write_marker(staging_repo_root, marker)
        _validate_repo_root(
            staging_repo_root,
            bootstrap_source_url=bootstrap_source_url,
            bootstrap_ref=bootstrap_ref,
            require_marker=True,
        )
        try:
            os.replace(staging_repo_root, repo_root)
        except (FileExistsError, OSError) as exc:
            raise OpenPIRuntimeError(
                "cache_stale",
                f"OPENPI_CACHE_STALE: managed cache path `{repo_root}` appeared during bootstrap; rerun after inspection.",
            ) from exc
        return marker, git_revision
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _http_json(url: str, payload: dict[str, Any], timeout_s: float = 3.0) -> dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=encoded, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
    if not body.strip():
        return {}
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise OpenPIRuntimeError(
            "policy_server_unhealthy",
            f"OPENPI_POLICY_SERVER_UNHEALTHY: expected JSON object from `{url}`.",
        )
    return parsed


def _check_policy_server(*, policy_server_url: str, checkpoint: str, openpi_conda_env: str, openpi_repo_root: str) -> None:
    payload = {
        "ping": "pi0",
        "checkpoint": checkpoint,
        "runtime": {
            "OPENPI_CONDA_ENV": openpi_conda_env,
            "OPENPI_REPO_ROOT": openpi_repo_root,
        },
    }
    errors: list[str] = []
    for endpoint in ("/healthz", "/health", "/"):
        health_url = f"{policy_server_url.rstrip('/')}{endpoint}"
        try:
            response = _http_json(health_url, payload)
            status = str(response.get("status", "ok")).strip().lower()
            if status not in {"ok", "healthy", "ready"}:
                raise OpenPIRuntimeError(
                    "policy_server_unhealthy",
                    f"OPENPI_POLICY_SERVER_UNHEALTHY: `{health_url}` returned status `{status}`.",
                )
            return
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError, OpenPIRuntimeError) as exc:
            errors.append(f"{health_url}: {exc}")

    details = " | ".join(errors)
    raise OpenPIRuntimeError(
        "policy_server_unhealthy",
        "OPENPI_POLICY_SERVER_UNHEALTHY: unable to reach a healthy OpenPI policy server. "
        f"details={details}",
    )


def _emit_env_file(path: Path, metadata: RuntimeMetadata) -> None:
    lines = [
        f"export OPENPI_REPO_ROOT={shlex.quote(metadata.repo_root)}",
        f"export OPENVLA_MANISKILL_OPENPI_REPO_ROOT={shlex.quote(metadata.repo_root)}",
        f"export OPENPI_CONDA_ENV={shlex.quote(metadata.openpi_conda_env)}",
        f"export OPENVLA_MANISKILL_OPENPI_CONDA_ENV={shlex.quote(metadata.openpi_conda_env)}",
        f"export OPENPI_POLICY_SERVER_URL={shlex.quote(metadata.policy_server_url)}",
        f"export OPENVLA_MANISKILL_PI0_POLICY_SERVER_URL={shlex.quote(metadata.policy_server_url)}",
        f"export OPENPI_BOOTSTRAP_CACHE_STATE={shlex.quote(metadata.cache_state)}",
        f"export OPENPI_BOOTSTRAP_ACTION={shlex.quote(metadata.bootstrap_action)}",
        f"export OPENPI_BOOTSTRAP_MARKER={shlex.quote(metadata.bootstrap_marker_path)}",
        f"export OPENPI_BOOTSTRAP_SOURCE_URL={shlex.quote(metadata.bootstrap_source_url)}",
        f"export OPENPI_BOOTSTRAP_REF={shlex.quote(metadata.bootstrap_ref)}",
        f"export OPENPI_BOOTSTRAP_GIT_REVISION={shlex.quote(metadata.git_revision)}",
        f"export OPENPI_POLICY_SERVER_STATUS={shlex.quote(metadata.policy_server_status)}",
        f"export OPENPI_POLICY_SERVER_PYTHON={shlex.quote(metadata.policy_server_python)}",
        f"export OPENPI_POLICY_SERVER_ENTRYPOINT={shlex.quote(metadata.policy_server_entrypoint)}",
        f"export OPENPI_POLICY_SERVER_LAUNCH_PREFIX={shlex.quote(metadata.policy_server_launch_prefix)}",
    ]
    path.write_text("\n".join(lines) + "\n")


def _policy_server_python() -> str:
    return "python3"


def _policy_server_entrypoint(repo_root: Path) -> Path:
    return (repo_root / "scripts" / "serve_policy.py").resolve()


def _policy_server_launch_prefix(repo_root: Path) -> str:
    command = [_policy_server_python(), str(_policy_server_entrypoint(repo_root))]
    return shlex.join(command)


def _build_metadata(args: argparse.Namespace) -> RuntimeMetadata:
    managed_repo_root = _managed_repo_root()
    bootstrap_source_url = args.bootstrap_repo_url.strip() or DEFAULT_OPENPI_REPO_URL
    bootstrap_ref = args.bootstrap_ref.strip() or DEFAULT_OPENPI_BOOTSTRAP_REF
    explicit_repo_root = args.openpi_repo_root.strip()
    checkpoint = args.checkpoint.strip() or DEFAULT_OPENPI_CHECKPOINT
    openpi_conda_env = args.openpi_conda_env.strip() or DEFAULT_OPENPI_CONDA_ENV
    policy_server_url = args.policy_server_url.strip() or DEFAULT_OPENPI_POLICY_SERVER_URL

    if explicit_repo_root:
        repo_root = Path(explicit_repo_root).expanduser().resolve()
        _, git_revision = _validate_repo_root(
            repo_root,
            bootstrap_source_url=bootstrap_source_url,
            require_marker=False,
        )
        cache_state = "explicit_repo_valid"
        bootstrap_action = "explicit_repo_reused"
        marker_path = repo_root / BOOTSTRAP_MARKER_NAME
    else:
        repo_root = managed_repo_root
        try:
            marker, git_revision = _validate_repo_root(
                repo_root,
                bootstrap_source_url=bootstrap_source_url,
                bootstrap_ref=bootstrap_ref,
                require_marker=True,
            )
            if marker is None:
                raise OpenPIRuntimeError("cache_invalid", f"OPENPI_CACHE_INVALID: missing marker for `{repo_root}`.")
            cache_state = "cache_valid"
            bootstrap_action = "cache_reused"
            marker_path = repo_root / BOOTSTRAP_MARKER_NAME
        except OpenPIRuntimeError as exc:
            if exc.state not in {"cache_missing", "cache_invalid", "cache_stale"}:
                raise
            if exc.state in {"cache_invalid", "cache_stale"} and repo_root.exists():
                shutil.rmtree(repo_root, ignore_errors=True)
            marker, git_revision = _bootstrap_managed_repo(
                repo_root,
                bootstrap_source_url=bootstrap_source_url,
                bootstrap_ref=bootstrap_ref,
            )
            cache_state = f"{exc.state}_bootstrapped"
            bootstrap_action = "cache_rebootstrapped" if exc.state != "cache_missing" else "cache_bootstrapped"
            marker_path = _write_marker(repo_root, marker)

    policy_server_status = "skipped"
    if args.require_policy_server_health:
        _check_policy_server(
            policy_server_url=policy_server_url,
            checkpoint=checkpoint,
            openpi_conda_env=openpi_conda_env,
            openpi_repo_root=str(repo_root),
        )
        policy_server_status = "healthy"

    return RuntimeMetadata(
        status="ready",
        cache_state=cache_state,
        bootstrap_action=bootstrap_action,
        repo_root=str(repo_root),
        managed_repo_root=str(managed_repo_root),
        bootstrap_marker_path=str(marker_path),
        bootstrap_source_url=bootstrap_source_url,
        bootstrap_ref=bootstrap_ref,
        git_revision=git_revision,
        policy_server_status=policy_server_status,
        policy_server_url=policy_server_url,
        policy_server_python=_policy_server_python(),
        policy_server_entrypoint=str(_policy_server_entrypoint(repo_root)),
        policy_server_launch_prefix=_policy_server_launch_prefix(repo_root),
        checkpoint=checkpoint,
        openpi_conda_env=openpi_conda_env,
    )


def _print_metadata(metadata: RuntimeMetadata) -> None:
    print(f"openpi_runtime_status={metadata.status}")
    print(f"openpi_cache_state={metadata.cache_state}")
    print(f"openpi_bootstrap_action={metadata.bootstrap_action}")
    print(f"openpi_repo_root={metadata.repo_root}")
    print(f"openpi_managed_repo_root={metadata.managed_repo_root}")
    print(f"openpi_bootstrap_marker={metadata.bootstrap_marker_path}")
    print(f"openpi_bootstrap_source_url={metadata.bootstrap_source_url}")
    print(f"openpi_bootstrap_ref={metadata.bootstrap_ref}")
    print(f"openpi_bootstrap_git_revision={metadata.git_revision}")
    print(f"openpi_policy_server_status={metadata.policy_server_status}")
    print(f"openpi_policy_server_url={metadata.policy_server_url}")
    print(f"openpi_policy_server_python={metadata.policy_server_python}")
    print(f"openpi_policy_server_entrypoint={metadata.policy_server_entrypoint}")
    print(f"openpi_policy_server_launch_prefix={metadata.policy_server_launch_prefix}")
    print(f"openpi_checkpoint={metadata.checkpoint}")
    print(f"openpi_conda_env={metadata.openpi_conda_env}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate or bootstrap the managed OpenPI runtime checkout.")
    parser.add_argument("--openpi-repo-root", default=os.environ.get("OPENPI_REPO_ROOT", ""))
    parser.add_argument("--openpi-conda-env", default=os.environ.get("OPENPI_CONDA_ENV", DEFAULT_OPENPI_CONDA_ENV))
    parser.add_argument("--checkpoint", default=os.environ.get("OPENPI_CHECKPOINT", DEFAULT_OPENPI_CHECKPOINT))
    parser.add_argument(
        "--policy-server-url",
        default=os.environ.get("OPENPI_POLICY_SERVER_URL", DEFAULT_OPENPI_POLICY_SERVER_URL),
    )
    parser.add_argument(
        "--bootstrap-repo-url",
        default=os.environ.get("OPENPI_BOOTSTRAP_REPO_URL", DEFAULT_OPENPI_REPO_URL),
    )
    parser.add_argument(
        "--bootstrap-ref",
        default=os.environ.get("OPENPI_BOOTSTRAP_REF", DEFAULT_OPENPI_BOOTSTRAP_REF),
    )
    parser.add_argument("--emit-env-file", default="")
    parser.add_argument("--require-policy-server-health", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metadata = _build_metadata(args)
    except OpenPIRuntimeError as exc:
        print("openpi_runtime_status=error")
        print(f"openpi_error_state={exc.state}")
        print(f"openpi_error_message={exc}")
        return 1

    if args.emit_env_file.strip():
        env_path = Path(args.emit_env_file).expanduser().resolve()
        env_path.parent.mkdir(parents=True, exist_ok=True)
        _emit_env_file(env_path, metadata)
    _print_metadata(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
