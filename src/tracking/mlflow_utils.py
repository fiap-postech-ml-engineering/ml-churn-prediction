from __future__ import annotations

import datetime as dt
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def _safe_git(cmd: list[str]) -> str:
    """Execute a git command and return its stdout or a safe fallback.

    Args:
        cmd: Command tokens to execute with subprocess.

    Returns:
        The trimmed stdout value when the command succeeds or "unknown".
    """
    try:
        return (
            subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
            )
    except Exception:
        return "unknown"


def configure_mlflow_tracking(
    experiment_name: str,
    db_path: str | Path,
    experiment_tags: dict[str, str] | None = None,
    artifact_root_path: str | Path | None = None,
) -> str:
    """Configure MLflow backend and optionally persist experiment-level tags.

    Args:
        experiment_name: Logical MLflow experiment name.
        db_path: Filesystem path to the MLflow file store (backend_store_uri).
        Uses file:// for directory-based storage instead of SQLite.
        experiment_tags: Optional key-value tags to set on the experiment.
        artifact_root_path: Optional filesystem path for experiment artifacts.

    Returns:
        The tracking URI configured for MLflow in the current process.
    """
    # Use file-based store (directory) instead of SQLite for better portability
    store_path = Path(db_path)
    store_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = store_path.as_posix()

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None and artifact_root_path is not None:
        artifact_root_dir = Path(artifact_root_path).resolve()
        artifact_root_dir.mkdir(parents=True, exist_ok=True)
        # Store path as string for portability
        artifact_location = str(artifact_root_path).replace("\\", "/")
        client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
            tags=experiment_tags,
        )

    mlflow.set_experiment(experiment_name)

    return tracking_uri


def build_default_run_tags(
    extra_tags: dict[str, str] | None = None,
) -> dict[str, str]:
    """Create standardized run tags for reproducible experiment tracking.

    Args:
        extra_tags: Optional extra tags to merge into the resulting payload.

    Returns:
        Dictionary of normalized tag values for MLflow runs.
    """
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    branch = (os.getenv("GITHUB_REF_NAME") or
              _safe_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]))
    commit = os.getenv("GITHUB_SHA") or _safe_git(["git", "rev-parse", "HEAD"])

    tags = {
        "runner": "github_actions" if is_ci else "local",
        "ci": str(is_ci).lower(),
        "git_branch": branch,
        "git_commit": commit,
        "python_version": sys.version.split()[0],
        "platform_os": platform.platform(),
        "host": socket.gethostname(),
        "run_timestamp_utc":(
            dt.datetime
            .now(dt.timezone.utc).replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
            )
    }

    if extra_tags:
        tags.update(extra_tags)

    return tags


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize parameters into MLflow-compatible primitive values.

    Args:
        params: Arbitrary parameter dictionary.

    Returns:
        New dictionary with non-primitive values converted to strings.
    """
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized
