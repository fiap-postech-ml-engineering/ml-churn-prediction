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
        The trimmed stdout value when the command succeeds; otherwise "unknown".
    """
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def get_owner() -> str:
    """Resolve the run owner from common CI/local environment variables.

    Priority order:
    1. GITHUB_ACTOR (GitHub Actions)
    2. USER
    3. USERNAME

    Returns:
        Resolved owner identifier, or "unknown" when none is available.
    """
    return (
        os.getenv("GITHUB_ACTOR")
        or os.getenv("USER")
        or os.getenv("USERNAME")
        or "unknown"
    )


def configure_mlflow_tracking(
    experiment_name: str,
    db_path: str | Path,
    experiment_tags: dict[str, str] | None = None,
) -> str:
    """Configure MLflow backend and optionally persist experiment-level tags.

    Args:
        experiment_name: Logical MLflow experiment name.
        db_path: Filesystem path to the SQLite database used as backend store.
        experiment_tags: Optional key-value tags to set on the experiment.

    Returns:
        The tracking URI configured for MLflow in the current process.
    """
    db_path = Path(db_path).resolve().as_posix()
    tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if experiment_tags:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            for key, value in experiment_tags.items():
                client.set_experiment_tag(experiment.experiment_id, key, value)

    return tracking_uri


def build_default_run_tags(
    stage: str,
    model_family: str,
    sampling_strategy: str,
    extra_tags: dict[str, str] | None = None,
) -> dict[str, str]:
    """Create standardized run tags for reproducible experiment tracking.

    Args:
        stage: Lifecycle stage such as baseline, modeling, or final.
        model_family: Model category, for example logistic_regression or mlp.
        sampling_strategy: Class balancing approach, e.g. none, smote, balanced.
        extra_tags: Optional extra tags to merge into the resulting payload.

    Returns:
        Dictionary of normalized tag values for MLflow runs.
    """
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    branch = os.getenv("GITHUB_REF_NAME") or _safe_git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = os.getenv("GITHUB_SHA") or _safe_git(["git", "rev-parse", "HEAD"])

    tags = {
        "owner": get_owner(),
        "stage": stage,
        "model_family": model_family,
        "sampling_strategy": sampling_strategy,
        "runner": "github_actions" if is_ci else "local",
        "ci": str(is_ci).lower(),
        "git_branch": branch,
        "git_commit": commit,
        "python_version": sys.version.split()[0],
        "platform_os": platform.platform(),
        "host": socket.gethostname(),
        "run_timestamp_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
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