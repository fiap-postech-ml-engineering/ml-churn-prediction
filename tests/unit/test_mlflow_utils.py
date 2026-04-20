from pathlib import Path

from src.tracking.mlflow_utils import (
    _safe_git,
    build_default_run_tags,
    configure_mlflow_tracking,
    normalize_params,
)


def test_safe_git_returns_stdout_on_success(monkeypatch) -> None:
    def fake_check_output(cmd, stderr=None):
        return b"main\n"

    monkeypatch.setattr(
        "src.tracking.mlflow_utils.subprocess.check_output",
        fake_check_output,
    )

    result = _safe_git(["git", "branch", "--show-current"])

    assert result == "main"


def test_safe_git_returns_unknown_on_failure(monkeypatch) -> None:
    def fake_check_output(cmd, stderr=None):
        raise RuntimeError("git failed")

    monkeypatch.setattr(
        "src.tracking.mlflow_utils.subprocess.check_output",
        fake_check_output,
    )

    result = _safe_git(["git", "branch", "--show-current"])

    assert result == "unknown"


def test_normalize_params_keeps_primitives_and_converts_objects() -> None:
    class CustomObject:
        def __str__(self) -> str:
            return "custom-object"

    params = {
        "name": "mlp",
        "epochs": 10,
        "lr": 0.001,
        "enabled": True,
        "optional": None,
        "custom": CustomObject(),
        "path": Path("models/model.pth"),
    }

    normalized = normalize_params(params)

    assert normalized["name"] == "mlp"
    assert normalized["epochs"] == 10
    assert normalized["lr"] == 0.001
    assert normalized["enabled"] is True
    assert normalized["optional"] is None
    assert normalized["custom"] == "custom-object"
    assert normalized["path"].replace("\\", "/") == "models/model.pth"


def test_build_default_run_tags_uses_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REF_NAME", "feature/test")
    monkeypatch.setenv("GITHUB_SHA", "abc123")

    tags = build_default_run_tags(extra_tags={"team": "ml"})

    assert tags["runner"] == "github_actions"
    assert tags["ci"] == "true"
    assert tags["git_branch"] == "feature/test"
    assert tags["git_commit"] == "abc123"
    assert tags["team"] == "ml"
    assert "python_version" in tags
    assert "platform_os" in tags
    assert "host" in tags
    assert "run_timestamp_utc" in tags


def test_build_default_run_tags_uses_safe_git_when_env_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    responses = {
        ("git", "rev-parse", "--abbrev-ref", "HEAD"): "dev-branch",
        ("git", "rev-parse", "HEAD"): "commit123",
    }

    def fake_safe_git(cmd):
        return responses[tuple(cmd)]

    monkeypatch.setattr(
        "src.tracking.mlflow_utils._safe_git",
        fake_safe_git,
    )

    tags = build_default_run_tags()

    assert tags["runner"] == "local"
    assert tags["ci"] == "false"
    assert tags["git_branch"] == "dev-branch"
    assert tags["git_commit"] == "commit123"


def test_configure_mlflow_tracking_creates_experiment_when_missing(tmp_path, monkeypatch) -> None:
    calls = {
        "set_tracking_uri": None,
        "set_experiment": None,
        "create_experiment": None,
    }

    class FakeClient:
        def get_experiment_by_name(self, experiment_name):
            assert experiment_name == "churn-exp"
            return None

        def create_experiment(self, name, artifact_location, tags):
            calls["create_experiment"] = {
                "name": name,
                "artifact_location": artifact_location,
                "tags": tags,
            }
            return "exp-id"

    def fake_set_tracking_uri(uri):
        calls["set_tracking_uri"] = uri

    def fake_set_experiment(name):
        calls["set_experiment"] = name

    monkeypatch.setattr(
        "src.tracking.mlflow_utils.MlflowClient",
        FakeClient,
    )
    monkeypatch.setattr(
        "src.tracking.mlflow_utils.mlflow.set_tracking_uri",
        fake_set_tracking_uri,
    )
    monkeypatch.setattr(
        "src.tracking.mlflow_utils.mlflow.set_experiment",
        fake_set_experiment,
    )

    db_path = tmp_path / "mlruns_store"
    artifact_root = tmp_path / "artifacts"

    tracking_uri = configure_mlflow_tracking(
        experiment_name="churn-exp",
        db_path=db_path,
        experiment_tags={"stage": "dev"},
        artifact_root_path=artifact_root,
    )

    assert db_path.exists()
    assert artifact_root.exists()
    assert tracking_uri.startswith("file://")
    assert calls["set_tracking_uri"] == tracking_uri
    assert calls["set_experiment"] == "churn-exp"
    assert calls["create_experiment"]["name"] == "churn-exp"
    assert calls["create_experiment"]["tags"] == {"stage": "dev"}


def test_configure_mlflow_tracking_does_not_create_experiment_when_it_exists(tmp_path, monkeypatch) -> None:
    calls = {
        "set_tracking_uri": None,
        "set_experiment": None,
        "create_experiment_called": False,
    }

    class FakeExperiment:
        experiment_id = "123"

    class FakeClient:
        def get_experiment_by_name(self, experiment_name):
            return FakeExperiment()

        def create_experiment(self, name, artifact_location, tags):
            calls["create_experiment_called"] = True
            return "exp-id"

    def fake_set_tracking_uri(uri):
        calls["set_tracking_uri"] = uri

    def fake_set_experiment(name):
        calls["set_experiment"] = name

    monkeypatch.setattr(
        "src.tracking.mlflow_utils.MlflowClient",
        FakeClient,
    )
    monkeypatch.setattr(
        "src.tracking.mlflow_utils.mlflow.set_tracking_uri",
        fake_set_tracking_uri,
    )
    monkeypatch.setattr(
        "src.tracking.mlflow_utils.mlflow.set_experiment",
        fake_set_experiment,
    )

    db_path = tmp_path / "mlruns_store"

    tracking_uri = configure_mlflow_tracking(
        experiment_name="existing-exp",
        db_path=db_path,
        experiment_tags={"stage": "prod"},
        artifact_root_path=tmp_path / "artifacts",
    )

    assert db_path.exists()
    assert tracking_uri.startswith("file://")
    assert calls["set_tracking_uri"] == tracking_uri
    assert calls["set_experiment"] == "existing-exp"
    assert calls["create_experiment_called"] is False