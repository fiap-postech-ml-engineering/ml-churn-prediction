import runpy
from pathlib import Path


CONFIG_FILE = Path("src/config.py")


def run_config_module() -> dict:
    return runpy.run_path(str(CONFIG_FILE))


def test_config_uses_default_values(monkeypatch) -> None:
    monkeypatch.delenv("APPROVAL_THRESHOLD", raising=False)
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    result = run_config_module()

    assert result["RANDOM_SEED"] == 42
    assert result["APPROVAL_THRESHOLD"] == 0.6
    assert result["MODEL_VERSION"] == "0.1.0"
    assert result["LOG_LEVEL"] == "INFO"
    assert result["ENVIRONMENT"] == "development"
    assert result["DEBUG"] is True

    assert result["API_TITLE"] == "Churn API"
    assert result["API_VERSION"] == "0.1.0"
    assert result["API_DESCRIPTION"] == "Churn Prediction API using FastAPI"


def test_config_respects_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("APPROVAL_THRESHOLD", "0.85")
    monkeypatch.setenv("MODEL_VERSION", "2.1.3")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ENVIRONMENT", "production")

    result = run_config_module()

    assert result["APPROVAL_THRESHOLD"] == 0.85
    assert result["MODEL_VERSION"] == "2.1.3"
    assert result["LOG_LEVEL"] == "DEBUG"
    assert result["ENVIRONMENT"] == "production"
    assert result["DEBUG"] is False


def test_config_creates_expected_paths() -> None:
    result = run_config_module()

    base_dir = result["BASE_DIR"]
    logs_dir = result["LOGS_DIR"]
    models_dir = result["MODELS_DIR"]
    data_dir = result["DATA_DIR"]
    drift_log_path = result["DRIFT_LOG_PATH"]

    assert isinstance(base_dir, Path)
    assert isinstance(logs_dir, Path)
    assert isinstance(models_dir, Path)
    assert isinstance(data_dir, Path)
    assert isinstance(drift_log_path, Path)

    assert logs_dir.exists()
    assert models_dir.exists()
    assert data_dir.exists()

    assert logs_dir == base_dir / "logs"
    assert models_dir == base_dir / "models"
    assert data_dir == base_dir / "data"
    assert drift_log_path == logs_dir / "input_samples.jsonl"