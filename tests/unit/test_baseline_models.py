import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.models.baseline_models import (
    build_dummy_classifier,
    build_logistic_regression,
    log_baseline_run_to_mlflow,
    train_and_evaluate_baseline,
)


def make_classification_data():
    x_train = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.9, 0.1],
        ]
    )
    y_train = np.array([0, 1, 0, 1, 0, 1])

    x_test = np.array(
        [
            [0.05, 0.95],
            [0.95, 0.05],
            [0.3, 0.7],
            [0.7, 0.3],
        ]
    )
    y_test = np.array([0, 1, 0, 1])

    return x_train, y_train, x_test, y_test


def test_build_dummy_classifier_returns_dummy_classifier() -> None:
    model = build_dummy_classifier()

    assert isinstance(model, DummyClassifier)
    assert model.strategy == "most_frequent"


def test_build_logistic_regression_returns_logistic_regression() -> None:
    model = build_logistic_regression()

    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 1000
    assert model.random_state == 42
    assert model.class_weight is None


def test_build_logistic_regression_with_balanced_class_weight() -> None:
    model = build_logistic_regression(class_weight="balanced")

    assert isinstance(model, LogisticRegression)
    assert model.class_weight == "balanced"


def test_train_and_evaluate_baseline_returns_expected_metrics() -> None:
    x_train, y_train, x_test, y_test = make_classification_data()
    model = build_logistic_regression()

    metrics = train_and_evaluate_baseline(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    expected_keys = {"roc_auc", "accuracy", "precision", "recall", "f1"}
    assert set(metrics.keys()) == expected_keys

    for value in metrics.values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


def test_train_and_evaluate_baseline_with_dummy_classifier() -> None:
    x_train, y_train, x_test, y_test = make_classification_data()
    model = build_dummy_classifier()

    metrics = train_and_evaluate_baseline(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    expected_keys = {"roc_auc", "accuracy", "precision", "recall", "f1"}
    assert set(metrics.keys()) == expected_keys


def test_log_baseline_run_to_mlflow_logs_params_and_metrics(monkeypatch) -> None:
    logged_params = {}
    logged_metrics = {}

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    def fake_start_run(run_name=None):
        assert run_name == "baseline_run"
        return DummyRun()

    def fake_log_param(key, value):
        logged_params[key] = value

    def fake_log_metrics(metrics):
        logged_metrics.update(metrics)

    monkeypatch.setattr(
        "src.models.baseline_models.mlflow.start_run",
        fake_start_run,
    )
    monkeypatch.setattr(
        "src.models.baseline_models.mlflow.log_param",
        fake_log_param,
    )
    monkeypatch.setattr(
        "src.models.baseline_models.mlflow.log_metrics",
        fake_log_metrics,
    )

    metrics = {
        "roc_auc": 0.91,
        "accuracy": 0.87,
        "precision": 0.88,
        "recall": 0.85,
        "f1": 0.86,
    }
    params = {"class_weight": "balanced", "max_iter": 1000}

    log_baseline_run_to_mlflow(
        run_name="baseline_run",
        model_name="logistic_regression",
        metrics=metrics,
        params=params,
    )

    assert logged_params["model_type"] == "logistic_regression"
    assert logged_params["class_weight"] == "balanced"
    assert logged_params["max_iter"] == 1000
    assert logged_metrics == metrics
