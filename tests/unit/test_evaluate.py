import numpy as np

from src.models.evaluate import compute_classification_metrics


def test_compute_classification_metrics_returns_expected_keys() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = compute_classification_metrics(y_true, y_pred_proba)

    expected_keys = {"roc_auc", "accuracy", "precision", "recall", "f1"}
    assert set(metrics.keys()) == expected_keys


def test_compute_classification_metrics_perfect_predictions() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = compute_classification_metrics(y_true, y_pred_proba, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_classification_metrics_threshold_changes_predictions() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred_proba = np.array([0.4, 0.6, 0.45, 0.3])

    metrics_default = compute_classification_metrics(y_true, y_pred_proba, threshold=0.5)
    metrics_strict = compute_classification_metrics(y_true, y_pred_proba, threshold=0.7)

    assert metrics_default["recall"] >= metrics_strict["recall"]


def test_compute_classification_metrics_handles_zero_division() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = compute_classification_metrics(y_true, y_pred_proba, threshold=0.5)

    assert metrics["roc_auc"] == 0.0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
