from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.models.evaluate import compute_classification_metrics


def build_dummy_classifier() -> DummyClassifier:
    """
    Cria o baseline DummyClassifier usado como referência mínima.
    """
    return DummyClassifier(strategy="most_frequent")


def build_logistic_regression(
    random_state: int = 42,
    class_weight: str | None = None,
) -> LogisticRegression:
    """
    Cria LogisticRegression alinhada aos experimentos do notebook 02.

    Args:
        random_state: Seed para reprodutibilidade.
        class_weight: Pode ser None ou "balanced".

    Returns:
        Instância configurada de LogisticRegression.
    """
    return LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight=class_weight,
    )


def train_and_evaluate_baseline(
    model: Any,
    x_train,
    y_train,
    x_test,
    y_test,
) -> dict[str, float]:
    """
    Treina um baseline e retorna métricas no test set.
    """
    model.fit(x_train, y_train)

    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(x_test)[:, 1]
    else:
        y_pred_proba = model.predict(x_test)

    return compute_classification_metrics(
        y_true=np.asarray(y_test),
        y_pred_proba=np.asarray(y_pred_proba),
    )


def log_baseline_run_to_mlflow(
    run_name: str,
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
) -> None:
    """
    Registra parâmetros e métricas de um baseline no MLflow.

    Observação:
        Mantido simples para ficar alinhado ao card e ao notebook 02.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", model_name)

        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)

        mlflow.log_metrics(metrics)
