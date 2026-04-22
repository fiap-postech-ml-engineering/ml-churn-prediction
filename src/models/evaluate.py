from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calcula métricas de classificação binária a partir das probabilidades previstas.

    Args:
        y_true: Rótulos verdadeiros (0/1).
        y_pred_proba: Probabilidades previstas para a classe positiva.
        threshold: Threshold para converter probabilidade em classe prevista.

    Returns:
        Dicionário com ROC-AUC, Accuracy, Precision, Recall e F1.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        roc_auc = 0.0
    else:
        roc_auc = float(roc_auc_score(y_true, y_pred_proba))

    return {
        "roc_auc": roc_auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
