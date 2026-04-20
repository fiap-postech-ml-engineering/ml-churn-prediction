from __future__ import annotations

import copy
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.evaluate import compute_classification_metrics


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Executa uma época de treino e retorna a loss média.
    """
    model.train()
    train_loss_total = 0.0
    train_samples = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * len(y_batch)
        train_samples += len(y_batch)

    return train_loss_total / train_samples


def evaluate_one_epoch(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Executa avaliação em um DataLoader e retorna:
    - y_true
    - y_pred_proba
    """
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()

            preds_all.append(probs.flatten())
            targets_all.append(y_batch.cpu().numpy())

    y_pred_proba = np.concatenate(preds_all, axis=0)
    y_true = np.concatenate(targets_all, axis=0)

    return y_true, y_pred_proba


def train_mlp_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    learning_rate: float = 1e-4,
    pos_weight: float = 1.0,
    max_epochs: int = 200,
    early_stopping_patience: int = 20,
    mlflow_run_name: str = "mlp_pytorch_training",
    use_mlflow: bool = True,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, float], dict[str, Any]]:
    """
    Treina o MLP com early stopping monitorando ROC-AUC no validation set.

    Returns:
        best_model:
            Modelo com os melhores pesos encontrados.
        history:
            Histórico por época.
        best_val_metrics:
            Métricas do melhor validation result.
        training_summary:
            Resumo do treino (best_epoch, epochs_trained, best_val_roc_auc).
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    history = {
        "epochs": [],
        "train_loss": [],
        "val_roc_auc": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_val_roc_auc = -np.inf
    epochs_without_improvement = 0
    best_model_state = None
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}

    def _train_loop() -> None:
        nonlocal best_val_roc_auc, epochs_without_improvement, best_model_state
        nonlocal best_epoch, best_val_metrics

        for epoch in range(max_epochs):
            train_loss_avg = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            val_targets, val_preds = evaluate_one_epoch(
                model=model,
                data_loader=val_loader,
                device=device,
            )

            val_metrics = compute_classification_metrics(
                y_true=val_targets,
                y_pred_proba=val_preds,
            )
            val_roc_auc = val_metrics["roc_auc"]

            history["epochs"].append(epoch + 1)
            history["train_loss"].append(train_loss_avg)
            history["val_roc_auc"].append(val_roc_auc)
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1"])

            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "val_roc_auc": val_roc_auc,
                        "val_accuracy": val_metrics["accuracy"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "val_f1": val_metrics["f1"],
                    },
                    step=epoch,
                )

            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                best_val_metrics = val_metrics
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

    if use_mlflow:
        with mlflow.start_run(run_name=mlflow_run_name):
            mlflow.log_param("model_type", "MLP")
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("pos_weight", float(pos_weight))
            mlflow.log_param("max_epochs", max_epochs)
            mlflow.log_param("early_stopping_patience", early_stopping_patience)

            if hasattr(model, "input_size"):
                mlflow.log_param("input_features", model.input_size)
            if hasattr(model, "hidden_dims"):
                mlflow.log_param("hidden_dims", str(model.hidden_dims))
            if hasattr(model, "dropout_rates"):
                mlflow.log_param("dropout_rates", str(model.dropout_rates))

            _train_loop()

            mlflow.log_metrics(
                {
                    "final_epochs_trained": len(history["epochs"]),
                    "best_epoch": best_epoch,
                    "best_val_roc_auc_final": best_val_roc_auc,
                }
            )
    else:
        _train_loop()

    training_summary = {
        "best_epoch": best_epoch,
        "epochs_trained": len(history["epochs"]),
        "best_val_roc_auc": float(best_val_roc_auc),
    }

    return model, history, best_val_metrics, training_summary