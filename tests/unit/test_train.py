import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp_model import MLPNetworkChurn
from src.models.train import (
    evaluate_one_epoch,
    train_mlp_model,
    train_one_epoch,
)


def make_dataloader(
    n_samples: int = 12,
    input_size: int = 4,
    batch_size: int = 4,
):
    X = torch.randn(n_samples, input_size)
    y = torch.tensor([0, 1] * (n_samples // 2), dtype=torch.float32)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_train_one_epoch_returns_float_loss() -> None:
    device = torch.device("cpu")
    model = MLPNetworkChurn(
        input_size=4,
        hidden_dims=[8, 4],
        dropout_rates=[0.1, 0.1],
    ).to(device)

    train_loader = make_dataloader(n_samples=12, input_size=4, batch_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_evaluate_one_epoch_returns_arrays_with_same_length() -> None:
    device = torch.device("cpu")
    model = MLPNetworkChurn(
        input_size=4,
        hidden_dims=[8, 4],
        dropout_rates=[0.1, 0.1],
    ).to(device)

    data_loader = make_dataloader(n_samples=12, input_size=4, batch_size=4)

    y_true, y_pred_proba = evaluate_one_epoch(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred_proba, np.ndarray)
    assert len(y_true) == len(y_pred_proba) == 12


def test_train_mlp_model_returns_expected_outputs_without_mlflow() -> None:
    device = torch.device("cpu")
    model = MLPNetworkChurn(
        input_size=4,
        hidden_dims=[8, 4],
        dropout_rates=[0.1, 0.1],
    ).to(device)

    train_loader = make_dataloader(n_samples=12, input_size=4, batch_size=4)
    val_loader = make_dataloader(n_samples=12, input_size=4, batch_size=4)

    trained_model, history, best_val_metrics, training_summary = train_mlp_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-3,
        pos_weight=1.0,
        max_epochs=3,
        early_stopping_patience=2,
        use_mlflow=False,
    )

    assert trained_model is model

    assert isinstance(history, dict)
    assert "epochs" in history
    assert "train_loss" in history
    assert "val_roc_auc" in history

    assert len(history["epochs"]) >= 1
    assert len(history["train_loss"]) >= 1
    assert len(history["val_roc_auc"]) >= 1

    expected_metric_keys = {"roc_auc", "accuracy", "precision", "recall", "f1"}
    assert set(best_val_metrics.keys()) == expected_metric_keys

    assert isinstance(training_summary, dict)
    assert "best_epoch" in training_summary
    assert "epochs_trained" in training_summary
    assert "best_val_roc_auc" in training_summary

    assert training_summary["epochs_trained"] >= 1