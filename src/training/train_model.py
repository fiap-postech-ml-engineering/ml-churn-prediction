from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import settings
from src.data.load_data import load_csv_data
from src.data.preprocessing import (
    fit_tabular_preprocessing_pipeline,
    save_tabular_preprocessing_pipeline,
    select_tabular_raw_features,
    split_tabular_features_target,
    transform_tabular_features,
)
from src.models.evaluate import compute_classification_metrics
from src.models.mlp_model import MLPNetworkChurn
from src.models.train import train_mlp_model
from src.tracking.mlflow_utils import (
    build_default_run_tags,
    configure_mlflow_tracking,
    normalize_params,
)

RAW_DATASET_PATH = Path("data/raw/telco_customer_churn.csv")


def _load_training_frame() -> pd.DataFrame:
    df = load_csv_data(str(RAW_DATASET_PATH))
    if df is None:
        raise FileNotFoundError(f"Training dataset not found at: {RAW_DATASET_PATH}")
    return df


def _build_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x_val).float(),
        torch.from_numpy(y_val).float(),
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    )


def _save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def train_tabular_mlp_from_raw_dataset() -> dict[str, float]:
    configure_mlflow_tracking(
        experiment_name=settings.EXPERIMENT_NAME,
        db_path=settings.MLFLOW_TRACKING_PATH,
        experiment_tags=settings.EXPERIMENT_TAGS,
        artifact_root_path=settings.MLFLOW_ARTIFACTS_PATH,
    )

    raw_df = _load_training_frame()
    raw_df = select_tabular_raw_features(raw_df, require_target=True)

    x, y = split_tabular_features_target(raw_df)
    y = y.astype(int).to_numpy()

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=settings.TEST_SIZE + settings.VALIDATION_SIZE,
        random_state=settings.RANDOM_SEED,
        stratify=y,
    )
    relative_val_size = settings.VALIDATION_SIZE / (
        settings.TEST_SIZE + settings.VALIDATION_SIZE
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=1.0 - relative_val_size,
        random_state=settings.RANDOM_SEED,
        stratify=y_temp,
    )

    preprocessing_pipeline, feature_names = fit_tabular_preprocessing_pipeline(x_train)
    save_tabular_preprocessing_pipeline(
        preprocessing_pipeline=preprocessing_pipeline,
        feature_names=feature_names,
        output_path=settings.TABULAR_PREPROCESSING_PIPELINE_PATH,
    )

    x_train_transformed = transform_tabular_features(preprocessing_pipeline, x_train)
    x_val_transformed = transform_tabular_features(preprocessing_pipeline, x_val)
    x_test_transformed = transform_tabular_features(preprocessing_pipeline, x_test)

    train_loader, val_loader = _build_dataloaders(
        x_train=x_train_transformed,
        y_train=y_train,
        x_val=x_val_transformed,
        y_val=y_val,
        batch_size=settings.BATCH_SIZE,
    )

    class_counts = np.bincount(y_train)
    pos_weight = float(class_counts[0] / max(class_counts[1], 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPNetworkChurn(
        input_size=len(feature_names),
        hidden_dims=settings.MLP_HIDDEN_DIMS,
        dropout_rates=settings.MLP_DROPOUT_RATES,
    ).to(device)

    with mlflow.start_run(run_name="tabular_v2_training", tags=build_default_run_tags({"pipeline_version": "v2"})):
        mlflow.log_params(
            normalize_params(
                {
                    "model_type": "MLP",
                    "learning_rate": settings.LEARNING_RATE,
                    "pos_weight": pos_weight,
                    "max_epochs": settings.MAX_EPOCHS,
                    "early_stopping_patience": settings.EARLY_STOPPING_PATIENCE,
                    "batch_size": settings.BATCH_SIZE,
                    "random_seed": settings.RANDOM_SEED,
                    "input_features": len(feature_names),
                }
            )
        )

        trained_model, history, best_val_metrics, training_summary = train_mlp_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=settings.LEARNING_RATE,
            pos_weight=pos_weight,
            max_epochs=settings.MAX_EPOCHS,
            early_stopping_patience=settings.EARLY_STOPPING_PATIENCE,
            use_mlflow=False,
        )

        model_dir = settings.TABULAR_MLP_MODEL_PATH.parent
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(trained_model.state_dict(), settings.TABULAR_MLP_MODEL_PATH)
        import joblib

        joblib.dump(feature_names, settings.TABULAR_MODEL_FEATURE_NAMES_PATH)

        test_targets = y_test
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_test_transformed).float()),
            batch_size=settings.BATCH_SIZE,
            shuffle=False,
        )
        probabilities = []
        with torch.no_grad():
            for batch in test_loader:
                logits = trained_model(batch[0].to(device))
                probabilities.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        test_metrics = compute_classification_metrics(
            y_true=test_targets,
            y_pred_proba=np.asarray(probabilities),
        )

        metrics = {
            "test_roc_auc": test_metrics["roc_auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "best_val_roc_auc": best_val_metrics.get("roc_auc", 0.0),
            "best_epoch": training_summary["best_epoch"],
            "epochs_trained": training_summary["epochs_trained"],
            "input_features": len(feature_names),
        }
        _save_metrics(metrics, settings.TABULAR_MLP_METRICS_PATH)

        mlflow.log_metrics(
            {
                "test_roc_auc": metrics["test_roc_auc"],
                "test_accuracy": metrics["test_accuracy"],
                "test_precision": metrics["test_precision"],
                "test_recall": metrics["test_recall"],
                "test_f1": metrics["test_f1"],
                "best_val_roc_auc": metrics["best_val_roc_auc"],
                "best_epoch": float(metrics["best_epoch"]),
                "epochs_trained": float(metrics["epochs_trained"]),
            }
        )
        mlflow.log_dict(history, "training_history.json")
        mlflow.log_dict(metrics, "training_metrics.json")
        mlflow.log_artifact(str(settings.TABULAR_MLP_MODEL_PATH))
        mlflow.log_artifact(str(settings.TABULAR_PREPROCESSING_PIPELINE_PATH))
        mlflow.log_artifact(str(settings.TABULAR_MODEL_FEATURE_NAMES_PATH))
        mlflow.log_artifact(str(settings.TABULAR_MLP_METRICS_PATH))

    return metrics


def main() -> None:
    metrics = train_tabular_mlp_from_raw_dataset()
    print("Training completed")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
