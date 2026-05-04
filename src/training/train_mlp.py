import json

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.config.settings import (
    APPROVAL_THRESHOLD,
    BATCH_SIZE,
    DATA_DIR,
    EARLY_STOPPING_PATIENCE,
    EXPERIMENT_NAME,
    EXPERIMENT_TAGS,
    LEARNING_RATE,
    MAX_EPOCHS,
    MLFLOW_ARTIFACTS_PATH,
    MLFLOW_TRACKING_PATH,
    MLP_DROPOUT_RATES,
    MLP_HIDDEN_DIMS,
    RANDOM_SEED,
    TABULAR_MLP_METRICS_PATH,
    TABULAR_MLP_MODEL_PATH,
    TABULAR_MODEL_FEATURE_NAMES_PATH,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.data.business_metrics import weighted_recall
from src.data.data_cleaning import clean_dataframe_for_modeling
from src.data.data_split import split_features_target, split_train_val_test
from src.data.feature_selection import select_tabular_raw_features
from src.data.tabular_pipeline import (
    load_tabular_preprocessing_pipeline,
    transform_tabular_features,
)
from src.models.mlp_model import MLPNetworkChurn
from src.tracking.mlflow_utils import build_default_run_tags, configure_mlflow_tracking

RAW_DATA_PATH = DATA_DIR / "raw" / "telco_customer_churn.csv"


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(RAW_DATA_PATH, sep=";")
    df = select_tabular_raw_features(df, require_target=True)
    df = clean_dataframe_for_modeling(df)
    return split_features_target(df)


def _extract_cltv(x: pd.DataFrame) -> pd.Series:
    if "CLTV" not in x.columns:
        raise ValueError("CLTV column not found in features.")
    return x["CLTV"].reset_index(drop=True)


def _drop_cltv_from_array(
    arrays: list[np.ndarray],
    feature_names: list[str],
) -> tuple[list[np.ndarray], list[str]]:
    if "CLTV" not in feature_names:
        return arrays, feature_names
    idx = feature_names.index("CLTV")
    trimmed = [np.delete(arr, idx, axis=1) for arr in arrays]
    names = [f for f in feature_names if f != "CLTV"]
    return trimmed, names


def _make_loaders(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32)).to(device)

    train_ds = TensorDataset(_to_tensor(x_train), _to_tensor(y_train))
    val_ds = TensorDataset(_to_tensor(x_val), _to_tensor(y_val))
    test_ds = TensorDataset(_to_tensor(x_test), _to_tensor(y_test))

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
    )


def _predict_proba(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for x_batch, _ in loader:
            logits = model(x_batch)
            probs.append(torch.sigmoid(logits).cpu().numpy().flatten())
    return np.concatenate(probs)


def _compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cltv: pd.Series,
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f2_score": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "weighted_recall": float(weighted_recall(y_true, y_pred, cltv) or 0.0),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _tune_threshold(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    cltv_val: pd.Series,
) -> float:
    best_thr = float(APPROVAL_THRESHOLD)
    best_wr = (
        weighted_recall(val_targets, (val_probs >= best_thr).astype(int), cltv_val)
        or 0.0
    )

    for thr in np.linspace(0.0, 1.0, 101):
        preds = (val_probs >= thr).astype(int)
        prec = precision_score(val_targets, preds, zero_division=0)
        wr = weighted_recall(val_targets, preds, cltv_val) or 0.0

        with mlflow.start_run(run_name=f"threshold_{thr:.2f}", nested=True):
            mlflow.log_param("threshold_candidate", float(thr))
            mlflow.log_metrics(
                {
                    "threshold_precision": float(prec),
                    "threshold_weighted_recall": float(wr),
                    "threshold_recall": float(
                        recall_score(val_targets, preds, zero_division=0)
                    ),
                    "threshold_f2": float(
                        fbeta_score(val_targets, preds, beta=2, zero_division=0)
                    ),
                }
            )

        if prec >= 0.5 and wr > best_wr:
            best_wr = wr
            best_thr = float(thr)

    return best_thr


def train() -> None:
    _set_seeds(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- dados ---
    x, y = _load_and_prepare_data()
    x_train_df, x_val_df, x_test_df, y_train_s, y_val_s, y_test_s = (
        split_train_val_test(x, y, test_size=TEST_SIZE, val_size=VALIDATION_SIZE)
    )

    # cltv_train = _extract_cltv(x_train_df)
    cltv_val = _extract_cltv(x_val_df)
    cltv_test = _extract_cltv(x_test_df)

    # --- pipeline sklearn ---
    artifact = load_tabular_preprocessing_pipeline()
    pipeline = artifact["preprocessing_pipeline"]
    feature_names: list[str] = list(artifact["feature_names"])

    x_train_t = transform_tabular_features(pipeline, x_train_df)
    x_val_t = transform_tabular_features(pipeline, x_val_df)
    x_test_t = transform_tabular_features(pipeline, x_test_df)

    [x_train_t, x_val_t, x_test_t], feature_names = _drop_cltv_from_array(
        [x_train_t, x_val_t, x_test_t], feature_names
    )

    y_train = y_train_s.values.astype(np.float32)
    y_val = y_val_s.values.astype(np.float32)
    y_test = y_test_s.values.astype(np.float32)

    train_loader, val_loader, test_loader = _make_loaders(
        x_train_t, x_val_t, x_test_t, y_train, y_val, y_test, device
    )

    n_features = x_train_t.shape[1]
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    # --- mlflow ---
    configure_mlflow_tracking(
        experiment_name=EXPERIMENT_NAME,
        db_path=MLFLOW_TRACKING_PATH,
        experiment_tags=EXPERIMENT_TAGS,
        artifact_root_path=MLFLOW_ARTIFACTS_PATH,
    )

    with mlflow.start_run(run_name="mlp_pytorch_training"):
        mlflow.set_tags(build_default_run_tags({"stage": "training"}))
        mlflow.log_params(
            {
                "model_type": "MLP",
                "input_features": n_features,
                "hidden_dims": str(MLP_HIDDEN_DIMS),
                "dropout_rates": str(MLP_DROPOUT_RATES),
                "loss_function": "BCEWithLogitsLoss",
                "learning_rate": LEARNING_RATE,
                "max_epochs": MAX_EPOCHS,
                "pos_weight": pos_weight,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "batch_size": BATCH_SIZE,
                "lr_scheduler": "ReduceLROnPlateau",
            }
        )

        model = MLPNetworkChurn(
            input_size=n_features,
            hidden_dims=list(MLP_HIDDEN_DIMS),
            dropout_rates=list(MLP_DROPOUT_RATES),
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        best_val_pr_auc = -np.inf
        best_model_state: dict | None = None
        best_epoch = 0
        no_improve = 0

        for epoch in range(MAX_EPOCHS):
            # treino
            model.train()
            train_losses = []
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(x_batch).view(-1), y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # validação
            val_probs = _predict_proba(model, val_loader)
            val_pr_auc = float(average_precision_score(y_val, val_probs))

            scheduler.step(val_pr_auc)

            mlflow.log_metrics(
                {
                    "train_loss": float(np.mean(train_losses)),
                    "val_pr_auc": val_pr_auc,
                    "val_roc_auc": float(roc_auc_score(y_val, val_probs)),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping na época {epoch + 1} (best epoch {best_epoch})")
                break

        if best_model_state is not None:
            model.load_state_dict(
                {k: v.to(device) for k, v in best_model_state.items()}
            )

        # threshold tuning no val set
        val_probs_final = _predict_proba(model, val_loader)
        threshold = _tune_threshold(val_probs_final, y_val, cltv_val)
        mlflow.log_param("threshold", threshold)

        # avaliação no test set
        test_probs = _predict_proba(model, test_loader)
        test_metrics = _compute_metrics(y_test, test_probs, threshold, cltv_test)
        test_metrics["best_epoch"] = best_epoch
        test_metrics["best_val_pr_auc"] = float(best_val_pr_auc)
        test_metrics["threshold"] = threshold

        mlflow.log_metrics(
            {f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, float)}
        )
        mlflow.pytorch.log_model(model, "model")

        # artefatos locais
        TABULAR_MLP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), TABULAR_MLP_MODEL_PATH)
        joblib.dump(feature_names, TABULAR_MODEL_FEATURE_NAMES_PATH)
        with open(TABULAR_MLP_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        print(f"Treino concluído — test PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(f"Weighted Recall: {test_metrics['weighted_recall']:.4f}")
        print(f"Threshold: {threshold:.3f}")


if __name__ == "__main__":
    train()
