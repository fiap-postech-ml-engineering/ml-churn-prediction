from __future__ import annotations

from pathlib import Path
import math

import joblib
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.schemas import PredictRequest
from src.config.settings import (
    RAW_CATEGORICAL_FEATURES,
    RAW_FLOAT_FEATURES,
    RAW_INT_FEATURES,
    TABULAR_RAW_FEATURES,
    SELECTED_FEATURES,
    TARGET_COLUMN,
)
from src.inference.feature_contract import (
    build_model_ready_inference_features,
    build_raw_inference_feature_names,
)
from src.inference.predict import load_model_artifacts
from src.inference.prepare_inference_data import prepare_inference_batch, run_inference
from src.data.preprocessing import select_tabular_raw_features
from src.inference.prepare_inference_data import run_inference


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATASET_PATH = ROOT_DIR / "data/raw/telco_customer_churn.csv"
PROCESSED_DATASET_PATH = (
    ROOT_DIR / "data/processed/telco_customer_churn_eda_pre-processed_encoded.csv"
)
MODEL_FEATURES_PATH = ROOT_DIR / "models/mlp/churn_mlp_input_features_v1.joblib"
CLIENT = TestClient(app)


class ConstantLogitModel(torch.nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = torch.tensor([[logit]], dtype=torch.float32)

    def forward(self, x):
        return self.logit.repeat(x.shape[0], 1).to(x.device)


def _build_api_feature_frame(raw_row: pd.Series) -> pd.DataFrame:
    raw_payload = raw_row.to_dict()
    df = pd.DataFrame([raw_payload])
    model_feature_names = joblib.load(MODEL_FEATURES_PATH)

    return build_model_ready_inference_features(
        df_raw_features=df,
        model_feature_names=model_feature_names,
        selected_raw_features=build_raw_inference_feature_names(
            selected_features=SELECTED_FEATURES,
            target_column=TARGET_COLUMN,
        ),
        raw_int_features=[
            feature for feature in RAW_INT_FEATURES if feature != TARGET_COLUMN
        ],
        raw_float_features=RAW_FLOAT_FEATURES,
        raw_categorical_features=RAW_CATEGORICAL_FEATURES,
    )


def test_predict_request_accepts_semantically_incomplete_payload() -> None:
    request = PredictRequest(features={"Dependents": "No"})

    assert request.features == {"Dependents": "No"}


def test_run_inference_threshold_changes_label_without_changing_probability() -> None:
    device = torch.device("cpu")
    x = torch.zeros((1, 3), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=1, shuffle=False)

    model = ConstantLogitModel(logit=torch.logit(torch.tensor(0.58)).item())

    probabilities_high, classes_high = run_inference(
        inference_loader=loader,
        model=model,
        device=device,
        approval_threshold=0.6,
    )
    probabilities_low, classes_low = run_inference(
        inference_loader=loader,
        model=model,
        device=device,
        approval_threshold=0.5,
    )

    assert pytest.approx(probabilities_high[0], rel=1e-5) == probabilities_low[0]
    assert classes_high.tolist() == [0]
    assert classes_low.tolist() == [1]


def test_same_sample_offline_vs_api_reveals_feature_space_divergence() -> None:
    raw_df = pd.read_csv(RAW_DATASET_PATH, sep=";")
    processed_df = pd.read_csv(PROCESSED_DATASET_PATH)

    raw_row = raw_df.iloc[0]
    offline_row = processed_df.iloc[0]
    api_ready = _build_api_feature_frame(raw_row)
    model_feature_names = joblib.load(MODEL_FEATURES_PATH)

    assert list(api_ready.columns) == model_feature_names
    assert api_ready.shape[1] == len(model_feature_names)

    zero_filled_columns = [
        column
        for column in [
            "Latitude",
            "Longitude",
            "CLTV",
            "Gender_Male",
            "Senior Citizen_Yes",
            "Partner_Yes",
        ]
        if column in api_ready.columns
    ]

    assert zero_filled_columns
    for column in zero_filled_columns:
        assert api_ready.loc[0, column] == 0.0

    assert offline_row["Gender_Male"] == 1


def test_api_contract_is_narrower_than_model_feature_space() -> None:
    model_feature_names = joblib.load(MODEL_FEATURES_PATH)
    api_raw_features = build_raw_inference_feature_names(
        selected_features=SELECTED_FEATURES,
        target_column=TARGET_COLUMN,
    )

    assert len(api_raw_features) < len(model_feature_names)
    assert "Gender" not in api_raw_features
    assert "Senior Citizen" not in api_raw_features
    assert "Partner" not in api_raw_features


def _json_safe_features(row: pd.Series) -> dict:
    return {
        key: (None if isinstance(value, float) and math.isnan(value) else value)
        for key, value in row.to_dict().items()
    }


def test_api_matches_offline_on_real_churn_sample() -> None:
    raw_df = pd.read_csv(RAW_DATASET_PATH, sep=";")
    row = raw_df.iloc[0]
    expected_target = int(row[TARGET_COLUMN])
    model_artifacts = load_model_artifacts()

    raw_features = select_tabular_raw_features(
        pd.DataFrame([row.drop(labels=[TARGET_COLUMN])]),
        require_target=False,
    )

    offline_loader = prepare_inference_batch(
        df_features=raw_features,
        scaler=model_artifacts.scaler,
        device=model_artifacts.device,
    )
    offline_probabilities, offline_classes = run_inference(
        inference_loader=offline_loader,
        model=model_artifacts.model,
        device=model_artifacts.device,
        approval_threshold=0.6,
    )

    response = CLIENT.post(
        "/predict",
        json={"features": _json_safe_features(row.drop(labels=[TARGET_COLUMN]))},
    )

    assert response.status_code == 200
    response_json = response.json()

    assert expected_target == 1
    assert response_json["predicao"]["classe"] == int(offline_classes[0])
    assert response_json["predicao"]["probabilidade_churn"] == pytest.approx(
        float(offline_probabilities[0]),
        rel=1e-6,
    )


def test_api_matches_offline_on_real_no_churn_sample() -> None:
    raw_df = pd.read_csv(RAW_DATASET_PATH, sep=";")
    row = raw_df.iloc[1869]
    model_artifacts = load_model_artifacts()

    raw_features = select_tabular_raw_features(
        pd.DataFrame([row.drop(labels=[TARGET_COLUMN])]),
        require_target=False,
    )
    offline_loader = prepare_inference_batch(
        df_features=raw_features,
        scaler=model_artifacts.scaler,
        device=model_artifacts.device,
    )
    offline_probabilities, offline_classes = run_inference(
        inference_loader=offline_loader,
        model=model_artifacts.model,
        device=model_artifacts.device,
        approval_threshold=0.6,
    )

    response = CLIENT.post(
        "/predict",
        json={"features": _json_safe_features(row.drop(labels=[TARGET_COLUMN]))},
    )

    assert response.status_code == 200
    response_json = response.json()

    assert int(row[TARGET_COLUMN]) == 0
    assert response_json["predicao"]["classe"] == int(offline_classes[0])
    assert response_json["predicao"]["probabilidade_churn"] == pytest.approx(
        float(offline_probabilities[0]),
        rel=1e-6,
    )


def test_api_rejects_incomplete_payload_with_missing_critical_feature() -> None:
    raw_df = pd.read_csv(RAW_DATASET_PATH, sep=";")
    row = raw_df.iloc[0].drop(labels=["Internet Service"])  # critical field

    response = CLIENT.post("/predict", json={"features": _json_safe_features(row)})

    assert response.status_code == 422
    assert "Missing critical raw inference features" in response.json()["detail"]


def test_real_sample_contract_still_mismatches_training_scaler_names() -> None:
    raw_df = pd.read_csv(RAW_DATASET_PATH, sep=";")
    row = raw_df.iloc[0]
    art = load_model_artifacts()

    scaler_feature_names = list(art.scaler.feature_names_in_)

    assert len(art.feature_names) == 39
    assert list(scaler_feature_names) == TABULAR_RAW_FEATURES
    assert "fiber_price_impact" in art.feature_names
    assert "total_services" in art.feature_names
    assert "Total Charges Log" in art.feature_names
    assert "avg_ticket_log" in art.feature_names
    assert "Total Charges Log" not in scaler_feature_names
