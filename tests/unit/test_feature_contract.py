import pandas as pd
import pytest

from src.inference.feature_contract import (
    align_to_model_feature_contract,
    build_model_ready_inference_features,
    build_raw_inference_feature_names,
    ensure_feature_engineering_columns,
    ensure_model_one_hot_columns,
    validate_critical_raw_inference_features,
)


def test_build_raw_inference_feature_names_excludes_target() -> None:
    selected = ["A", "B", "target", "C"]

    result = build_raw_inference_feature_names(
        selected_features=selected,
        target_column="target",
    )

    assert result == ["A", "B", "C"]


def test_align_to_model_feature_contract_reorders_and_fills_missing() -> None:
    df = pd.DataFrame(
        [
            {
                "f2": 2,
                "f1": 1,
                "extra": 999,
            }
        ]
    )

    aligned = align_to_model_feature_contract(
        df_features=df,
        model_feature_names=["f1", "f2", "f3"],
    )

    assert list(aligned.columns) == ["f1", "f2", "f3"]
    assert aligned.loc[0, "f1"] == 1.0
    assert aligned.loc[0, "f2"] == 2.0
    assert aligned.loc[0, "f3"] == 0.0
    assert "extra" not in aligned.columns


def test_align_to_model_feature_contract_requires_non_empty_contract() -> None:
    with pytest.raises(ValueError, match="model_feature_names cannot be empty"):
        align_to_model_feature_contract(
            df_features=pd.DataFrame([{"f1": 1}]),
            model_feature_names=[],
        )


def test_ensure_feature_engineering_columns_from_raw_values() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "Internet Service": "Fiber optic",
                "Online Security": "Yes",
                "Online Backup": "No",
                "Device Protection": "Yes",
                "Tech Support": "No",
                "Streaming TV": "Yes",
                "Streaming Movies": "No",
            }
        ]
    )
    encoded_df = pd.DataFrame([{"Monthly Charges": 70.0, "Total Charges": 400.0}])

    result = ensure_feature_engineering_columns(
        df_encoded=encoded_df,
        df_raw_features=raw_df,
    )

    assert result.loc[0, "Internet Service_Fiber optic"] == 1
    assert result.loc[0, "Online Security_Yes"] == 1
    assert result.loc[0, "Online Backup_Yes"] == 0
    assert result.loc[0, "Device Protection_Yes"] == 1
    assert result.loc[0, "Tech Support_Yes"] == 0
    assert result.loc[0, "Streaming TV_Yes"] == 1
    assert result.loc[0, "Streaming Movies_Yes"] == 0

def test_ensure_model_one_hot_columns_rebuilds_model_columns_from_raw_values() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "Phone Service": "Yes",
                "Paperless Billing": "Yes",
                "Contract": "Month-to-month",
                "Payment Method": "Electronic check",
                "Internet Service": "Fiber optic",
                "Online Backup": "Yes",
            }
        ]
    )
    encoded_df = pd.DataFrame(
        [
            {
                "Tenure Months": 1,
                "Monthly Charges": 105.0,
                "Total Charges": 105.0,
            }
        ]
    )

    result = ensure_model_one_hot_columns(
        df_encoded=encoded_df,
        df_raw_features=raw_df,
        model_feature_names=[
            "Phone Service_Yes",
            "Paperless Billing_Yes",
            "Contract_One year",
            "Contract_Two year",
            "Payment Method_Credit card (automatic)",
            "Payment Method_Electronic check",
            "Payment Method_Mailed check",
            "Internet Service_Fiber optic",
            "Online Backup_Yes",
        ],
        categorical_features=[
            "Phone Service",
            "Paperless Billing",
            "Contract",
            "Payment Method",
            "Internet Service",
            "Online Backup",
        ],
    )

    assert result.loc[0, "Phone Service_Yes"] == 1
    assert result.loc[0, "Paperless Billing_Yes"] == 1
    assert result.loc[0, "Contract_One year"] == 0
    assert result.loc[0, "Contract_Two year"] == 0
    assert result.loc[0, "Payment Method_Credit card (automatic)"] == 0
    assert result.loc[0, "Payment Method_Electronic check"] == 1
    assert result.loc[0, "Payment Method_Mailed check"] == 0
    assert result.loc[0, "Internet Service_Fiber optic"] == 1
    assert result.loc[0, "Online Backup_Yes"] == 1


def test_validate_critical_raw_inference_features_requires_columns() -> None:
    raw_df = pd.DataFrame([
        {
            "Tenure Months": 12,
            "Monthly Charges": 50.0,
            "Total Charges": 600.0,
            "Internet Service": "Fiber optic",
        }
    ])

    with pytest.raises(ValueError, match="Missing critical raw inference features"):
        validate_critical_raw_inference_features(raw_df)


def test_build_model_ready_inference_features_aligns_contract() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "Dependents": "No",
                "Tenure Months": 12,
                "Phone Service": "Yes",
                "Multiple Lines": "No",
                "Internet Service": "Fiber optic",
                "Online Security": "Yes",
                "Online Backup": "No",
                "Device Protection": "Yes",
                "Tech Support": "No",
                "Streaming TV": "Yes",
                "Streaming Movies": "No",
                "Contract": "Month-to-month",
                "Paperless Billing": "Yes",
                "Payment Method": "Electronic check",
                "Monthly Charges": 85.0,
                "Total Charges": 1020.0,
            }
        ]
    )

    model_feature_names = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Internet Service_Fiber optic",
        "Online Security_Yes",
        "Online Backup_Yes",
        "Device Protection_Yes",
        "Tech Support_Yes",
        "Streaming TV_Yes",
        "Streaming Movies_Yes",
        "total_services",
        "fiber_price_impact",
        "avg_ticket",
        "Monthly Charges_log",
        "is_new_customer",
    ]

    result = build_model_ready_inference_features(
        df_raw_features=raw_df,
        model_feature_names=model_feature_names,
        selected_raw_features=[column for column in raw_df.columns],
        raw_int_features=["Tenure Months"],
        raw_float_features=["Monthly Charges", "Total Charges"],
        raw_categorical_features=[
            "Dependents",
            "Phone Service",
            "Multiple Lines",
            "Internet Service",
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Contract",
            "Paperless Billing",
            "Payment Method",
        ],
    )

    assert list(result.columns) == model_feature_names
    assert result.loc[0, "Internet Service_Fiber optic"] == 1.0
    assert result.loc[0, "Online Backup_Yes"] == 0.0
    assert result.loc[0, "total_services"] == 3.0
