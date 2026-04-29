import pandas as pd
import pytest

from src.features.missing_values import clean_missing_values


def make_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dependents": ["Yes", " ", "unknown", None],
            "Tenure Months": [12, None, "x", 0],
            "Monthly Charges": [50.0, "n/a", 80.0, None],
            "Total Charges": [600.0, None, "bad", 0],
            "Churn Value": [0, 1, 0, 1],
            "Phone Service": ["Yes", "No", "Yes", "Yes"],
            "Multiple Lines": ["No", "Yes", "No", "No"],
            "Internet Service": ["Fiber optic", "DSL", "Fiber optic", "No"],
            "Online Security": ["Yes", "No", "Yes", "No"],
            "Online Backup": ["Yes", "No", "Yes", "No"],
            "Device Protection": ["No", "Yes", "No", "No"],
            "Tech Support": ["Yes", "No", "Yes", "No"],
            "Streaming TV": ["Yes", "No", "Yes", "No"],
            "Streaming Movies": ["No", "Yes", "No", "No"],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
            "Paperless Billing": ["Yes", "No", "Yes", "No"],
            "Payment Method": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "Extra Column": ["keep", "out", "of", "scope"],
        }
    )


def test_clean_missing_values_selects_only_selected_features() -> None:
    df = make_input_df()

    result = clean_missing_values(
        df,
        selected_features=[
            "Dependents",
            "Tenure Months",
            "Monthly Charges",
            "Total Charges",
            "Churn Value",
        ],
        strict=False,
    )

    assert "Extra Column" not in result.columns
    assert result.columns.tolist() == [
        "Dependents",
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Churn Value",
    ]


def test_clean_missing_values_raises_when_selected_feature_is_missing_in_strict_mode() -> None:
    df = make_input_df().drop(columns=["Dependents"])

    with pytest.raises(ValueError, match="Missing selected features"):
        clean_missing_values(
            df,
            selected_features=[
                "Dependents",
                "Tenure Months",
                "Monthly Charges",
                "Total Charges",
                "Churn Value",
            ],
            strict=True,
        )


def test_clean_missing_values_allows_missing_selected_feature_in_non_strict_mode() -> None:
    df = make_input_df().drop(columns=["Dependents"])

    result = clean_missing_values(
        df,
        selected_features=[
            "Dependents",
            "Tenure Months",
            "Monthly Charges",
            "Total Charges",
            "Churn Value",
        ],
        strict=False,
    )

    assert "Dependents" not in result.columns
    assert "Tenure Months" in result.columns


def test_clean_missing_values_normalizes_and_imputes_categorical() -> None:
    df = make_input_df()

    result = clean_missing_values(
        df,
        selected_features=[
            "Dependents",
            "Tenure Months",
            "Monthly Charges",
            "Total Charges",
            "Churn Value",
        ],
        categorical_features=["Dependents"],
        int_features=["Tenure Months", "Churn Value"],
        float_features=["Monthly Charges", "Total Charges"],
        categorical_fill_value="missing",
    )

    assert result["Dependents"].tolist() == ["Yes", "missing", "missing", "missing"]


def test_clean_missing_values_imputes_numeric_with_zero_after_coercion() -> None:
    df = make_input_df()

    result = clean_missing_values(
        df,
        selected_features=[
            "Dependents",
            "Tenure Months",
            "Monthly Charges",
            "Total Charges",
            "Churn Value",
        ],
        categorical_features=["Dependents"],
        int_features=["Tenure Months", "Churn Value"],
        float_features=["Monthly Charges", "Total Charges"],
    )

    assert result["Tenure Months"].tolist() == [12.0, 0.0, 0.0, 0.0]
    assert result["Monthly Charges"].tolist() == [50.0, 0.0, 80.0, 0.0]
    assert result["Total Charges"].tolist() == [600.0, 0.0, 0.0, 0.0]


def test_clean_missing_values_respects_custom_missing_markers() -> None:
    df = pd.DataFrame(
        {
            "Dependents": ["NULO", "sem_info", "Yes"],
            "Tenure Months": [1, 2, 3],
            "Monthly Charges": [10.0, 20.0, 30.0],
            "Total Charges": [100.0, 200.0, 300.0],
            "Churn Value": [0, 1, 0],
        }
    )

    result = clean_missing_values(
        df,
        selected_features=[
            "Dependents",
            "Tenure Months",
            "Monthly Charges",
            "Total Charges",
            "Churn Value",
        ],
        categorical_features=["Dependents"],
        int_features=["Tenure Months", "Churn Value"],
        float_features=["Monthly Charges", "Total Charges"],
        extra_missing_markers=("nulo", "sem_info"),
    )

    assert result["Dependents"].tolist() == ["missing", "missing", "Yes"]
