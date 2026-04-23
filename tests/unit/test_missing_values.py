import pandas as pd
import pytest

from src.features.missing_values import clean_missing_values


def make_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Dependents": ["Yes", " ", "unknown", None],
            "Tenure Months": [12, None, "x", 0],
            "Monthly Charge": [50.0, "n/a", 80.0, None],
            "Total Charges": [600.0, None, "bad", 0],
            "Churn Value": [0, 1, 0, 1],
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
            "Monthly Charge",
            "Total Charges",
            "Churn Value",
        ],
    )

    assert "Extra Column" not in result.columns
    assert result.columns.tolist() == [
        "Dependents",
        "Tenure Months",
        "Monthly Charge",
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
                "Monthly Charge",
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
            "Monthly Charge",
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
            "Monthly Charge",
            "Total Charges",
            "Churn Value",
        ],
        categorical_features=["Dependents"],
        int_features=["Tenure Months", "Churn Value"],
        float_features=["Monthly Charge", "Total Charges"],
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
            "Monthly Charge",
            "Total Charges",
            "Churn Value",
        ],
        categorical_features=["Dependents"],
        int_features=["Tenure Months", "Churn Value"],
        float_features=["Monthly Charge", "Total Charges"],
    )

    assert result["Tenure Months"].tolist() == [12.0, 0.0, 0.0, 0.0]
    assert result["Monthly Charge"].tolist() == [50.0, 0.0, 80.0, 0.0]
    assert result["Total Charges"].tolist() == [600.0, 0.0, 0.0, 0.0]


def test_clean_missing_values_respects_custom_missing_markers() -> None:
    df = pd.DataFrame(
        {
            "Dependents": ["NULO", "sem_info", "Yes"],
            "Tenure Months": [1, 2, 3],
            "Monthly Charge": [10.0, 20.0, 30.0],
            "Total Charges": [100.0, 200.0, 300.0],
            "Churn Value": [0, 1, 0],
        }
    )

    result = clean_missing_values(
        df,
        selected_features=[
            "Dependents",
            "Tenure Months",
            "Monthly Charge",
            "Total Charges",
            "Churn Value",
        ],
        categorical_features=["Dependents"],
        int_features=["Tenure Months", "Churn Value"],
        float_features=["Monthly Charge", "Total Charges"],
        extra_missing_markers=("nulo", "sem_info"),
    )

    assert result["Dependents"].tolist() == ["missing", "missing", "Yes"]