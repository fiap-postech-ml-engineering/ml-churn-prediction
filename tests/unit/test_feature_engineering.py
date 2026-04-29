import math

import pandas as pd
import pytest

from src.config.settings import (
    TABULAR_DERIVED_FEATURES,
    TABULAR_RAW_CATEGORICAL_FEATURES,
    TABULAR_RAW_FEATURES,
    TABULAR_RAW_NUMERIC_FEATURES,
)
from src.features.feature_engineering import TabularFeatureEngineer


def make_base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Latitude": [0.0],
            "Longitude": [0.0],
            "Tenure Months": [12],
            "Monthly Charges": [100.0],
            "Total Charges": [1200.0],
            "CLTV": [2500],
            "Gender": ["Female"],
            "Senior Citizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "Phone Service": ["Yes"],
            "Multiple Lines": ["Yes"],
            "Internet Service": ["Fiber optic"],
            "Online Security": ["Yes"],
            "Online Backup": ["Yes"],
            "Device Protection": ["No"],
            "Tech Support": ["Yes"],
            "Streaming TV": ["No"],
            "Streaming Movies": ["Yes"],
            "Contract": ["Month-to-month"],
            "Paperless Billing": ["Yes"],
            "Payment Method": ["Electronic check"],
        }
    )


def make_engineer() -> TabularFeatureEngineer:
    engineer = TabularFeatureEngineer()
    engineer.fit(make_base_frame())
    return engineer


def test_transform_returns_raw_and_derived_features() -> None:
    result = make_engineer().transform(make_base_frame())

    expected_columns = (
        list(TABULAR_RAW_NUMERIC_FEATURES)
        + list(TABULAR_RAW_CATEGORICAL_FEATURES)
        + list(TABULAR_DERIVED_FEATURES)
    )

    assert list(result.columns) == expected_columns
    for column in expected_columns:
        assert column in result.columns


def test_transform_computes_total_services_correctly() -> None:
    result = make_engineer().transform(make_base_frame())

    assert result.loc[0, "total_services"] == 4


def test_transform_computes_fiber_price_impact_correctly() -> None:
    result = make_engineer().transform(make_base_frame())

    assert result.loc[0, "fiber_price_impact"] == 100.0


def test_transform_computes_log_and_avg_ticket_correctly() -> None:
    result = make_engineer().transform(make_base_frame())

    assert math.isclose(result.loc[0, "Total Charges Log"], math.log1p(1200.0))
    assert math.isclose(result.loc[0, "avg_ticket"], 100.0)
    assert math.isclose(result.loc[0, "avg_ticket_log"], math.log1p(100.0))


def test_transform_sets_new_customer_flag_for_short_tenure() -> None:
    frame = make_base_frame()
    frame.loc[0, "Tenure Months"] = 3

    result = make_engineer().transform(frame)

    assert result.loc[0, "is_new_customer"] == 1


def test_transform_sets_new_customer_flag_for_long_tenure() -> None:
    result = make_engineer().transform(make_base_frame())

    assert result.loc[0, "is_new_customer"] == 0


def test_transform_protects_against_zero_tenure_division() -> None:
    frame = make_base_frame()
    frame.loc[0, "Tenure Months"] = 0
    frame.loc[0, "Total Charges"] = 300.0

    result = make_engineer().transform(frame)

    assert result.loc[0, "avg_ticket"] == 300.0
    assert math.isclose(result.loc[0, "avg_ticket_log"], math.log1p(300.0))
    assert result.loc[0, "is_new_customer"] == 1


def test_transform_raises_when_required_raw_columns_are_missing() -> None:
    frame = make_base_frame().drop(columns=["Internet Service"])

    with pytest.raises(ValueError, match="Missing required raw columns"):
        make_engineer().transform(frame)
