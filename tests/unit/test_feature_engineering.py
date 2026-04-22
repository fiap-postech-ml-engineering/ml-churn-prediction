import math

from src.features.feature_engineering import apply_feature_engineering


def make_base_features() -> dict:
    return {
        "Tenure Months": 12,
        "Monthly Charges": 100.0,
        "Total Charges": 1200.0,
        "Internet Service_Fiber optic": 1,
        "Online Security_Yes": 1,
        "Online Backup_Yes": 1,
        "Device Protection_Yes": 0,
        "Tech Support_Yes": 1,
        "Streaming TV_Yes": 0,
        "Streaming Movies_Yes": 1,
    }


def test_apply_feature_engineering_returns_original_and_derived_features() -> None:
    features = make_base_features()

    result = apply_feature_engineering(features)

    assert "Tenure Months" in result
    assert "Monthly Charges" in result
    assert "Total Charges" in result

    assert "total_services" in result
    assert "fiber_price_impact" in result
    assert "Total Charges Log" in result
    assert "avg_ticket" in result
    assert "avg_ticket_log" in result
    assert "is_new_customer" in result


def test_apply_feature_engineering_computes_total_services_correctly() -> None:
    features = make_base_features()

    result = apply_feature_engineering(features)

    assert result["total_services"] == 4.0


def test_apply_feature_engineering_computes_fiber_price_impact_correctly() -> None:
    features = make_base_features()

    result = apply_feature_engineering(features)

    assert result["fiber_price_impact"] == 100.0


def test_apply_feature_engineering_computes_log_and_avg_ticket_correctly() -> None:
    features = make_base_features()

    result = apply_feature_engineering(features)

    assert math.isclose(result["Total Charges Log"], math.log1p(1200.0))
    assert math.isclose(result["avg_ticket"], 100.0)
    assert math.isclose(result["avg_ticket_log"], math.log1p(100.0))


def test_apply_feature_engineering_sets_new_customer_flag_for_short_tenure() -> None:
    features = make_base_features()
    features["Tenure Months"] = 3

    result = apply_feature_engineering(features)

    assert result["is_new_customer"] == 1.0


def test_apply_feature_engineering_sets_new_customer_flag_for_long_tenure() -> None:
    features = make_base_features()
    features["Tenure Months"] = 12

    result = apply_feature_engineering(features)

    assert result["is_new_customer"] == 0.0


def test_apply_feature_engineering_protects_against_zero_tenure_division() -> None:
    features = make_base_features()
    features["Tenure Months"] = 0
    features["Total Charges"] = 300.0

    result = apply_feature_engineering(features)

    assert result["avg_ticket"] == 300.0
    assert math.isclose(result["avg_ticket_log"], math.log1p(300.0))
    assert result["is_new_customer"] == 1.0


def test_apply_feature_engineering_uses_defaults_when_fields_are_missing() -> None:
    result = apply_feature_engineering({})

    assert result["total_services"] == 0.0
    assert result["fiber_price_impact"] == 0.0
    assert result["Total Charges Log"] == 0.0
    assert result["avg_ticket"] == 0.0
    assert result["avg_ticket_log"] == 0.0
    assert result["is_new_customer"] == 1.0
