from src.api.routes.health import health_check


def test_health_check_returns_operational_status_with_components() -> None:
    result = health_check(
        model=object(),
        scaler=object(),
        feature_names=["f1", "f2"],
        device="cpu",
    )

    assert result["api_status"] == "operacional"
    assert "timestamp" in result

    componentes = result["componentes"]
    assert componentes["modelo_carregado"] is True
    assert componentes["scaler_carregado"] is True
    assert componentes["features_carregadas"] is True
    assert componentes["device"] == "cpu"


def test_health_check_returns_warning_when_model_is_missing() -> None:
    result = health_check(
        model=None,
        scaler=None,
        feature_names=None,
        device="cpu",
    )

    assert result["api_status"] == "operacional"
    assert result["componentes"]["modelo_carregado"] is False
    assert result["componentes"]["scaler_carregado"] is False
    assert result["componentes"]["features_carregadas"] is False
    assert result["componentes"]["device"] == "cpu"
    assert "aviso" in result
