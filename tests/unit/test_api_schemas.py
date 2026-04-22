from src.api.schemas import (
    ChurnPrediction,
    ChurnRequest,
    ChurnResponse,
    PredictRequest,
    PredictResponse,
)


def test_predict_request_accepts_features_dict() -> None:
    payload = {
        "features": {
            "Latitude": 47.0,
            "Longitude": -122.0,
            "Tenure Months": 12.0,
            "Monthly Charges": 65.0,
            "Total Charges": 780.0,
        }
    }

    request = PredictRequest(**payload)

    assert isinstance(request.features, dict)
    assert request.features["Latitude"] == 47.0
    assert request.features["Tenure Months"] == 12.0


def test_churn_prediction_model_builds_correctly() -> None:
    prediction = ChurnPrediction(
        classe=1,
        classe_descricao="Churn Detectado",
        probabilidade_churn=0.87,
    )

    assert prediction.classe == 1
    assert prediction.classe_descricao == "Churn Detectado"
    assert prediction.probabilidade_churn == 0.87


def test_predict_response_builds_correctly() -> None:
    prediction = ChurnPrediction(
        classe=0,
        classe_descricao="Sem Churn",
        probabilidade_churn=0.12,
    )

    response = PredictResponse(
        sucesso=True,
        predicao=prediction,
        entrada_recebida={"Tenure Months": 12.0},
    )

    assert response.sucesso is True
    assert response.predicao.classe == 0
    assert response.entrada_recebida["Tenure Months"] == 12.0


def test_aliases_keep_compatibility() -> None:
    assert ChurnRequest is PredictRequest
    assert ChurnResponse is PredictResponse
