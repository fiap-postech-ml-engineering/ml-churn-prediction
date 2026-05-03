"""
Testes unitários para a rota de predição da API.
"""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import HTTPException

from src.api.routes.predict import predict
from src.api.schemas import ChurnRequest, ChurnResponse


class TestPredictRoute:
    """Testes para a rota /predict."""

    @pytest.fixture
    def sample_request(self):
        """Cria um request de exemplo."""
        return ChurnRequest(
            features={
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 45.65,
                "TotalCharges": 538.20,
            }
        )

    @patch("src.api.routes.predict.model", None)
    @patch("src.api.routes.predict.logger")
    def test_predict_model_not_loaded(self, mock_logger, sample_request):
        """Verifica erro quando modelo não está carregado."""
        with pytest.raises(HTTPException) as exc_info:
            predict(sample_request)

        assert exc_info.value.status_code == 503
        assert "Modelo não está disponível" in exc_info.value.detail
        mock_logger.error.assert_called_with("❌ ERRO: Modelo não foi carregado")

    @patch("src.api.routes.predict.model")
    @patch("src.api.routes.predict.scaler")
    @patch("src.api.routes.predict.device")
    @patch("src.api.routes.predict.select_tabular_raw_features")
    @patch("src.api.routes.predict.prepare_inference_batch")
    @patch("src.api.routes.predict.run_inference")
    @patch("src.api.routes.predict.logger")
    def test_predict_success(
        self,
        mock_logger,
        mock_run_inference,
        mock_prepare_batch,
        mock_select_features,
        mock_device,
        mock_scaler,
        mock_model,
        sample_request,
    ):
        """Verifica predição bem-sucedida."""
        # Setup mocks
        mock_model.__bool__ = lambda: True
        mock_model.__nonzero__ = lambda: True

        mock_df = pd.DataFrame([sample_request.features])
        mock_select_features.return_value = mock_df

        mock_dataloader = MagicMock()
        mock_prepare_batch.return_value = mock_dataloader

        mock_run_inference.return_value = ([0.7], [1])

        response = predict(sample_request)

        assert isinstance(response, ChurnResponse)
        assert response.sucesso is True
        assert response.predicao.classe == 1
        assert response.predicao.classe_descricao == "Churn"
        assert response.predicao.probabilidade_churn == 0.7
        assert response.entrada_recebida == sample_request.features

    @patch("src.api.routes.predict.model")
    @patch("src.api.routes.predict.scaler")
    @patch("src.api.routes.predict.device")
    @patch("src.api.routes.predict.select_tabular_raw_features")
    @patch("src.api.routes.predict.logger")
    def test_predict_select_features_error(
        self,
        mock_logger,
        mock_select_features,
        mock_device,
        mock_scaler,
        mock_model,
        sample_request,
    ):
        """Verifica tratamento de erro na seleção de features."""
        mock_model.__bool__ = lambda: True
        mock_model.__nonzero__ = lambda: True

        mock_select_features.side_effect = Exception("Feature selection error")

        with pytest.raises(HTTPException) as exc_info:
            predict(sample_request)

        assert exc_info.value.status_code == 500
        assert "Erro ao processar predição" in exc_info.value.detail
        mock_logger.exception.assert_called_with(
            "❌ Falha no passo 'select_tabular_raw_features'"
        )

    @patch("src.api.routes.predict.model")
    @patch("src.api.routes.predict.scaler")
    @patch("src.api.routes.predict.device")
    @patch("src.api.routes.predict.select_tabular_raw_features")
    @patch("src.api.routes.predict.prepare_inference_batch")
    @patch("src.api.routes.predict.logger")
    def test_predict_prepare_batch_error(
        self,
        mock_logger,
        mock_prepare_batch,
        mock_select_features,
        mock_device,
        mock_scaler,
        mock_model,
        sample_request,
    ):
        """Verifica tratamento de erro na preparação do batch."""
        mock_model.__bool__ = lambda: True
        mock_model.__nonzero__ = lambda: True

        mock_df = pd.DataFrame([sample_request.features])
        mock_select_features.return_value = mock_df

        mock_prepare_batch.side_effect = Exception("Batch preparation error")

        with pytest.raises(HTTPException) as exc_info:
            predict(sample_request)

        assert exc_info.value.status_code == 500
        assert "Erro ao processar predição" in exc_info.value.detail
        mock_logger.exception.assert_called_with(
            "❌ Falha no passo 'prepare_inference_batch'"
        )

    @patch("src.api.routes.predict.model")
    @patch("src.api.routes.predict.scaler")
    @patch("src.api.routes.predict.device")
    @patch("src.api.routes.predict.select_tabular_raw_features")
    @patch("src.api.routes.predict.prepare_inference_batch")
    @patch("src.api.routes.predict.run_inference")
    @patch("src.api.routes.predict.logger")
    def test_predict_inference_error(
        self,
        mock_logger,
        mock_run_inference,
        mock_prepare_batch,
        mock_select_features,
        mock_device,
        mock_scaler,
        mock_model,
        sample_request,
    ):
        """Verifica tratamento de erro na inferência."""
        mock_model.__bool__ = lambda: True
        mock_model.__nonzero__ = lambda: True

        mock_df = pd.DataFrame([sample_request.features])
        mock_select_features.return_value = mock_df

        mock_dataloader = MagicMock()
        mock_prepare_batch.return_value = mock_dataloader

        mock_run_inference.side_effect = Exception("Inference error")

        with pytest.raises(HTTPException) as exc_info:
            predict(sample_request)

        assert exc_info.value.status_code == 500
        assert "Erro ao processar predição" in exc_info.value.detail
        mock_logger.exception.assert_called_with("❌ Falha no passo 'run_inference'")

    @patch("src.api.routes.predict.model")
    @patch("src.api.routes.predict.scaler")
    @patch("src.api.routes.predict.device")
    @patch("src.api.routes.predict.select_tabular_raw_features")
    @patch("src.api.routes.predict.prepare_inference_batch")
    @patch("src.api.routes.predict.run_inference")
    @patch("src.api.routes.predict.logger")
    def test_predict_value_error_payload(
        self,
        mock_logger,
        mock_run_inference,
        mock_prepare_batch,
        mock_select_features,
        mock_device,
        mock_scaler,
        mock_model,
        sample_request,
    ):
        """Verifica tratamento de ValueError (payload inválido)."""
        mock_model.__bool__ = lambda: True
        mock_model.__nonzero__ = lambda: True

        mock_df = pd.DataFrame([sample_request.features])
        mock_select_features.return_value = mock_df

        mock_dataloader = MagicMock()
        mock_prepare_batch.return_value = mock_dataloader

        mock_run_inference.side_effect = ValueError("Invalid input data")

        with pytest.raises(HTTPException) as exc_info:
            predict(sample_request)

        assert exc_info.value.status_code == 422
        assert "Invalid input data" in exc_info.value.detail
        mock_logger.warning.assert_called_with(
            "⚠️ Payload inválido para inferência: Invalid input data"
        )

    @patch("src.api.routes.predict.model")
    @patch("src.api.routes.predict.scaler")
    @patch("src.api.routes.predict.device")
    @patch("src.api.routes.predict.select_tabular_raw_features")
    @patch("src.api.routes.predict.prepare_inference_batch")
    @patch("src.api.routes.predict.run_inference")
    @patch("src.api.routes.predict.logger")
    def test_predict_unexpected_error(
        self,
        mock_logger,
        mock_run_inference,
        mock_prepare_batch,
        mock_select_features,
        mock_device,
        mock_scaler,
        mock_model,
        sample_request,
    ):
        """Verifica tratamento de erro inesperado."""
        mock_model.__bool__ = lambda: True
        mock_model.__nonzero__ = lambda: True

        mock_df = pd.DataFrame([sample_request.features])
        mock_select_features.return_value = mock_df

        mock_dataloader = MagicMock()
        mock_prepare_batch.return_value = mock_dataloader

        mock_run_inference.side_effect = Exception("Unexpected error")

        with pytest.raises(HTTPException) as exc_info:
            predict(sample_request)

        assert exc_info.value.status_code == 500
        assert "Erro ao processar predição" in exc_info.value.detail
        assert any(
            call_args[0][0].startswith("❌ ERRO durante predição:")
            for call_args in mock_logger.error.call_args_list
        )

    @patch("src.api.routes.predict.model")
    @patch("src.api.routes.predict.scaler")
    @patch("src.api.routes.predict.device")
    @patch("src.api.routes.predict.select_tabular_raw_features")
    @patch("src.api.routes.predict.prepare_inference_batch")
    @patch("src.api.routes.predict.run_inference")
    @patch("src.api.routes.predict.logger")
    def test_predict_no_churn_prediction(
        self,
        mock_logger,
        mock_run_inference,
        mock_prepare_batch,
        mock_select_features,
        mock_device,
        mock_scaler,
        mock_model,
        sample_request,
    ):
        """Verifica predição de não churn."""
        mock_model.__bool__ = lambda: True
        mock_model.__nonzero__ = lambda: True

        mock_df = pd.DataFrame([sample_request.features])
        mock_select_features.return_value = mock_df

        mock_dataloader = MagicMock()
        mock_prepare_batch.return_value = mock_dataloader

        # Probabilidade baixa = não churn
        mock_run_inference.return_value = ([0.3], [0])

        response = predict(sample_request)

        assert response.predicao.classe == 0
        assert response.predicao.classe_descricao == "Não Churn"
        assert response.predicao.probabilidade_churn == 0.3

    @patch("src.api.routes.predict.model")
    @patch("src.api.routes.predict.scaler")
    @patch("src.api.routes.predict.device")
    @patch("src.api.routes.predict.select_tabular_raw_features")
    @patch("src.api.routes.predict.prepare_inference_batch")
    @patch("src.api.routes.predict.run_inference")
    @patch("src.api.routes.predict.logger")
    def test_predict_boundary_probability(
        self,
        mock_logger,
        mock_run_inference,
        mock_prepare_batch,
        mock_select_features,
        mock_device,
        mock_scaler,
        mock_model,
        sample_request,
    ):
        """Verifica predição com probabilidade de fronteira (0.5)."""
        mock_model.__bool__ = lambda: True
        mock_model.__nonzero__ = lambda: True

        mock_df = pd.DataFrame([sample_request.features])
        mock_select_features.return_value = mock_df

        mock_dataloader = MagicMock()
        mock_prepare_batch.return_value = mock_dataloader

        # Probabilidade exatamente 0.5
        mock_run_inference.return_value = (
            [0.5],
            [1],
        )  # Classe 1 por padrão no threshold 0.5

        response = predict(sample_request)

        assert response.predicao.classe == 1
        assert response.predicao.probabilidade_churn == 0.5

    @patch("src.api.routes.predict.model")
    @patch("src.api.routes.predict.scaler")
    @patch("src.api.routes.predict.device")
    @patch("src.api.routes.predict.select_tabular_raw_features")
    @patch("src.api.routes.predict.prepare_inference_batch")
    @patch("src.api.routes.predict.run_inference")
    @patch("src.api.routes.predict.logger")
    def test_predict_empty_features(
        self,
        mock_logger,
        mock_run_inference,
        mock_prepare_batch,
        mock_select_features,
        mock_device,
        mock_scaler,
        mock_model,
    ):
        """Verifica tratamento com features vazias."""
        mock_model.__bool__ = lambda: True
        mock_model.__nonzero__ = lambda: True

        request = ChurnRequest(features={})

        mock_df = pd.DataFrame([{}])
        mock_select_features.return_value = mock_df

        mock_dataloader = MagicMock()
        mock_prepare_batch.return_value = mock_dataloader

        mock_run_inference.return_value = ([0.1], [0])

        response = predict(request)

        assert response.sucesso is True
        assert response.entrada_recebida == {}
