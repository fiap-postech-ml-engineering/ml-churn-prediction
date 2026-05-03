"""
Testes unitários para o módulo predict.py.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch

from src.inference.predict import ModelArtifacts, load_model_artifacts
from src.models.mlp_model import MLPNetworkChurn


class TestModelArtifacts:
    """Testes para a classe ModelArtifacts."""

    def test_model_artifacts_init(self):
        """Verifica inicialização da classe ModelArtifacts."""
        model = MagicMock()
        scaler = MagicMock()
        feature_names = ["feature1", "feature2"]
        model_metrics = {"accuracy": 0.95}
        device = torch.device("cpu")

        artifacts = ModelArtifacts(
            model=model,
            scaler=scaler,
            feature_names=feature_names,
            model_metrics=model_metrics,
            device=device,
        )

        assert artifacts.model == model
        assert artifacts.scaler == scaler
        assert artifacts.feature_names == feature_names
        assert artifacts.model_metrics == model_metrics
        assert artifacts.device == device

    def test_model_artifacts_init_defaults(self):
        """Verifica inicialização com valores padrão."""
        artifacts = ModelArtifacts()

        assert artifacts.model is None
        assert artifacts.scaler is None
        assert artifacts.feature_names is None
        assert artifacts.model_metrics is None
        assert artifacts.device is None


class TestLoadModelArtifacts:
    """Testes para a função load_model_artifacts."""

    @patch("src.inference.predict.try_load_tabular_preprocessing_pipeline")
    @patch("src.inference.predict.try_load_preprocessing_pipeline")
    @patch("src.inference.predict.torch.load")
    @patch("src.inference.predict.joblib.load")
    @patch("src.inference.predict.json.load")
    @patch("src.inference.predict.MLPNetworkChurn")
    @patch("src.inference.predict.torch.device")
    @patch("src.inference.predict.logger")
    def test_load_model_artifacts_success_tabular_pipeline(
        self,
        mock_logger,
        mock_device,
        mock_mlp_class,
        mock_json_load,
        mock_joblib_load,
        mock_torch_load,
        mock_try_load_preprocessing,
        mock_try_load_tabular,
    ):
        """Verifica carregamento bem-sucedido com pipeline tabular."""
        # Setup mocks
        mock_device.return_value = torch.device("cpu")
        mock_try_load_tabular.return_value = {
            "preprocessing_pipeline": MagicMock(),
            "feature_names": ["feature1", "feature2"],
        }
        mock_try_load_preprocessing.return_value = None

        mock_metrics = {"ROC-AUC": 0.85}
        mock_json_load.return_value = mock_metrics

        mock_model_instance = MagicMock()
        mock_mlp_class.return_value = mock_model_instance

        # Mock file existence
        with patch("pathlib.Path.exists", return_value=True):
            artifacts = load_model_artifacts()

        assert artifacts.scaler is not None
        assert artifacts.feature_names == ["feature1", "feature2"]
        assert artifacts.model_metrics == mock_metrics
        assert artifacts.device == torch.device("cpu")
        mock_mlp_class.assert_called_once_with(input_size=2)
        mock_model_instance.load_state_dict.assert_called_once()
        mock_model_instance.to.assert_called_once()
        mock_model_instance.eval.assert_called_once()

    @patch("src.inference.predict.try_load_tabular_preprocessing_pipeline")
    @patch("src.inference.predict.try_load_preprocessing_pipeline")
    @patch("src.inference.predict.torch.load")
    @patch("src.inference.predict.joblib.load")
    @patch("src.inference.predict.json.load")
    @patch("src.inference.predict.MLPNetworkChurn")
    @patch("src.inference.predict.torch.device")
    @patch("src.inference.predict.logger")
    def test_load_model_artifacts_success_legacy_pipeline(
        self,
        mock_logger,
        mock_device,
        mock_mlp_class,
        mock_json_load,
        mock_joblib_load,
        mock_torch_load,
        mock_try_load_preprocessing,
        mock_try_load_tabular,
    ):
        """Verifica carregamento bem-sucedido com pipeline legado."""
        # Setup mocks
        mock_device.return_value = torch.device("cpu")
        mock_try_load_tabular.return_value = None
        mock_try_load_preprocessing.return_value = {
            "scaler": MagicMock(),
            "feature_names": ["feature1", "feature2", "feature3"],
        }

        mock_metrics = {"accuracy": 0.92}
        mock_json_load.return_value = mock_metrics

        mock_model_instance = MagicMock()
        mock_mlp_class.return_value = mock_model_instance

        # Mock file existence
        with patch("pathlib.Path.exists", return_value=True):
            artifacts = load_model_artifacts()

        assert artifacts.scaler is not None
        assert len(artifacts.feature_names) == 3
        assert artifacts.model_metrics == mock_metrics
        mock_mlp_class.assert_called_once_with(input_size=3)

    @patch("src.inference.predict.try_load_tabular_preprocessing_pipeline")
    @patch("src.inference.predict.try_load_preprocessing_pipeline")
    @patch("src.inference.predict.torch.load")
    @patch("src.inference.predict.joblib.load")
    @patch("src.inference.predict.json.load")
    @patch("src.inference.predict.MLPNetworkChurn")
    @patch("src.inference.predict.torch.device")
    @patch("src.inference.predict.logger")
    def test_load_model_artifacts_success_fallback_artifacts(
        self,
        mock_logger,
        mock_device,
        mock_mlp_class,
        mock_json_load,
        mock_joblib_load,
        mock_torch_load,
        mock_try_load_preprocessing,
        mock_try_load_tabular,
    ):
        """Verifica carregamento bem-sucedido com artefatos separados (fallback)."""
        # Setup mocks
        mock_device.return_value = torch.device("cpu")
        mock_try_load_tabular.return_value = None
        mock_try_load_preprocessing.return_value = None

        mock_scaler = MagicMock()
        mock_features = ["feature1", "feature2"]
        mock_joblib_load.side_effect = [mock_scaler, mock_features]

        mock_metrics = {"f1_score": 0.88}
        mock_json_load.return_value = mock_metrics

        mock_model_instance = MagicMock()
        mock_mlp_class.return_value = mock_model_instance

        # Mock file existence
        with patch("pathlib.Path.exists", return_value=True):
            artifacts = load_model_artifacts()

        assert artifacts.scaler == mock_scaler
        assert artifacts.feature_names == mock_features
        assert artifacts.model_metrics == mock_metrics
        # Verificar que joblib.load foi chamado para scaler e features
        assert mock_joblib_load.call_count == 2

    @patch("src.inference.predict.try_load_tabular_preprocessing_pipeline")
    @patch("src.inference.predict.try_load_preprocessing_pipeline")
    @patch("src.inference.predict.torch.device")
    @patch("src.inference.predict.logger")
    def test_load_model_artifacts_scaler_not_found(
        self,
        mock_logger,
        mock_device,
        mock_try_load_preprocessing,
        mock_try_load_tabular,
    ):
        """Verifica tratamento quando scaler não é encontrado."""
        mock_device.return_value = torch.device("cpu")
        mock_try_load_tabular.return_value = None
        mock_try_load_preprocessing.return_value = None

        # Mock scaler path não existe
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            artifacts = load_model_artifacts()

        assert artifacts.scaler is None
        assert artifacts.model is None
        mock_logger.error.assert_called()

    @patch("src.inference.predict.try_load_tabular_preprocessing_pipeline")
    @patch("src.inference.predict.try_load_preprocessing_pipeline")
    @patch("src.inference.predict.torch.device")
    @patch("src.inference.predict.logger")
    def test_load_model_artifacts_features_not_found(
        self,
        mock_logger,
        mock_device,
        mock_try_load_preprocessing,
        mock_try_load_tabular,
    ):
        """Verifica tratamento quando features não são encontradas."""
        mock_device.return_value = torch.device("cpu")
        mock_try_load_tabular.return_value = None
        mock_try_load_preprocessing.return_value = None

        # Mock apenas scaler existe
        def exists_side_effect(path):
            return "scaler" in str(path)

        with patch("pathlib.Path.exists", side_effect=exists_side_effect):
            artifacts = load_model_artifacts()

        assert artifacts.feature_names is None
        assert artifacts.model is None
        mock_logger.error.assert_called()

    @patch("src.inference.predict.try_load_tabular_preprocessing_pipeline")
    @patch("src.inference.predict.try_load_preprocessing_pipeline")
    @patch("src.inference.predict.torch.load")
    @patch("src.inference.predict.joblib.load")
    @patch("src.inference.predict.MLPNetworkChurn")
    @patch("src.inference.predict.torch.device")
    @patch("src.inference.predict.logger")
    def test_load_model_artifacts_model_not_found(
        self,
        mock_logger,
        mock_device,
        mock_mlp_class,
        mock_joblib_load,
        mock_torch_load,
        mock_try_load_preprocessing,
        mock_try_load_tabular,
    ):
        """Verifica tratamento quando modelo não é encontrado."""
        mock_device.return_value = torch.device("cpu")
        mock_try_load_tabular.return_value = {
            "preprocessing_pipeline": MagicMock(),
            "feature_names": ["feature1"],
        }
        mock_try_load_preprocessing.return_value = None

        mock_scaler = MagicMock()
        mock_joblib_load.side_effect = [mock_scaler, ["feature1"]]

        # Mock apenas scaler e features existem
        def exists_side_effect(path):
            path_str = str(path)
            return "scaler" in path_str or "features" in path_str or "metrics" in path_str

        with patch("pathlib.Path.exists", side_effect=exists_side_effect):
            artifacts = load_model_artifacts()

        assert artifacts.model is None
        mock_logger.error.assert_called()

    @patch("src.inference.predict.try_load_tabular_preprocessing_pipeline")
    @patch("src.inference.predict.try_load_preprocessing_pipeline")
    @patch("src.inference.predict.torch.load")
    @patch("src.inference.predict.joblib.load")
    @patch("src.inference.predict.json.load")
    @patch("src.inference.predict.MLPNetworkChurn")
    @patch("src.inference.predict.torch.device")
    @patch("src.inference.predict.logger")
    def test_load_model_artifacts_cuda_device(
        self,
        mock_logger,
        mock_device,
        mock_mlp_class,
        mock_json_load,
        mock_joblib_load,
        mock_torch_load,
        mock_try_load_preprocessing,
        mock_try_load_tabular,
    ):
        """Verifica carregamento com dispositivo CUDA."""
        mock_device.return_value = torch.device("cuda")
        mock_try_load_tabular.return_value = {
            "preprocessing_pipeline": MagicMock(),
            "feature_names": ["feature1"],
        }
        mock_try_load_preprocessing.return_value = None

        mock_scaler = MagicMock()
        mock_joblib_load.side_effect = [mock_scaler, ["feature1"]]

        mock_metrics = {"recall": 0.87}
        mock_json_load.return_value = mock_metrics

        mock_model_instance = MagicMock()
        mock_mlp_class.return_value = mock_model_instance

        with patch("pathlib.Path.exists", return_value=True):
            artifacts = load_model_artifacts()

        assert artifacts.device == torch.device("cuda")
        mock_model_instance.to.assert_called_once_with(torch.device("cuda"))
