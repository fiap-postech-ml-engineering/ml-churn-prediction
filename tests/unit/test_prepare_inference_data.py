"""Testes para o módulo prepare_inference_data.py"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.inference.prepare_inference_data import prepare_inference_batch, run_inference


class TestPrepareInferenceBatch:
    """Testes para a função prepare_inference_batch."""

    @pytest.fixture
    def sample_dataframe(self):
        """Cria um DataFrame de exemplo."""
        return pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

    @pytest.fixture
    def mock_scaler(self):
        """Cria um mock do StandardScaler."""
        scaler = MagicMock(spec=StandardScaler)
        # Simula transformação que retorna array numpy
        scaler.transform.return_value = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
            [0.9, 1.0],
        ])
        return scaler

    @pytest.fixture
    def mock_device(self):
        """Cria um mock do torch.device."""
        device = MagicMock(spec=torch.device)
        device.type = "cpu"
        return device

    @patch("src.inference.prepare_inference_data.torch.from_numpy")
    def test_prepare_inference_batch_returns_dataloader(self, mock_from_numpy, sample_dataframe, mock_scaler, mock_device):
        """Verifica se retorna um DataLoader."""
        # Mock tensor chain
        mock_tensor = MagicMock()
        mock_from_numpy.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor

        result = prepare_inference_batch(sample_dataframe, mock_scaler, mock_device)
        assert isinstance(result, DataLoader)

    def test_prepare_inference_batch_raises_error_for_empty_dataframe(self, mock_scaler, mock_device):
        """Verifica se levanta erro para DataFrame vazio."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame de features não pode estar vazio"):
            prepare_inference_batch(empty_df, mock_scaler, mock_device)

    def test_prepare_inference_batch_raises_error_for_none_scaler(self, sample_dataframe, mock_device):
        """Verifica se levanta erro quando scaler é None."""
        with pytest.raises(ValueError, match="Scaler não foi carregado corretamente"):
            prepare_inference_batch(sample_dataframe, None, mock_device)

    @patch("src.inference.prepare_inference_data.torch.from_numpy")
    @patch("src.inference.prepare_inference_data.logger")
    def test_prepare_inference_batch_calls_scaler_transform(self, mock_logger, mock_from_numpy, sample_dataframe, mock_scaler, mock_device):
        """Verifica se o scaler.transform é chamado."""
        # Mock tensor chain
        mock_tensor = MagicMock()
        mock_from_numpy.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor

        prepare_inference_batch(sample_dataframe, mock_scaler, mock_device)
        mock_scaler.transform.assert_called_once_with(sample_dataframe)

    @patch("src.inference.prepare_inference_data.torch.from_numpy")
    @patch("src.inference.prepare_inference_data.logger")
    def test_prepare_inference_batch_creates_tensor_correctly(self, mock_logger, mock_from_numpy, sample_dataframe, mock_scaler, mock_device):
        """Verifica se o tensor é criado corretamente."""
        mock_tensor = MagicMock()
        mock_from_numpy.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor

        prepare_inference_batch(sample_dataframe, mock_scaler, mock_device)

        mock_from_numpy.assert_called_once()
        mock_tensor.float.assert_called_once()
        mock_tensor.to.assert_called_once_with(mock_device)

    @patch("src.inference.prepare_inference_data.torch.from_numpy")
    @patch("src.inference.prepare_inference_data.logger")
    def test_prepare_inference_batch_handles_sparse_matrix(self, mock_logger, mock_from_numpy, sample_dataframe, mock_device):
        """Verifica se trata matrizes esparsas corretamente."""
        # Mock tensor chain
        mock_tensor = MagicMock()
        mock_from_numpy.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor

        # Mock scaler que retorna matriz esparsa
        mock_scaler = MagicMock()
        sparse_matrix = MagicMock()
        sparse_matrix.toarray.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_scaler.transform.return_value = sparse_matrix

        result = prepare_inference_batch(sample_dataframe.head(2), mock_scaler, mock_device)
        assert isinstance(result, DataLoader)
        sparse_matrix.toarray.assert_called_once()

    @patch("src.inference.prepare_inference_data.torch.from_numpy")
    @patch("src.inference.prepare_inference_data.logger")
    def test_prepare_inference_batch_warns_high_zero_ratio(self, mock_logger, mock_from_numpy, sample_dataframe, mock_scaler, mock_device):
        """Verifica se avisa quando há alta proporção de zeros."""
        # Mock tensor chain
        mock_tensor = MagicMock()
        mock_from_numpy.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor

        # Configurar scaler para retornar muitos zeros
        mock_scaler.transform.return_value = np.zeros((5, 2))

        prepare_inference_batch(sample_dataframe, mock_scaler, mock_device)

        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0]
        assert "Alta proporção de zeros" in warning_call[0]

    @patch("src.inference.prepare_inference_data.torch.from_numpy")
    @patch("src.inference.prepare_inference_data.logger")
    def test_prepare_inference_batch_logs_info_messages(self, mock_logger, mock_from_numpy, sample_dataframe, mock_scaler, mock_device):
        """Verifica se registra mensagens de log apropriadas."""
        # Mock tensor chain
        mock_tensor = MagicMock()
        mock_from_numpy.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor

        prepare_inference_batch(sample_dataframe, mock_scaler, mock_device)

        assert mock_logger.info.call_count >= 3  # Pelo menos 3 chamadas de log

    @patch("src.inference.prepare_inference_data.torch.from_numpy")
    def test_prepare_inference_batch_uses_correct_batch_size(self, mock_from_numpy, sample_dataframe, mock_scaler, mock_device):
        """Verifica se usa o batch_size correto."""
        # Mock tensor chain
        mock_tensor = MagicMock()
        mock_from_numpy.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor

        batch_size = 2
        result = prepare_inference_batch(sample_dataframe, mock_scaler, mock_device, batch_size=batch_size)
        assert result.batch_size == batch_size

    @patch("src.inference.prepare_inference_data.torch.from_numpy")
    @patch("src.inference.prepare_inference_data.DataLoader")
    def test_prepare_inference_batch_no_shuffle(self, mock_dataloader_class, mock_from_numpy, sample_dataframe, mock_scaler, mock_device):
        """Verifica se DataLoader não faz shuffle."""
        # Mock tensor chain
        mock_tensor = MagicMock()
        mock_from_numpy.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor

        prepare_inference_batch(sample_dataframe, mock_scaler, mock_device)

        mock_dataloader_class.assert_called_once()
        args, kwargs = mock_dataloader_class.call_args
        assert not kwargs.get('shuffle')

    @patch("src.inference.prepare_inference_data.logger")
    def test_prepare_inference_batch_handles_exceptions(self, mock_logger, sample_dataframe, mock_scaler, mock_device):
        """Verifica tratamento de exceções."""
        mock_scaler.transform.side_effect = Exception("Transform error")

        with pytest.raises(RuntimeError, match="Erro na preparação de dados"):
            prepare_inference_batch(sample_dataframe, mock_scaler, mock_device)


class TestRunInference:
    """Testes para a função run_inference."""

    @pytest.fixture
    def mock_dataloader(self):
        """Cria um mock do DataLoader."""
        # Simular batches de dados com mock do método .to()
        batch1_tensor = MagicMock()
        batch1_tensor.to.return_value = batch1_tensor
        batch1 = [batch1_tensor]

        batch2_tensor = MagicMock()
        batch2_tensor.to.return_value = batch2_tensor
        batch2 = [batch2_tensor]

        dataloader = MagicMock(spec=DataLoader)
        dataloader.__iter__.return_value = [batch1, batch2]
        return dataloader

    @pytest.fixture
    def mock_model(self):
        """Cria um mock do modelo PyTorch."""
        model = MagicMock()
        # Simular saída do modelo (logits) com mock do método .to()
        output1 = MagicMock()
        output1.to.return_value = output1
        output2 = MagicMock()
        output2.to.return_value = output2

        model.return_value = output1
        model.side_effect = [output1, output2]
        return model

    @pytest.fixture
    def mock_device(self):
        """Cria um mock do torch.device."""
        device = MagicMock(spec=torch.device)
        device.type = "cpu"
        return device

    @patch("src.inference.prepare_inference_data.torch.sigmoid")
    def test_run_inference_returns_tuple_of_arrays(self, mock_sigmoid, mock_dataloader, mock_model, mock_device):
        """Verifica se retorna tupla de arrays numpy."""
        # Mock sigmoid return
        mock_sigmoid.return_value = torch.tensor([[0.6], [0.4]])

        probas, classes = run_inference(mock_dataloader, mock_model, mock_device)

        assert isinstance(probas, np.ndarray)
        assert isinstance(classes, np.ndarray)
        assert len(probas) == len(classes)

    def test_run_inference_raises_error_for_none_model(self, mock_dataloader, mock_device):
        """Verifica se levanta erro quando modelo é None."""
        with pytest.raises(RuntimeError, match="Modelo não foi carregado corretamente"):
            run_inference(mock_dataloader, None, mock_device)

    @patch("src.inference.prepare_inference_data.torch.sigmoid")
    def test_run_inference_applies_sigmoid_to_logits(self, mock_sigmoid, mock_dataloader, mock_model, mock_device):
        """Verifica se aplica sigmoid aos logits."""
        mock_sigmoid.return_value = torch.tensor([[0.6], [0.4]])

        run_inference(mock_dataloader, mock_model, mock_device)

        # Deve ser chamado para cada batch
        assert mock_sigmoid.call_count == 2

    @patch("src.inference.prepare_inference_data.torch.sigmoid")
    def test_run_inference_uses_correct_threshold(self, mock_sigmoid, mock_dataloader, mock_model, mock_device):
        """Verifica se usa o threshold correto para classificação."""
        # Mock sigmoid return
        mock_sigmoid.return_value = torch.tensor([[0.6], [0.4]])

        threshold = 0.7
        probas, classes = run_inference(mock_dataloader, mock_model, mock_device, approval_threshold=threshold)

        # Com threshold 0.7, valores >= 0.7 devem ser 1
        expected_classes = (probas >= threshold).astype(int)
        np.testing.assert_array_equal(classes, expected_classes)

    @patch("src.inference.prepare_inference_data.torch.sigmoid")
    def test_run_inference_model_in_eval_mode(self, mock_sigmoid, mock_dataloader, mock_model, mock_device):
        """Verifica se o modelo é colocado em modo eval."""
        # Mock sigmoid return
        mock_sigmoid.return_value = torch.tensor([[0.6], [0.4]])

        run_inference(mock_dataloader, mock_model, mock_device)

        mock_model.eval.assert_called_once()

    @patch("src.inference.prepare_inference_data.torch.no_grad")
    def test_run_inference_uses_no_grad_context(self, mock_no_grad, mock_dataloader, mock_model, mock_device):
        """Verifica se usa torch.no_grad()."""
        mock_no_grad.return_value.__enter__ = MagicMock()
        mock_no_grad.return_value.__exit__ = MagicMock()

        run_inference(mock_dataloader, mock_model, mock_device)

        mock_no_grad.assert_called_once()

    @patch("src.inference.prepare_inference_data.torch.sigmoid")
    @patch("src.inference.prepare_inference_data.logger")
    def test_run_inference_logs_info_messages(self, mock_logger, mock_sigmoid, mock_dataloader, mock_model, mock_device):
        """Verifica se registra mensagens de log."""
        # Mock sigmoid return
        mock_sigmoid.return_value = torch.tensor([[0.6], [0.4]])

        run_inference(mock_dataloader, mock_model, mock_device)

        assert mock_logger.info.call_count >= 2  # Pelo menos 2 chamadas

    @patch("src.inference.prepare_inference_data.torch.sigmoid")
    def test_run_inference_handles_single_sample(self, mock_sigmoid, mock_model, mock_device):
        """Verifica funcionamento com uma única amostra."""
        # Mock sigmoid return
        mock_sigmoid.return_value = torch.tensor([[0.6]])

        # DataLoader com uma única amostra - mock do método .to()
        batch_tensor = MagicMock()
        batch_tensor.to.return_value = batch_tensor
        single_batch = [batch_tensor]
        dataloader = MagicMock(spec=DataLoader)
        dataloader.__iter__.return_value = [single_batch]

        # Modelo retorna logit único com mock .to()
        model_output = MagicMock()
        model_output.to.return_value = model_output
        mock_model.return_value = model_output
        mock_model.side_effect = None

        probas, classes = run_inference(dataloader, mock_model, mock_device)

        assert len(probas) == 1
        assert len(classes) == 1

    @patch("src.inference.prepare_inference_data.torch.sigmoid")
    @patch("src.inference.prepare_inference_data.logger")
    def test_run_inference_handles_exceptions(self, mock_logger, mock_sigmoid, mock_device):
        """Verifica tratamento de exceções."""
        # Criar dataloader customizado para este teste
        batch_tensor = MagicMock()
        batch_tensor.to.return_value = batch_tensor
        batch = [batch_tensor]
        dataloader = MagicMock(spec=DataLoader)
        dataloader.__iter__.return_value = [batch]

        mock_model = MagicMock()
        mock_model.side_effect = Exception("Model prediction error")

        with pytest.raises(RuntimeError, match="Erro na execução de inferência"):
            run_inference(dataloader, mock_model, mock_device)

    @patch("src.inference.prepare_inference_data.torch.sigmoid")
    def test_run_inference_correct_probability_range(self, mock_sigmoid, mock_dataloader, mock_model, mock_device):
        """Verifica se probabilidades estão no range correto [0,1]."""
        # Mock sigmoid return
        mock_sigmoid.return_value = torch.tensor([[0.6], [0.4]])

        probas, classes = run_inference(mock_dataloader, mock_model, mock_device)

        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    @patch("src.inference.prepare_inference_data.torch.sigmoid")
    def test_run_inference_classes_are_binary(self, mock_sigmoid, mock_dataloader, mock_model, mock_device):
        """Verifica se classes são apenas 0 ou 1."""
        # Mock sigmoid return
        mock_sigmoid.return_value = torch.tensor([[0.6], [0.4]])

        probas, classes = run_inference(mock_dataloader, mock_model, mock_device)

        assert np.all(np.isin(classes, [0, 1]))

    @patch("src.inference.prepare_inference_data.torch.sigmoid")
    def test_run_inference_default_threshold_half(self, mock_sigmoid, mock_dataloader, mock_model, mock_device):
        """Verifica se threshold padrão é 0.5."""
        # Mock sigmoid return
        mock_sigmoid.return_value = torch.tensor([[0.6], [0.4]])

        probas, classes = run_inference(mock_dataloader, mock_model, mock_device)

        expected_classes = (probas >= 0.5).astype(int)
        np.testing.assert_array_equal(classes, expected_classes)
