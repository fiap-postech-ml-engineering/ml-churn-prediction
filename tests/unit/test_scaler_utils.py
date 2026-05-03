"""Testes para o módulo scaler_utils.py"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.scaler_utils import fit_scaler, transform_features


class TestFitScaler:
    """Testes para a função fit_scaler."""

    @pytest.fixture
    def sample_data(self):
        """Cria dados de exemplo para fit do scaler."""
        return pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

    def test_fit_scaler_returns_standard_scaler(self, sample_data):
        """Verifica se a função retorna um StandardScaler."""
        scaler = fit_scaler(sample_data)
        assert isinstance(scaler, StandardScaler)

    def test_fit_scaler_is_fitted(self, sample_data):
        """Verifica se o scaler retornado está ajustado."""
        scaler = fit_scaler(sample_data)
        assert hasattr(scaler, "mean_")
        assert hasattr(scaler, "scale_")

    def test_fit_scaler_computes_correct_mean(self, sample_data):
        """Verifica se a média é calculada corretamente."""
        scaler = fit_scaler(sample_data)
        expected_mean = sample_data.mean().values
        np.testing.assert_array_almost_equal(scaler.mean_, expected_mean)

    def test_fit_scaler_with_single_row(self):
        """Verifica se funciona com uma única linha."""
        df = pd.DataFrame({"feature1": [5.0], "feature2": [10.0]})
        scaler = fit_scaler(df)
        assert scaler.mean_[0] == 5.0

    def test_fit_scaler_with_constant_column(self):
        """Verifica se funciona com coluna constante."""
        df = pd.DataFrame({
            "feature1": [5.0, 5.0, 5.0],
            "feature2": [1.0, 2.0, 3.0]
        })
        scaler = fit_scaler(df)
        assert scaler.mean_[0] == 5.0


class TestTransformFeatures:
    """Testes para a função transform_features."""

    @pytest.fixture
    def scaler_and_data(self):
        """Cria um scaler ajustado e dados para transformar."""
        x_train = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [10.0, 20.0, 30.0],
        })
        scaler = fit_scaler(x_train)

        x_val = pd.DataFrame({
            "feature1": [2.0, 3.0],
            "feature2": [20.0, 30.0],
        })

        x_test = pd.DataFrame({
            "feature1": [1.5, 2.5],
            "feature2": [15.0, 25.0],
        })

        return scaler, x_train, x_val, x_test

    def test_transform_features_returns_three_arrays(self, scaler_and_data):
        """Verifica se a função retorna 3 arrays."""
        scaler, x_train, x_val, x_test = scaler_and_data
        result = transform_features(scaler, x_train, x_val, x_test)

        assert len(result) == 3
        assert all(isinstance(r, np.ndarray) for r in result)

    def test_transform_features_returns_numpy_arrays(self, scaler_and_data):
        """Verifica se os resultados são numpy arrays."""
        scaler, x_train, x_val, x_test = scaler_and_data
        x_train_scaled, x_val_scaled, x_test_scaled = transform_features(
            scaler, x_train, x_val, x_test
        )

        assert isinstance(x_train_scaled, np.ndarray)
        assert isinstance(x_val_scaled, np.ndarray)
        assert isinstance(x_test_scaled, np.ndarray)

    def test_transform_features_preserves_shape(self, scaler_and_data):
        """Verifica se as formas são preservadas."""
        scaler, x_train, x_val, x_test = scaler_and_data
        x_train_scaled, x_val_scaled, x_test_scaled = transform_features(
            scaler, x_train, x_val, x_test
        )

        assert x_train_scaled.shape == x_train.shape
        assert x_val_scaled.shape == x_val.shape
        assert x_test_scaled.shape == x_test.shape

    def test_transform_features_centers_around_zero(self, scaler_and_data):
        """Verifica se os dados são centralizados próximos a zero."""
        scaler, x_train, x_val, x_test = scaler_and_data
        x_train_scaled, x_val_scaled, x_test_scaled = transform_features(
            scaler, x_train, x_val, x_test
        )

        # A média do treino deve ser próxima a zero
        assert np.allclose(x_train_scaled.mean(axis=0), 0, atol=1e-10)

    def test_transform_features_scales_variance(self, scaler_and_data):
        """Verifica se a variância é escalada."""
        scaler, x_train, x_val, x_test = scaler_and_data
        x_train_scaled, x_val_scaled, x_test_scaled = transform_features(
            scaler, x_train, x_val, x_test
        )

        # O desvio padrão do treino deve ser próximo a 1
        std = x_train_scaled.std(axis=0)
        assert np.allclose(std, 1, atol=1e-10)

    def test_transform_features_different_datasets_different_scales(self, scaler_and_data):
        """Verifica se val/test mantêm suas escalas após transformação."""
        scaler, x_train, x_val, x_test = scaler_and_data
        x_train_scaled, x_val_scaled, x_test_scaled = transform_features(
            scaler, x_train, x_val, x_test
        )

        # Val e teste devem manter sua forma após transformação
        assert x_val_scaled.shape == x_val.shape
        assert x_test_scaled.shape == x_test.shape
        # Treino está centrado em zero (por definição do scaler)
        assert np.allclose(x_train_scaled.mean(axis=0), 0, atol=1e-10)

    def test_transform_features_with_single_row(self):
        """Verifica se funciona com uma linha de dados."""
        x_train = pd.DataFrame({"feature1": [1.0], "feature2": [10.0]})
        scaler = fit_scaler(x_train)

        x_val = pd.DataFrame({"feature1": [2.0], "feature2": [20.0]})
        x_test = pd.DataFrame({"feature1": [1.5], "feature2": [15.0]})

        result = transform_features(scaler, x_train, x_val, x_test)
        assert len(result) == 3
        assert all(r.shape[0] == 1 for r in result)

    def test_transform_features_maintains_row_order(self, scaler_and_data):
        """Verifica se a ordem das linhas é mantida."""
        scaler, x_train, x_val, x_test = scaler_and_data
        x_val_scaled = transform_features(scaler, x_train, x_val, x_test)[1]

        # Todas as linhas devem estar presentes
        assert x_val_scaled.shape[0] == len(x_val)
