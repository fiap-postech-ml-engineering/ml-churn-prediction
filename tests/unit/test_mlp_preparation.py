"""Testes para o módulo mlp_preparation.py"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.config.settings import (
    TABULAR_RAW_CATEGORICAL_FEATURES,
    TABULAR_RAW_NUMERIC_FEATURES,
)
from src.data.mlp_preparation import prepare_mlp_data
from src.data.preprocessing_config import PreprocessingConfig, TARGET_COLUMN


@pytest.fixture
def sample_processed_df():
    """Cria um dataframe processado e pronto para MLP."""
    n_rows = 100
    numeric_values = {col: np.random.randn(n_rows) for col in TABULAR_RAW_NUMERIC_FEATURES}
    categorical_values = {col: [0, 1] * (n_rows // 2) for col in TABULAR_RAW_CATEGORICAL_FEATURES}
    
    df = pd.DataFrame({**numeric_values, **categorical_values})
    df[TARGET_COLUMN] = np.random.randint(0, 2, n_rows)
    
    return df


class TestPrepareMlpData:
    """Testes para a função prepare_mlp_data."""

    def test_prepare_mlp_data_returns_tuple(self, sample_processed_df):
        """Verifica se a função retorna uma tupla."""
        result = prepare_mlp_data(sample_processed_df)
        assert isinstance(result, tuple)
        assert len(result) == 8

    def test_prepare_mlp_data_returns_correct_types(self, sample_processed_df):
        """Verifica se os tipos de retorno estão corretos."""
        (x_train, x_val, x_test, y_train, y_val, y_test, scaler, feature_names) = prepare_mlp_data(
            sample_processed_df
        )
        
        assert isinstance(x_train, np.ndarray)
        assert isinstance(x_val, np.ndarray)
        assert isinstance(x_test, np.ndarray)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_val, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(scaler, StandardScaler)
        assert isinstance(feature_names, list)

    def test_prepare_mlp_data_preserves_total_rows(self, sample_processed_df):
        """Verifica se o número total de linhas é preservado."""
        (x_train, x_val, x_test, y_train, y_val, y_test, _, _) = prepare_mlp_data(
            sample_processed_df
        )
        
        total_rows = len(x_train) + len(x_val) + len(x_test)
        assert total_rows == len(sample_processed_df)

    def test_prepare_mlp_data_returns_scaled_features(self, sample_processed_df):
        """Verifica se as features são escaladas."""
        (x_train, x_val, x_test, _, _, _, _, _) = prepare_mlp_data(
            sample_processed_df
        )
        
        # Treino deve estar normalizado
        assert np.allclose(x_train.mean(axis=0), 0, atol=1e-1)
        assert np.allclose(x_train.std(axis=0), 1, atol=1e-1)

    def test_prepare_mlp_data_returns_feature_names(self, sample_processed_df):
        """Verifica se os nomes das features são retornados."""
        _, _, _, _, _, _, _, feature_names = prepare_mlp_data(
            sample_processed_df
        )
        
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)

    def test_prepare_mlp_data_preserves_class_distribution(self, sample_processed_df):
        """Verifica se a distribuição de classes é similar entre splits."""
        (x_train, x_val, x_test, y_train, y_val, y_test, _, _) = prepare_mlp_data(
            sample_processed_df
        )
        
        # Todas as classes devem estar presentes em cada split
        for y_split in [y_train, y_val, y_test]:
            assert len(y_split.unique()) >= 1

    def test_prepare_mlp_data_with_custom_target_col(self, sample_processed_df):
        """Verifica se funciona com coluna target customizada."""
        sample_processed_df = sample_processed_df.rename(columns={TARGET_COLUMN: "custom_target"})
        
        result = prepare_mlp_data(
            sample_processed_df,
            target_col="custom_target"
        )
        
        assert len(result) == 8

    def test_prepare_mlp_data_with_custom_sizes(self, sample_processed_df):
        """Verifica se funciona com tamanhos customizados."""
        test_size = 0.15
        val_size = 0.1
        
        (x_train, x_val, x_test, y_train, y_val, y_test, _, _) = prepare_mlp_data(
            sample_processed_df,
            test_size=test_size,
            val_size=val_size
        )
        
        expected_test = int(len(sample_processed_df) * test_size)
        assert len(x_test) == expected_test

    def test_prepare_mlp_data_deterministic_with_seed(self, sample_processed_df):
        """Verifica se o resultado é determinístico com seed."""
        result1 = prepare_mlp_data(sample_processed_df, seed=42)
        result2 = prepare_mlp_data(sample_processed_df, seed=42)
        
        # Verifica se os índices dos splits são iguais
        np.testing.assert_array_equal(result1[3].index, result2[3].index)

    def test_prepare_mlp_data_with_custom_config(self, sample_processed_df):
        """Verifica se funciona com configuração customizada."""
        custom_config = PreprocessingConfig(test_size=0.1, val_size=0.1)
        
        result = prepare_mlp_data(
            sample_processed_df,
            config=custom_config
        )
        
        assert len(result) == 8

    def test_prepare_mlp_data_feature_count_consistency(self, sample_processed_df):
        """Verifica se o número de features é consistente."""
        (x_train, x_val, x_test, _, _, _, _, feature_names) = prepare_mlp_data(
            sample_processed_df
        )
        
        assert x_train.shape[1] == len(feature_names)
        assert x_val.shape[1] == len(feature_names)
        assert x_test.shape[1] == len(feature_names)

    def test_prepare_mlp_data_handles_edge_cases(self):
        """Verifica se funciona com dataset pequeno mas viável para estratificação."""
        # Precisa de pelo menos 4 amostras por classe para split estratificado
        n_samples = 10
        df = pd.DataFrame({
            col: list(range(n_samples)) for col in TABULAR_RAW_NUMERIC_FEATURES
        })
        df.update(pd.DataFrame({
            col: [i % 2 for i in range(n_samples)] for col in TABULAR_RAW_CATEGORICAL_FEATURES
        }))
        df[TARGET_COLUMN] = [i % 2 for i in range(n_samples)]
        
        result = prepare_mlp_data(df)
        assert len(result) == 8
