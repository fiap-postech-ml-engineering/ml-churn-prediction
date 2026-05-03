"""Testes para o módulo data_split.py"""

import pandas as pd
import pytest

from src.data.data_split import split_features_target, split_train_val_test
from src.data.preprocessing_config import DEFAULT_PREPROCESSING_CONFIG, PreprocessingConfig


class TestSplitFeaturesTarget:
    """Testes para a função split_features_target."""

    @pytest.fixture
    def sample_df(self):
        """Cria um dataframe de exemplo."""
        return pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0],
            "feature2": [5.0, 6.0, 7.0, 8.0],
            "Churn Value": [0, 1, 0, 1]
        })

    def test_split_features_target_returns_tuple(self, sample_df):
        """Verifica se a função retorna uma tupla."""
        result = split_features_target(sample_df, target_col="Churn Value")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_split_features_target_returns_dataframe_and_series(self, sample_df):
        """Verifica se retorna DataFrame e Series."""
        x, y = split_features_target(sample_df, target_col="Churn Value")
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_split_features_target_removes_target_column(self, sample_df):
        """Verifica se a coluna target é removida das features."""
        x, y = split_features_target(sample_df, target_col="Churn Value")
        assert "Churn Value" not in x.columns
        assert "feature1" in x.columns
        assert "feature2" in x.columns

    def test_split_features_target_preserves_target_values(self, sample_df):
        """Verifica se os valores da target são preservados."""
        x, y = split_features_target(sample_df, target_col="Churn Value")
        assert list(y.values) == [0, 1, 0, 1]

    def test_split_features_target_preserves_row_count(self, sample_df):
        """Verifica se o número de linhas é preservado."""
        x, y = split_features_target(sample_df, target_col="Churn Value")
        assert len(x) == len(sample_df)
        assert len(y) == len(sample_df)

    def test_split_features_target_raises_error_for_missing_target(self, sample_df):
        """Verifica se levanta erro quando target está ausente."""
        df_sem_target = sample_df.drop(columns=["Churn Value"])
        
        with pytest.raises(ValueError, match="Target column"):
            split_features_target(df_sem_target)

    def test_split_features_target_with_custom_target_col(self, sample_df):
        """Verifica comportamento com coluna target customizada."""
        df = sample_df.rename(columns={"Churn Value": "custom_target"})
        x, y = split_features_target(df, target_col="custom_target")
        
        assert "custom_target" not in x.columns
        assert len(y) == len(sample_df)


class TestSplitTrainValTest:
    """Testes para a função split_train_val_test."""

    @pytest.fixture
    def sample_data(self):
        """Cria dados de exemplo para split."""
        n_samples = 100
        return pd.DataFrame({
            "feature1": range(n_samples),
            "feature2": range(n_samples, 2 * n_samples),
            "Churn Value": [0, 1] * (n_samples // 2),
        }), pd.Series([0, 1] * (n_samples // 2))

    def test_split_returns_six_arrays(self, sample_data):
        """Verifica se a função retorna 6 arrays."""
        x, y = sample_data
        result = split_train_val_test(x, y)
        assert len(result) == 6

    def test_split_returns_dataframes_and_series(self, sample_data):
        """Verifica se retorna DataFrames e Series."""
        x, y = sample_data
        x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(x, y)
        
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(x_val, pd.DataFrame)
        assert isinstance(x_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_val, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_split_preserves_total_row_count(self, sample_data):
        """Verifica se o número total de linhas é preservado."""
        x, y = sample_data
        x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(x, y)
        
        total_rows = len(x_train) + len(x_val) + len(x_test)
        assert total_rows == len(x)

    def test_split_respects_test_size(self, sample_data):
        """Verifica se test_size é respeitado."""
        x, y = sample_data
        test_size = 0.2
        x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
            x, y, test_size=test_size
        )
        
        expected_test_count = int(len(x) * test_size)
        assert len(x_test) == expected_test_count

    def test_split_respects_val_size(self, sample_data):
        """Verifica se val_size é respeitado."""
        x, y = sample_data
        test_size = 0.2
        val_size = 0.15
        x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
            x, y, test_size=test_size, val_size=val_size
        )
        
        train_size = 1.0 - test_size
        expected_val_count = int(len(x) * train_size * val_size / (1.0 - test_size))
        assert len(x_val) == expected_val_count

    def test_split_produces_stratified_splits(self, sample_data):
        """Verifica se os splits são estratificados."""
        x, y = sample_data
        x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
            x, y, seed=42
        )
        
        # Verifica se todas as classes estão presentes em cada split
        for split_y in [y_train, y_val, y_test]:
            assert len(split_y.unique()) >= 1

    def test_split_with_custom_config(self, sample_data):
        """Verifica se funciona com configuração customizada."""
        x, y = sample_data
        custom_config = PreprocessingConfig(test_size=0.1, val_size=0.1)
        x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
            x, y, config=custom_config
        )
        
        total_rows = len(x_train) + len(x_val) + len(x_test)
        assert total_rows == len(x)

    def test_split_raises_error_for_invalid_split_sizes(self, sample_data):
        """Verifica se levanta erro para tamanhos inválidos."""
        x, y = sample_data
        
        with pytest.raises(ValueError):
            split_train_val_test(x, y, test_size=0.0, val_size=0.5)

    def test_split_deterministic_with_seed(self, sample_data):
        """Verifica se o split é determinístico com seed."""
        x, y = sample_data
        
        result1 = split_train_val_test(x, y, seed=42)
        result2 = split_train_val_test(x, y, seed=42)
        
        # Verifica se os índices são iguais
        assert result1[0].index.equals(result2[0].index)
        assert result1[1].index.equals(result2[1].index)
        assert result1[2].index.equals(result2[2].index)
