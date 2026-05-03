"""Testes para o módulo data_cleaning.py"""

import pandas as pd
import pytest

from src.data.data_cleaning import (
    clean_dataframe_for_modeling,
    validate_numeric_features,
    validate_split_sizes,
    validate_target_for_stratification,
)
from src.data.preprocessing_config import PreprocessingConfig


class TestCleanDataframeForModeling:
    """Testes para a função clean_dataframe_for_modeling."""

    def test_clean_dataframe_returns_dataframe(self):
        """Verifica se a função retorna um DataFrame."""
        df = pd.DataFrame({"col1": [1, 2], "Churn": [0, 1]})
        result = clean_dataframe_for_modeling(df)
        assert isinstance(result, pd.DataFrame)

    def test_clean_dataframe_makes_copy(self):
        """Verifica se a função retorna uma cópia."""
        df = pd.DataFrame({"col1": [1, 2], "Churn": [0, 1]})
        result = clean_dataframe_for_modeling(df)
        assert result is not df

    def test_clean_dataframe_handles_total_charges_coercion(self):
        """Verifica se Total Charges é coagido para numérico."""
        df = pd.DataFrame({
            "Total Charges": ["100.5", "200.5", "invalid"],
            "Churn": [0, 1, 0]
        })
        result = clean_dataframe_for_modeling(df)
        
        assert pd.api.types.is_numeric_dtype(result["Total Charges"])
        assert pd.isna(result.iloc[2]["Total Charges"])

    def test_clean_dataframe_renames_target_column(self):
        """Verifica se a coluna de target é renomeada corretamente."""
        df = pd.DataFrame({"customerID": [1, 2], "Churn": [0, 1]})
        result = clean_dataframe_for_modeling(df)
        
        assert "Churn" in result.columns


class TestValidateNumericFeatures:
    """Testes para a função validate_numeric_features."""

    def test_validate_numeric_features_passes_for_numeric_data(self):
        """Verifica se valida positivamente dados numéricos."""
        df = pd.DataFrame({
            "col1": [1.0, 2.0],
            "col2": [3, 4],
            "col3": [True, False]
        })
        
        # Não deve levantar exceção
        validate_numeric_features(df)

    def test_validate_numeric_features_raises_for_non_numeric(self):
        """Verifica se levanta erro para colunas não-numéricas."""
        df = pd.DataFrame({
            "col1": [1.0, 2.0],
            "col2": ["a", "b"]
        })
        
        with pytest.raises(ValueError, match="Non-numeric columns found"):
            validate_numeric_features(df)

    def test_validate_numeric_features_raises_for_mixed_types(self):
        """Verifica se levanta erro com tipos mistos."""
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": ["string", 3]
        })
        
        with pytest.raises(ValueError, match="Non-numeric columns found"):
            validate_numeric_features(df)


class TestValidateTargetForStratification:
    """Testes para a função validate_target_for_stratification."""

    def test_validate_target_passes_for_valid_target(self):
        """Verifica se valida positivamente target válido."""
        y = pd.Series([0, 1, 0, 1, 0, 1])
        
        # Não deve levantar exceção
        validate_target_for_stratification(y)

    def test_validate_target_raises_for_missing_values(self):
        """Verifica se levanta erro para valores faltantes."""
        y = pd.Series([0, 1, None, 1, 0, 1])
        
        with pytest.raises(ValueError, match="missing values"):
            validate_target_for_stratification(y)

    def test_validate_target_raises_for_empty_series(self):
        """Verifica se levanta erro para série vazia."""
        y = pd.Series([], dtype=int)
        
        with pytest.raises(ValueError, match="empty"):
            validate_target_for_stratification(y)

    def test_validate_target_raises_when_class_has_less_than_2_samples(self):
        """Verifica se levanta erro quando uma classe tem < 2 amostras."""
        y = pd.Series([0, 1, 1, 1, 1])
        
        with pytest.raises(ValueError, match="at least 2 samples"):
            validate_target_for_stratification(y)

    def test_validate_target_passes_with_balanced_classes(self):
        """Verifica se valida com classes balanceadas."""
        y = pd.Series([0] * 50 + [1] * 50)
        
        validate_target_for_stratification(y)

    def test_validate_target_passes_with_imbalanced_classes(self):
        """Verifica se valida com classes desbalanceadas mas válidas."""
        y = pd.Series([0] * 10 + [1] * 90)
        
        validate_target_for_stratification(y)


class TestValidateSplitSizes:
    """Testes para a função validate_split_sizes."""

    def test_validate_split_sizes_passes_for_valid_sizes(self):
        """Verifica se valida positivamente tamanhos válidos."""
        # Não deve levantar exceção
        validate_split_sizes(test_size=0.2, val_size=0.15)

    def test_validate_split_sizes_raises_for_test_size_zero(self):
        """Verifica se levanta erro para test_size=0."""
        with pytest.raises(ValueError, match="test_size must be between"):
            validate_split_sizes(test_size=0.0, val_size=0.15)

    def test_validate_split_sizes_raises_for_test_size_one(self):
        """Verifica se levanta erro para test_size=1."""
        with pytest.raises(ValueError, match="test_size must be between"):
            validate_split_sizes(test_size=1.0, val_size=0.15)

    def test_validate_split_sizes_raises_for_val_size_zero(self):
        """Verifica se levanta erro para val_size=0."""
        with pytest.raises(ValueError, match="val_size must be between"):
            validate_split_sizes(test_size=0.2, val_size=0.0)

    def test_validate_split_sizes_raises_for_val_size_too_large(self):
        """Verifica se levanta erro quando val_size é muito grande."""
        with pytest.raises(ValueError, match="val_size is too large"):
            validate_split_sizes(test_size=0.2, val_size=0.85)

    def test_validate_split_sizes_raises_for_val_size_equal_to_complement(self):
        """Verifica se levanta erro quando val_size >= (1 - test_size)."""
        with pytest.raises(ValueError, match="val_size is too large"):
            validate_split_sizes(test_size=0.2, val_size=0.8)
