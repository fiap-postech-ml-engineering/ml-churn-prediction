"""Testes para o módulo feature_selection.py"""

import pandas as pd
import pytest

from src.config.settings import TABULAR_RAW_FEATURES
from src.data.feature_selection import select_tabular_raw_features


class TestSelectTabularRawFeatures:
    """Testes para a função select_tabular_raw_features."""

    @pytest.fixture
    def sample_df(self):
        """Cria um dataframe de exemplo com todas as features necessárias."""
        features_dict = {col: [1, 2] for col in TABULAR_RAW_FEATURES}
        features_dict["Churn Value"] = [0, 1]
        return pd.DataFrame(features_dict)

    def test_select_features_returns_dataframe(self, sample_df):
        """Verifica se a função retorna um DataFrame."""
        result = select_tabular_raw_features(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_select_features_returns_all_required_features(self, sample_df):
        """Verifica se todas as features necessárias estão presentes."""
        result = select_tabular_raw_features(sample_df)
        for col in TABULAR_RAW_FEATURES:
            assert col in result.columns

    def test_select_features_preserves_row_count(self, sample_df):
        """Verifica se o número de linhas é preservado."""
        result = select_tabular_raw_features(sample_df)
        assert len(result) == len(sample_df)

    def test_select_features_makes_copy(self, sample_df):
        """Verifica se a função retorna uma cópia, não a original."""
        result = select_tabular_raw_features(sample_df)
        assert result is not sample_df
        result.iloc[0, 0] = 999
        assert sample_df.iloc[0, 0] != 999

    def test_select_features_with_require_target_false(self, sample_df):
        """Verifica comportamento quando require_target=False."""
        result = select_tabular_raw_features(sample_df, require_target=False)
        assert "Churn Value" not in result.columns
        assert len(result.columns) == len(TABULAR_RAW_FEATURES)

    def test_select_features_with_require_target_true(self, sample_df):
        """Verifica comportamento quando require_target=True."""
        result = select_tabular_raw_features(sample_df, require_target=True)
        assert "Churn Value" in result.columns
        assert len(result.columns) == len(TABULAR_RAW_FEATURES) + 1

    def test_select_features_raises_error_for_missing_columns(self):
        """Verifica se levanta erro quando faltam colunas necessárias."""
        df_incompleto = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        
        with pytest.raises(ValueError, match="Missing critical raw inference features"):
            select_tabular_raw_features(df_incompleto)

    def test_select_features_raises_error_when_target_required_but_missing(self, sample_df):
        """Verifica se levanta erro quando target é obrigatório mas não existe."""
        df_sem_target = sample_df.drop(columns=["Churn Value"])
        
        with pytest.raises(ValueError, match="Target column"):
            select_tabular_raw_features(df_sem_target, require_target=True)

    def test_select_features_with_extra_columns(self, sample_df):
        """Verifica se funciona com colunas extras no dataframe."""
        sample_df["extra_column"] = [10, 20]
        result = select_tabular_raw_features(sample_df)
        
        assert "extra_column" not in result.columns
        assert len(result.columns) == len(TABULAR_RAW_FEATURES)
