"""Testes para o módulo preprocessing_config.py"""

import numpy as np
import pytest

from src.data.preprocessing_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    COLUMNS_TO_DROP,
    DEFAULT_PREPROCESSING_PIPELINE_PATH,
    PreprocessingConfig,
    TABULAR_PIPELINE_TYPE,
    TABULAR_TARGET_COLUMN,
    TARGET_COLUMN,
    TARGET_SOURCE_COLUMN,
    TOTAL_CHARGES_COLUMN,
    ProcessedArray,
)


class TestPreprocessingConfig:
    """Testes para a classe PreprocessingConfig."""

    def test_config_has_required_attributes(self):
        """Verifica se a configuração padrão possui todos os atributos."""
        config = DEFAULT_PREPROCESSING_CONFIG

        assert hasattr(config, "target_source_column")
        assert hasattr(config, "target_column")
        assert hasattr(config, "total_charges_column")
        assert hasattr(config, "columns_to_drop")
        assert hasattr(config, "pipeline_path")
        assert hasattr(config, "random_seed")
        assert hasattr(config, "test_size")
        assert hasattr(config, "val_size")

    def test_config_is_frozen(self):
        """Verifica se a configuração é imutável."""
        config = DEFAULT_PREPROCESSING_CONFIG

        with pytest.raises(AttributeError):
            config.target_column = "new_value"

    def test_config_has_default_values(self):
        """Verifica se a configuração possui valores padrão válidos."""
        config = DEFAULT_PREPROCESSING_CONFIG

        assert isinstance(config.target_source_column, str)
        assert isinstance(config.target_column, str)
        assert isinstance(config.total_charges_column, str)
        assert isinstance(config.columns_to_drop, tuple)
        assert isinstance(config.random_seed, int)
        assert 0 < config.test_size < 1
        assert 0 < config.val_size < 1

    def test_config_can_be_created_with_custom_values(self):
        """Verifica se é possível criar uma configuração com valores customizados."""
        custom_config = PreprocessingConfig(
            target_source_column="custom_source",
            target_column="custom_target",
        )

        assert custom_config.target_source_column == "custom_source"
        assert custom_config.target_column == "custom_target"


class TestBackwardCompatibilityConstants:
    """Testes para constantes de compatibilidade backward."""

    def test_target_source_column_constant(self):
        """Verifica se TARGET_SOURCE_COLUMN está definido e é string."""
        assert isinstance(TARGET_SOURCE_COLUMN, str)
        assert len(TARGET_SOURCE_COLUMN) > 0

    def test_target_column_constant(self):
        """Verifica se TARGET_COLUMN está definido e é string."""
        assert isinstance(TARGET_COLUMN, str)
        assert len(TARGET_COLUMN) > 0

    def test_total_charges_column_constant(self):
        """Verifica se TOTAL_CHARGES_COLUMN está definido e é string."""
        assert isinstance(TOTAL_CHARGES_COLUMN, str)
        assert len(TOTAL_CHARGES_COLUMN) > 0

    def test_columns_to_drop_constant(self):
        """Verifica se COLUMNS_TO_DROP está definido e é tuple."""
        assert isinstance(COLUMNS_TO_DROP, tuple)

    def test_default_pipeline_path_constant(self):
        """Verifica se DEFAULT_PREPROCESSING_PIPELINE_PATH está definido."""
        assert DEFAULT_PREPROCESSING_PIPELINE_PATH is not None

    def test_tabular_target_column_constant(self):
        """Verifica se TABULAR_TARGET_COLUMN está definido e é string."""
        assert isinstance(TABULAR_TARGET_COLUMN, str)
        assert len(TABULAR_TARGET_COLUMN) > 0

    def test_tabular_pipeline_type_constant(self):
        """Verifica se TABULAR_PIPELINE_TYPE está definido e é string."""
        assert isinstance(TABULAR_PIPELINE_TYPE, str)
        assert TABULAR_PIPELINE_TYPE == "tabular_mlp_preprocessing"


class TestProcessedArray:
    """Testes para o tipo ProcessedArray."""

    def test_processed_array_is_numpy_ndarray(self):
        """Verifica se ProcessedArray é numpy.ndarray."""
        assert ProcessedArray == np.ndarray

    def test_can_create_processed_array(self):
        """Verifica se é possível criar um ProcessedArray."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        assert isinstance(arr, ProcessedArray)
        assert arr.shape == (2, 3)
