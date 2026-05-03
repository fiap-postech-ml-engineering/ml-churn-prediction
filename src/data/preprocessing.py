"""
Módulo de Pré-processamento de Dados.

Compatibilidade backward: importa todas as funções dos submódulos.
"""

# Configurações
# Backward compatibility: ProcessedArray
import numpy as np

# Artifact utils
from .artifact_utils import (
    build_preprocessing_artifact,
    load_preprocessing_pipeline,
    save_preprocessing_pipeline,
    try_load_preprocessing_pipeline,
)

# Data cleaning and validation
from .data_cleaning import (
    clean_dataframe_for_modeling,
    validate_numeric_features,
    validate_split_sizes,
    validate_target_for_stratification,
)

# Data split
from .data_split import split_features_target, split_train_val_test

# Feature selection
from .feature_selection import select_tabular_raw_features

# MLP preparation
from .mlp_preparation import prepare_mlp_data
from .preprocessing_config import (
    COLUMNS_TO_DROP,
    DEFAULT_PREPROCESSING_CONFIG,
    DEFAULT_PREPROCESSING_PIPELINE_PATH,
    TABULAR_PIPELINE_TYPE,
    TABULAR_TARGET_COLUMN,
    TARGET_COLUMN,
    TARGET_SOURCE_COLUMN,
    TOTAL_CHARGES_COLUMN,
    PreprocessingConfig,
)

# Scaler utils
from .scaler_utils import fit_scaler, transform_features

# Tabular pipeline
from .tabular_pipeline import (
    build_tabular_preprocessing_artifact,
    build_tabular_preprocessing_pipeline,
    fit_tabular_preprocessing_pipeline,
    load_tabular_preprocessing_pipeline,
    save_tabular_preprocessing_pipeline,
    split_tabular_features_target,
    transform_tabular_features,
    try_load_tabular_preprocessing_pipeline,
)

ProcessedArray = np.ndarray

__all__ = [
    # Config
    "DEFAULT_PREPROCESSING_CONFIG",
    "PreprocessingConfig",
    "TABULAR_PIPELINE_TYPE",
    "TABULAR_TARGET_COLUMN",
    "TARGET_COLUMN",
    "TARGET_SOURCE_COLUMN",
    "TOTAL_CHARGES_COLUMN",
    "COLUMNS_TO_DROP",
    "DEFAULT_PREPROCESSING_PIPELINE_PATH",
    "ProcessedArray",
    # Feature selection
    "select_tabular_raw_features",
    # Tabular pipeline
    "build_tabular_preprocessing_artifact",
    "build_tabular_preprocessing_pipeline",
    "fit_tabular_preprocessing_pipeline",
    "load_tabular_preprocessing_pipeline",
    "save_tabular_preprocessing_pipeline",
    "split_tabular_features_target",
    "transform_tabular_features",
    "try_load_tabular_preprocessing_pipeline",
    # Data cleaning
    "clean_dataframe_for_modeling",
    "validate_numeric_features",
    "validate_split_sizes",
    "validate_target_for_stratification",
    # Data split
    "split_features_target",
    "split_train_val_test",
    # Scaler
    "fit_scaler",
    "transform_features",
    # MLP prep
    "prepare_mlp_data",
    # Artifacts
    "build_preprocessing_artifact",
    "load_preprocessing_pipeline",
    "save_preprocessing_pipeline",
    "try_load_preprocessing_pipeline",
]
