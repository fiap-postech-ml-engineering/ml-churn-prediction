import pandas as pd

from src.data.preprocessing_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig,
    TOTAL_CHARGES_COLUMN,
)


def clean_dataframe_for_modeling(
    df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> pd.DataFrame:
    """
    Aplica limpeza mínima antes da preparação final para o MLP.

    Espera-se que o dataframe já esteja processado/codificado.
    Esta função apenas protege contra pequenas inconsistências.
    """
    cfg = config or DEFAULT_PREPROCESSING_CONFIG

    df = df.copy()

    existing_cols_to_drop = [col for col in cfg.columns_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)

    if cfg.target_source_column in df.columns and cfg.target_column not in df.columns:
        df = df.rename(columns={cfg.target_source_column: cfg.target_column})
    elif cfg.target_source_column in df.columns and cfg.target_column in df.columns:
        df = df.drop(columns=[cfg.target_source_column])

    if cfg.total_charges_column in df.columns:
        df[cfg.total_charges_column] = pd.to_numeric(
            df[cfg.total_charges_column],
            errors="coerce",
        )

    return df


def validate_numeric_features(x: pd.DataFrame) -> None:
    """
    Garante que todas as features estejam numéricas.
    O pipeline do MLP parte de um dataset já codificado.
    """
    non_numeric_cols = x.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    if non_numeric_cols:
        raise ValueError(
            "MLP preprocessing expects an already encoded numeric dataset. "
            f"Non-numeric columns found: {non_numeric_cols}"
        )


def validate_target_for_stratification(y: pd.Series) -> None:
    """
    Garante que a target permita split estratificado.
    """
    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    class_counts = y.value_counts()

    if class_counts.empty:
        raise ValueError("Target column is empty.")

    if class_counts.min() < 2:
        raise ValueError(
            "Stratified split requires at least 2 samples in each target class."
        )


def validate_split_sizes(test_size: float, val_size: float) -> None:
    """
    Valida proporções de teste e validação para split em duas etapas.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")

    if val_size >= (1.0 - test_size):
        raise ValueError(
            "val_size is too large for a two-step split. It must be smaller than "
            "(1 - test_size)."
        )