import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessing_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig,
    TARGET_COLUMN,
)


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa features (X) e target (y).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    x = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    return x, y


def split_train_val_test(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
    val_size: float | None = None,
    seed: int | None = None,
    config: PreprocessingConfig | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Faz split estratificado em:
    - treino
    - validação
    - teste

    Estratégia:
    1. separa teste
    2. separa validação a partir do treino restante
    """
    cfg = config or DEFAULT_PREPROCESSING_CONFIG
    test_size = cfg.test_size if test_size is None else test_size
    val_size = cfg.val_size if val_size is None else val_size
    seed = cfg.random_seed if seed is None else seed

    from src.data.data_cleaning import validate_split_sizes, validate_target_for_stratification

    validate_split_sizes(test_size=test_size, val_size=val_size)

    validate_target_for_stratification(y)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    validate_target_for_stratification(y_train_full)

    val_relative_size = val_size / (1.0 - test_size)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_relative_size,
        stratify=y_train_full,
        random_state=seed,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test