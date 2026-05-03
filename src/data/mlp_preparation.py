import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.data_cleaning import clean_dataframe_for_modeling, validate_numeric_features
from src.data.data_split import split_features_target, split_train_val_test
from src.data.preprocessing_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig,
    ProcessedArray,
    TARGET_COLUMN,
)
from src.data.scaler_utils import fit_scaler, transform_features


def prepare_mlp_data(
    df: pd.DataFrame,
    target_col: str | None = None,
    test_size: float | None = None,
    val_size: float | None = None,
    seed: int | None = None,
    config: PreprocessingConfig | None = None,
) -> tuple[
    ProcessedArray,
    ProcessedArray,
    ProcessedArray,
    pd.Series,
    pd.Series,
    pd.Series,
    StandardScaler,
    list[str],
]:
    """
    Fluxo completo de preparação para o MLP:

    1. limpeza mínima
    2. separação X/y
    3. validação de dataset numérico
    4. split estratificado em treino/validação/teste
    5. fit do scaler apenas no treino
    6. transformação de treino/validação/teste

    Returns
    -------
    tuple
        (
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            y_train,
            y_val,
            y_test,
            scaler,
            feature_names,
        )
    """
    cfg = config or DEFAULT_PREPROCESSING_CONFIG
    target_col = cfg.target_column if target_col is None else target_col
    test_size = cfg.test_size if test_size is None else test_size
    val_size = cfg.val_size if val_size is None else val_size
    seed = cfg.random_seed if seed is None else seed

    df = clean_dataframe_for_modeling(df, config=cfg)

    x, y = split_features_target(df=df, target_col=target_col)

    validate_numeric_features(x)

    feature_names = x.columns.tolist()

    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
        x=x,
        y=y,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        config=cfg,
    )

    scaler = fit_scaler(x_train)

    x_train_scaled, x_val_scaled, x_test_scaled = transform_features(
        scaler=scaler,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
    )

    return (
        x_train_scaled,
        x_val_scaled,
        x_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
        feature_names,
    )