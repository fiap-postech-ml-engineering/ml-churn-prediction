from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_SOURCE_COLUMN = "Churn Value"
TARGET_COLUMN = "target"
TOTAL_CHARGES_COLUMN = "Total Charges"
COLUMNS_TO_DROP = ("Churn Score", "Count")

ProcessedArray = np.ndarray


def clean_dataframe_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpeza mínima antes da preparação final para o MLP.

    Espera-se que o dataframe já esteja processado/codificado.
    Esta função apenas protege contra pequenas inconsistências.
    """
    df = df.copy()

    existing_cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)

    if TARGET_SOURCE_COLUMN in df.columns and TARGET_COLUMN not in df.columns:
        df = df.rename(columns={TARGET_SOURCE_COLUMN: TARGET_COLUMN})
    elif TARGET_SOURCE_COLUMN in df.columns and TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_SOURCE_COLUMN])

    if TOTAL_CHARGES_COLUMN in df.columns:
        df[TOTAL_CHARGES_COLUMN] = pd.to_numeric(
            df[TOTAL_CHARGES_COLUMN],
            errors="coerce",
        )

    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa features (X) e target (y).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    return X, y


def validate_numeric_features(X: pd.DataFrame) -> None:
    """
    Garante que todas as features estejam numéricas.
    O pipeline do MLP parte de um dataset já codificado.
    """
    non_numeric_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

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


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
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
    validate_split_sizes(test_size=test_size, val_size=val_size)

    validate_target_for_stratification(y)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    validate_target_for_stratification(y_train_full)

    val_relative_size = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_relative_size,
        stratify=y_train_full,
        random_state=seed,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Ajusta o scaler somente nos dados de treino.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_features(
    scaler: StandardScaler,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[ProcessedArray, ProcessedArray, ProcessedArray]:
    """
    Aplica o scaler já ajustado em treino, validação e teste.
    """
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


def prepare_mlp_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
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
    df = clean_dataframe_for_modeling(df)

    X, y = split_features_target(df=df, target_col=target_col)

    validate_numeric_features(X)

    feature_names = X.columns.tolist()

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )

    scaler = fit_scaler(X_train)

    X_train_scaled, X_val_scaled, X_test_scaled = transform_features(
        scaler=scaler,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
        feature_names,
    )
