from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config.settings import RANDOM_SEED

TARGET_SOURCE_COLUMN = "Churn Value"
TARGET_COLUMN = "target"
TOTAL_CHARGES_COLUMN = "Total Charges"
COLUMNS_TO_DROP = ("Churn Score", "Count")
DEFAULT_PREPROCESSING_PIPELINE_PATH = Path(
    "models/preprocessing/churn_preprocessing_pipeline_v1.joblib"
)
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

    x = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    return x, y


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


def split_train_val_test(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = RANDOM_SEED,
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


def fit_scaler(x_train: pd.DataFrame) -> StandardScaler:
    """
    Ajusta o scaler somente nos dados de treino.
    """
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler


def transform_features(
    scaler: StandardScaler,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[ProcessedArray, ProcessedArray, ProcessedArray]:
    """
    Aplica o scaler já ajustado em treino, validação e teste.
    """
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_val_scaled, x_test_scaled


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

    x, y = split_features_target(df=df, target_col=target_col)

    validate_numeric_features(x)

    feature_names = x.columns.tolist()

    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
        x=x,
        y=y,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
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


def build_preprocessing_artifact(
    scaler: StandardScaler,
    feature_names: list[str],
    target_col: str = TARGET_COLUMN,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> dict:
    """
    Monta o artefato serializável de pré-processamento para treino/inferência.

    Esse bundle representa o preprocessing fitado que será reutilizado pela API.
    """
    if not feature_names:
        raise ValueError("feature_names cannot be empty.")

    return {
        "pipeline_type": "mlp_preprocessing",
        "scaler": scaler,
        "feature_names": feature_names,
        "target_col": target_col,
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
    }


def save_preprocessing_pipeline(
    scaler: StandardScaler,
    feature_names: list[str],
    output_path: Path | str = DEFAULT_PREPROCESSING_PIPELINE_PATH,
    target_col: str = TARGET_COLUMN,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Path:
    """
    Salva o bundle de pré-processamento fitado em joblib.

    Salva apenas artefato já fitado, pronto para ser reutilizado na inferência.
    """
    artifact = build_preprocessing_artifact(
        scaler=scaler,
        feature_names=feature_names,
        target_col=target_col,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, output_path)

    return output_path


def load_preprocessing_pipeline(
    input_path: Path | str = DEFAULT_PREPROCESSING_PIPELINE_PATH,
) -> dict:
    """
    Carrega o bundle de pré-processamento salvo em joblib.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Preprocessing pipeline artifact not found at: {input_path}"
        )

    artifact = joblib.load(input_path)

    if not isinstance(artifact, dict):
        raise ValueError(
            "Loaded preprocessing artifact is invalid. Expected a dictionary."
        )

    required_keys = {
        "pipeline_type",
        "scaler",
        "feature_names",
        "target_col",
        "seed",
        "test_size",
        "val_size",
    }

    missing_keys = required_keys.difference(artifact.keys())
    if missing_keys:
        raise ValueError(
            "Loaded preprocessing artifact is invalid. "
            f"Missing keys: {sorted(missing_keys)}"
        )

    pipeline_type = artifact["pipeline_type"]
    if pipeline_type != "mlp_preprocessing":
        raise ValueError(
            "Loaded preprocessing artifact is invalid. Unexpected pipeline_type."
        )

    feature_names = artifact["feature_names"]
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(
            "Loaded preprocessing artifact is invalid. "
            "'feature_names' must be a non-empty list."
        )

    scaler = artifact["scaler"]
    if not isinstance(scaler, StandardScaler):
        raise ValueError(
            "Loaded preprocessing artifact is invalid. "
            "'scaler' must be a fitted StandardScaler."
        )

    return artifact
