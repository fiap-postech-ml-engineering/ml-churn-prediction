from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import settings


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuração de preprocessing com defaults vindos de settings."""

    target_source_column: str = settings.TARGET_COLUMN
    target_column: str = settings.PREPROCESSING_TARGET_COLUMN
    total_charges_column: str = settings.TOTAL_CHARGES_COLUMN
    columns_to_drop: tuple[str, ...] = settings.PREPROCESSING_COLUMNS_TO_DROP
    pipeline_path: Path = settings.PREPROCESSING_PIPELINE_PATH
    random_seed: int = settings.RANDOM_SEED
    test_size: float = settings.TEST_SIZE
    val_size: float = settings.VALIDATION_SIZE


DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()

# Backward compatibility for modules importing these constants directly.
TARGET_SOURCE_COLUMN = DEFAULT_PREPROCESSING_CONFIG.target_source_column
TARGET_COLUMN = DEFAULT_PREPROCESSING_CONFIG.target_column
TOTAL_CHARGES_COLUMN = DEFAULT_PREPROCESSING_CONFIG.total_charges_column
COLUMNS_TO_DROP = DEFAULT_PREPROCESSING_CONFIG.columns_to_drop
DEFAULT_PREPROCESSING_PIPELINE_PATH = DEFAULT_PREPROCESSING_CONFIG.pipeline_path
ProcessedArray = np.ndarray


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


def build_preprocessing_artifact(
    scaler: StandardScaler,
    feature_names: list[str],
    target_col: str | None = None,
    seed: int | None = None,
    test_size: float | None = None,
    val_size: float | None = None,
    config: PreprocessingConfig | None = None,
) -> dict:
    """
    Monta o artefato serializável de pré-processamento para treino/inferência.

    Esse bundle representa o preprocessing fitado que será reutilizado pela API.
    """
    cfg = config or DEFAULT_PREPROCESSING_CONFIG
    target_col = cfg.target_column if target_col is None else target_col
    seed = cfg.random_seed if seed is None else seed
    test_size = cfg.test_size if test_size is None else test_size
    val_size = cfg.val_size if val_size is None else val_size

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
    output_path: Path | str | None = None,
    target_col: str | None = None,
    seed: int | None = None,
    test_size: float | None = None,
    val_size: float | None = None,
    config: PreprocessingConfig | None = None,
) -> Path:
    """
    Salva o bundle de pré-processamento fitado em joblib.

    Salva apenas artefato já fitado, pronto para ser reutilizado na inferência.
    """
    cfg = config or DEFAULT_PREPROCESSING_CONFIG
    output_path = cfg.pipeline_path if output_path is None else Path(output_path)

    artifact = build_preprocessing_artifact(
        scaler=scaler,
        feature_names=feature_names,
        target_col=target_col,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
        config=cfg,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, output_path)

    return output_path


def load_preprocessing_pipeline(
    input_path: Path | str | None = None,
    config: PreprocessingConfig | None = None,
) -> dict:
    """
    Carrega o bundle de pré-processamento salvo em joblib.
    """
    cfg = config or DEFAULT_PREPROCESSING_CONFIG
    input_path = cfg.pipeline_path if input_path is None else Path(input_path)

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
