from pathlib import Path

import joblib
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig,
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


def try_load_preprocessing_pipeline(
    input_path: Path | str | None = None,
    config: PreprocessingConfig | None = None,
) -> dict | None:
    """Tenta carregar o preprocessing salvo, retornando None quando não existir."""
    try:
        return load_preprocessing_pipeline(input_path=input_path, config=config)
    except FileNotFoundError:
        return None