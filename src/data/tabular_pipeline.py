from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings
from src.data.preprocessing_config import TABULAR_PIPELINE_TYPE, TABULAR_TARGET_COLUMN
from src.features.feature_engineering import TabularFeatureEngineer


def split_tabular_features_target(
    df: pd.DataFrame,
    target_col: str = TABULAR_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    x = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return x, y


def build_tabular_preprocessing_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Cria o pipeline sklearn único para treino e inferência."""
    resolved_numeric_features = numeric_features or list(settings.TABULAR_RAW_NUMERIC_FEATURES) + list(settings.TABULAR_DERIVED_FEATURES)
    resolved_categorical_features = categorical_features or list(settings.TABULAR_RAW_CATEGORICAL_FEATURES)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, resolved_numeric_features),
            (
                "categorical",
                categorical_pipeline,
                resolved_categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", TabularFeatureEngineer()),
            ("preprocessor", preprocessor),
        ]
    )


def fit_tabular_preprocessing_pipeline(
    x_train: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> tuple[Pipeline, list[str]]:
    pipeline = build_tabular_preprocessing_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    pipeline.fit(x_train)
    feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    return pipeline, feature_names


def transform_tabular_features(
    pipeline: Pipeline,
    x: pd.DataFrame,
) -> np.ndarray:
    transformed = pipeline.transform(x)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed)


def build_tabular_preprocessing_artifact(
    preprocessing_pipeline: Pipeline,
    feature_names: list[str],
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    target_col: str = TABULAR_TARGET_COLUMN,
) -> dict:
    if not feature_names:
        raise ValueError("feature_names cannot be empty.")

    return {
        "pipeline_type": TABULAR_PIPELINE_TYPE,
        "preprocessing_pipeline": preprocessing_pipeline,
        "feature_names": feature_names,
        "numeric_features": numeric_features
        or list(settings.TABULAR_RAW_NUMERIC_FEATURES)
        + list(settings.TABULAR_DERIVED_FEATURES),
        "categorical_features": categorical_features
        or list(settings.TABULAR_RAW_CATEGORICAL_FEATURES),
        "target_col": target_col,
    }


def save_tabular_preprocessing_pipeline(
    preprocessing_pipeline: Pipeline,
    feature_names: list[str],
    output_path: Path | str | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    target_col: str = TABULAR_TARGET_COLUMN,
) -> Path:
    output_path = (
        settings.TABULAR_PREPROCESSING_PIPELINE_PATH
        if output_path is None
        else Path(output_path)
    )

    artifact = build_tabular_preprocessing_artifact(
        preprocessing_pipeline=preprocessing_pipeline,
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=target_col,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return output_path


def load_tabular_preprocessing_pipeline(
    input_path: Path | str | None = None,
) -> dict:
    input_path = (
        settings.TABULAR_PREPROCESSING_PIPELINE_PATH
        if input_path is None
        else Path(input_path)
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Tabular preprocessing pipeline artifact not found at: {input_path}"
        )

    artifact = joblib.load(input_path)
    if not isinstance(artifact, dict):
        raise ValueError(
            "Loaded tabular preprocessing artifact is invalid. Expected a dictionary."
        )

    required_keys = {
        "pipeline_type",
        "preprocessing_pipeline",
        "feature_names",
        "numeric_features",
        "categorical_features",
        "target_col",
    }
    missing_keys = required_keys.difference(artifact.keys())
    if missing_keys:
        raise ValueError(
            "Loaded tabular preprocessing artifact is invalid. "
            f"Missing keys: {sorted(missing_keys)}"
        )

    if artifact["pipeline_type"] != TABULAR_PIPELINE_TYPE:
        raise ValueError(
            "Loaded tabular preprocessing artifact is invalid. Unexpected pipeline_type."
        )

    feature_names = artifact["feature_names"]
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(
            "Loaded tabular preprocessing artifact is invalid. "
            "'feature_names' must be a non-empty list."
        )

    preprocessing_pipeline = artifact["preprocessing_pipeline"]
    if not hasattr(preprocessing_pipeline, "transform"):
        raise ValueError(
            "Loaded tabular preprocessing artifact is invalid. "
            "'preprocessing_pipeline' must expose transform()."
        )

    return artifact


def try_load_tabular_preprocessing_pipeline(
    input_path: Path | str | None = None,
) -> dict | None:
    try:
        return load_tabular_preprocessing_pipeline(input_path=input_path)
    except FileNotFoundError:
        return None
