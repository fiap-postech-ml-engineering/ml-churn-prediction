from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.config.settings import (
    FEATURE_TYPE_STRICT,
    RAW_CATEGORICAL_FEATURES,
    RAW_FLOAT_FEATURES,
    RAW_INT_FEATURES,
)


def _validate_columns_exist(
    df: pd.DataFrame,
    columns: Sequence[str],
    group_name: str,
    strict: bool,
) -> list[str]:
    missing = [column for column in columns if column not in df.columns]
    if strict and missing:
        raise ValueError(f"Missing {group_name} columns for typing: {sorted(missing)}")
    return [column for column in columns if column in df.columns]


def cast_feature_types(
    df: pd.DataFrame,
    int_features: Sequence[str] | None = None,
    float_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    strict: bool = FEATURE_TYPE_STRICT,
) -> pd.DataFrame:
    """
    Aplica tipagem das features conforme schema definido no settings.

    Regras padrão:
    - int_features -> int64
    - float_features -> float64
    - categorical_features -> string
    """
    typed_df = df.copy()

    resolved_int_features = list(int_features or RAW_INT_FEATURES)
    resolved_float_features = list(float_features or RAW_FLOAT_FEATURES)
    resolved_categorical_features = list(categorical_features or RAW_CATEGORICAL_FEATURES)

    available_int_features = _validate_columns_exist(
        typed_df,
        resolved_int_features,
        group_name="int",
        strict=strict,
    )
    available_float_features = _validate_columns_exist(
        typed_df,
        resolved_float_features,
        group_name="float",
        strict=strict,
    )
    available_categorical_features = _validate_columns_exist(
        typed_df,
        resolved_categorical_features,
        group_name="categorical",
        strict=strict,
    )

    for column in available_int_features:
        typed_df[column] = pd.to_numeric(typed_df[column], errors="coerce")
        if strict and typed_df[column].isna().any():
            raise ValueError(
                f"Column '{column}' has invalid values for int casting."
            )
        typed_df[column] = typed_df[column].astype("int64")

    for column in available_float_features:
        typed_df[column] = pd.to_numeric(typed_df[column], errors="coerce")
        if strict and typed_df[column].isna().any():
            raise ValueError(
                f"Column '{column}' has invalid values for float casting."
            )
        typed_df[column] = typed_df[column].astype("float64")

    for column in available_categorical_features:
        typed_df[column] = typed_df[column].astype("string")

    return typed_df
