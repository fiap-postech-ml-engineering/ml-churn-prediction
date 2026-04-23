from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.config.settings import (
    RAW_CATEGORICAL_FEATURES,
    RAW_FLOAT_FEATURES,
    RAW_INT_FEATURES,
    SELECTED_FEATURES,
)


def _validate_columns_exist(
    df: pd.DataFrame,
    columns: Sequence[str],
    group_name: str,
    strict: bool,
) -> list[str]:
    missing = [column for column in columns if column not in df.columns]
    if strict and missing:
        raise ValueError(
            f"Missing {group_name} columns for missing-value handling: {sorted(missing)}"
        )
    return [column for column in columns if column in df.columns]


def _normalize_missing_markers(
    df: pd.DataFrame,
    categorical_features: Sequence[str],
    extra_markers: Sequence[str],
) -> pd.DataFrame:
    """
    Padroniza faltantes em colunas categóricas para <NA>.
    """
    normalized_df = df.copy()
    if not categorical_features:
        return normalized_df

    marker_set = {marker.strip().lower() for marker in extra_markers if marker.strip()}

    for column in categorical_features:
        series = normalized_df[column].astype("string")
        stripped = series.str.strip()

        # Vazios/espacos -> <NA>
        series = series.mask(stripped.eq(""), pd.NA)

        # Marcadores textuais (unknown, n/a, etc.) -> <NA>
        lower_stripped = stripped.str.lower()
        series = series.mask(lower_stripped.isin(marker_set), pd.NA)

        normalized_df[column] = series

    return normalized_df


def clean_missing_values(
    df: pd.DataFrame,
    selected_features: Sequence[str] | None = None,
    int_features: Sequence[str] | None = None,
    float_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    strict: bool = True,
    categorical_fill_value: str = "missing",
    extra_missing_markers: Sequence[str] = ("unknown", "n/a", "na", "none", "null"),
) -> pd.DataFrame:
    """
    Seleciona features e aplica tratamento direto de missing values.

    Fluxo:
    1. Seleciona apenas as features definidas para o modelo.
    2. Padroniza marcadores de missing em colunas categóricas para <NA>.
    3. Imputa numéricas com 0.
    4. Imputa categóricas com valor sentinela (default: "missing").
    """
    features = list(selected_features or SELECTED_FEATURES)
    if not features:
        raise ValueError("selected_features cannot be empty.")

    missing_features = [feature for feature in features if feature not in df.columns]
    if strict and missing_features:
        raise ValueError(f"Missing selected features: {sorted(missing_features)}")

    available_features = [feature for feature in features if feature in df.columns]
    cleaned_df = df[available_features].copy()

    resolved_int_features = list(int_features or RAW_INT_FEATURES)
    resolved_float_features = list(float_features or RAW_FLOAT_FEATURES)
    resolved_categorical_features = list(categorical_features or RAW_CATEGORICAL_FEATURES)

    available_int = _validate_columns_exist(cleaned_df, resolved_int_features, "int", strict)
    available_float = _validate_columns_exist(
        cleaned_df,
        resolved_float_features,
        "float",
        strict,
    )
    available_categorical = _validate_columns_exist(
        cleaned_df,
        resolved_categorical_features,
        "categorical",
        strict,
    )

    numeric_features = available_int + available_float

    cleaned_df = _normalize_missing_markers(
        df=cleaned_df,
        categorical_features=available_categorical,
        extra_markers=extra_missing_markers,
    )

    for column in numeric_features:
        numeric_series = pd.to_numeric(cleaned_df[column], errors="coerce")
        cleaned_df[column] = numeric_series.fillna(0)

    for column in available_categorical:
        cleaned_df[column] = cleaned_df[column].astype("string").fillna(categorical_fill_value)

    return cleaned_df