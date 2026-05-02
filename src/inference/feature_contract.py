from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.features.apply_one_hot_encoding import apply_one_hot_encoding
from src.features.feature_engineering import (
    REQUIRED_BASE_COLUMNS,
    SERVICE_BINARY_COLUMNS,
    apply_feature_engineering,
)
from src.features.missing_values import clean_missing_values
from src.features.select_features import select_model_features
from src.features.type_features import cast_feature_types

FEATURE_ENGINEERING_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "Internet Service_Fiber optic": ("Internet Service", "Fiber optic"),
    "Online Security_Yes": ("Online Security", "Yes"),
    "Online Backup_Yes": ("Online Backup", "Yes"),
    "Device Protection_Yes": ("Device Protection", "Yes"),
    "Tech Support_Yes": ("Tech Support", "Yes"),
    "Streaming TV_Yes": ("Streaming TV", "Yes"),
    "Streaming Movies_Yes": ("Streaming Movies", "Yes"),
}

CRITICAL_RAW_INFERENCE_FEATURES: tuple[str, ...] = (
    "Tenure Months",
    "Total Charges",
    "Monthly Charges",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
)


def build_raw_inference_feature_names(
    selected_features: Sequence[str],
    target_column: str,
) -> list[str]:
    """Retorna features RAW de inferencia sem a coluna-alvo."""
    return [feature for feature in selected_features if feature != target_column]


def align_to_model_feature_contract(
    df_features: pd.DataFrame,
    model_feature_names: Sequence[str],
) -> pd.DataFrame:
    """Alinha DataFrame para o contrato de features esperado pelo modelo.

    - Remove colunas extras
    - Cria colunas faltantes com 0.0
    - Reordena conforme `model_feature_names`
    - Coage valores para numerico
    """
    if not model_feature_names:
        raise ValueError("model_feature_names cannot be empty")

    aligned_df = df_features.reindex(columns=list(model_feature_names), fill_value=0.0)
    aligned_df = aligned_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return aligned_df.astype("float64")


def validate_critical_raw_inference_features(
    df_raw_features: pd.DataFrame,
    critical_features: Sequence[str] = CRITICAL_RAW_INFERENCE_FEATURES,
) -> None:
    """Valida campos RAW críticos para evitar inferência semanticamente incompleta."""
    missing_columns = [
        feature for feature in critical_features if feature not in df_raw_features.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing critical raw inference features: "
            f"{sorted(missing_columns)}"
        )

    missing_values = []
    for feature in critical_features:
        series = df_raw_features[feature]
        as_string = series.astype("string")
        has_empty_string = as_string.str.strip().eq("").any()
        if series.isna().any() or has_empty_string:
            missing_values.append(feature)

    if missing_values:
        raise ValueError(
            "Critical raw inference features with missing values: "
            f"{sorted(missing_values)}"
        )


def ensure_feature_engineering_columns(
    df_encoded: pd.DataFrame,
    df_raw_features: pd.DataFrame,
) -> pd.DataFrame:
    """Garante colunas esperadas pela engenharia de features.

    O fluxo de inferencia pode chegar com 1 linha; nesse caso o one-hot com
    drop_first remove colunas importantes. Esta funcao recompõe as colunas
    minimas derivadas das features RAW.
    """
    output_df = df_encoded.copy()

    for encoded_column, (raw_column, positive_value) in FEATURE_ENGINEERING_CATEGORY_MAP.items():
        if encoded_column in output_df.columns:
            continue
        if raw_column not in df_raw_features.columns:
            output_df[encoded_column] = 0
            continue
        output_df[encoded_column] = (
            df_raw_features[raw_column].astype("string").str.lower()
            == str(positive_value).lower()
        ).astype(int)

    return output_df


def ensure_model_one_hot_columns(
    df_encoded: pd.DataFrame,
    df_raw_features: pd.DataFrame,
    model_feature_names: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """
    Reconstrói colunas one-hot esperadas pelo modelo a partir das features RAW.

    Necessário para inferência com um único registro, porque
    pd.get_dummies(..., drop_first=True) pode não criar colunas categóricas
    quando existe apenas uma categoria presente no batch.

    Exemplo:
    RAW:
        Paperless Billing = Yes

    Feature esperada pelo modelo:
        Paperless Billing_Yes = 1
    """
    output_df = df_encoded.copy()

    for model_feature in model_feature_names:
        if model_feature in output_df.columns:
            continue

        for raw_column in categorical_features:
            prefix = f"{raw_column}_"

            if not model_feature.startswith(prefix):
                continue

            if raw_column not in df_raw_features.columns:
                output_df[model_feature] = 0
                break

            expected_category = model_feature.removeprefix(prefix)

            output_df[model_feature] = (
                df_raw_features[raw_column]
                .astype("string")
                .str.strip()
                .str.lower()
                .eq(str(expected_category).strip().lower())
                .astype(int)
            )

            break

    return output_df


def build_model_ready_inference_features(
    df_raw_features: pd.DataFrame,
    model_feature_names: Sequence[str],
    selected_raw_features: Sequence[str],
    raw_int_features: Sequence[str],
    raw_float_features: Sequence[str],
    raw_categorical_features: Sequence[str],
    strict_critical_raw_validation: bool = True,
) -> pd.DataFrame:
    """Executa o fluxo RAW -> contrato do modelo em um único ponto."""
    if strict_critical_raw_validation:
        validate_critical_raw_inference_features(df_raw_features=df_raw_features)

    df_selected_features = select_model_features(
        df_raw_features,
        selected_features=selected_raw_features,
        strict=False,
    )

    df_typed_features = cast_feature_types(
        df_selected_features,
        int_features=raw_int_features,
        float_features=raw_float_features,
        categorical_features=raw_categorical_features,
        strict=False,
    )

    df_no_missing_values = clean_missing_values(
        df_typed_features,
        selected_features=selected_raw_features,
        int_features=raw_int_features,
        float_features=raw_float_features,
        categorical_features=raw_categorical_features,
        strict=False,
    )

    df_ohe = apply_one_hot_encoding(df_no_missing_values)
    df_ohe_with_model_columns = ensure_model_one_hot_columns(
        df_encoded=df_ohe,
        df_raw_features=df_no_missing_values,
        model_feature_names=model_feature_names,
        categorical_features=raw_categorical_features,
    )

    df_ohe_ready_for_feat_eng = ensure_feature_engineering_columns(
        df_encoded=df_ohe_with_model_columns,
        df_raw_features=df_no_missing_values,
    )

    for required_column in (*REQUIRED_BASE_COLUMNS, *SERVICE_BINARY_COLUMNS):
        if required_column not in df_ohe_ready_for_feat_eng.columns:
            df_ohe_ready_for_feat_eng[required_column] = 0.0

    df_feat_eng = apply_feature_engineering(df_ohe_ready_for_feat_eng)
    return align_to_model_feature_contract(
        df_feat_eng,
        model_feature_names=model_feature_names,
    )
