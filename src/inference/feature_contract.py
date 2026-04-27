from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

FEATURE_ENGINEERING_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "Internet Service_Fiber optic": ("Internet Service", "Fiber optic"),
    "Online Security_Yes": ("Online Security", "Yes"),
    "Online Backup_Yes": ("Online Backup", "Yes"),
    "Device Protection_Yes": ("Device Protection", "Yes"),
    "Tech Support_Yes": ("Tech Support", "Yes"),
    "Streaming TV_Yes": ("Streaming TV", "Yes"),
    "Streaming Movies_Yes": ("Streaming Movies", "Yes"),
}


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
