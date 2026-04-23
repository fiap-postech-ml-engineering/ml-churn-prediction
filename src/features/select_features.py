from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.config.settings import SELECTED_FEATURES


def validate_required_features(
    df: pd.DataFrame,
    required_features: Sequence[str],
) -> list[str]:
    """
    Retorna a lista de colunas ausentes no dataframe.
    """
    return [feature for feature in required_features if feature not in df.columns]


def select_model_features(
    df: pd.DataFrame,
    selected_features: Sequence[str] | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Seleciona as features de entrada do modelo conforme schema configurado.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    selected_features : Sequence[str] | None
        Lista de features a selecionar. Se None, usa SELECTED_FEATURES do settings.
    strict : bool
        Se True, falha quando alguma coluna esperada estiver ausente.

    Returns
    -------
    pd.DataFrame
        DataFrame contendo apenas as colunas selecionadas e existentes.
    """
    features = list(selected_features or SELECTED_FEATURES)

    if not features:
        raise ValueError("selected_features cannot be empty.")

    missing_features = validate_required_features(df=df, required_features=features)
    if strict and missing_features:
        raise ValueError(
            "Missing required features for selection: "
            f"{sorted(missing_features)}"
        )

    available_features = [feature for feature in features if feature in df.columns]
    return df[available_features].copy()
