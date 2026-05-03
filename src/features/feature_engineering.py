from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.settings import (
    TABULAR_DERIVED_FEATURES,
    TABULAR_RAW_CATEGORICAL_FEATURES,
    TABULAR_RAW_NUMERIC_FEATURES,
)

SERVICE_COLUMNS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]

REQUIRED_BASE_COLUMNS = (
    "Tenure Months",
    "Total Charges",
    "Monthly Charges",
    "Internet Service_Fiber optic",
)

SERVICE_BINARY_COLUMNS = (
    "Online Security_Yes",
    "Online Backup_Yes",
    "Device Protection_Yes",
    "Tech Support_Yes",
    "Streaming TV_Yes",
    "Streaming Movies_Yes",
)


def _safe_log1p(series: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    positive_series = numeric_series.where(
        numeric_series.isna() | (numeric_series >= 0)
    )
    return np.log1p(positive_series)


class TabularFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria as features derivadas oficiais usadas no pipeline final."""

    def __init__(
        self,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.numeric_features = numeric_features or list(TABULAR_RAW_NUMERIC_FEATURES)
        self.categorical_features = categorical_features or list(
            TABULAR_RAW_CATEGORICAL_FEATURES
        )

    def fit(self, x: pd.DataFrame, y=None):
        self.feature_names_in_ = list(x.columns)
        self.output_feature_names_ = (
            list(TABULAR_RAW_NUMERIC_FEATURES)
            + list(TABULAR_RAW_CATEGORICAL_FEATURES)
            + list(TABULAR_DERIVED_FEATURES)
        )
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x, columns=self.feature_names_in_)

        df = x.copy()

        missing_columns = [
            column
            for column in list(TABULAR_RAW_NUMERIC_FEATURES)
            + list(TABULAR_RAW_CATEGORICAL_FEATURES)
            if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                "Missing required raw columns for feature engineering: "
                f"{sorted(missing_columns)}"
            )

        tenure_months = pd.to_numeric(df["Tenure Months"], errors="coerce")
        total_charges = pd.to_numeric(df["Total Charges"], errors="coerce")
        monthly_charges = pd.to_numeric(df["Monthly Charges"], errors="coerce")
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df["Tenure Months"] = tenure_months
        df["Monthly Charges"] = monthly_charges
        df["Total Charges"] = total_charges
        df["CLTV"] = pd.to_numeric(df["CLTV"], errors="coerce")

        service_flags = []
        for column in SERVICE_COLUMNS:
            service_flags.append(
                df[column]
                .astype("string")
                .str.strip()
                .str.lower()
                .eq("yes")
                .fillna(False)
                .astype(int)
            )

        df["total_services"] = sum(service_flags)
        fiber_flag = (
            df["Internet Service"]
            .astype("string")
            .str.strip()
            .str.lower()
            .eq("fiber optic")
            .fillna(False)
            .astype(int)
        )
        df["fiber_price_impact"] = fiber_flag * monthly_charges

        tenure_safe = tenure_months.replace(0, 1)
        avg_ticket = total_charges / tenure_safe
        df["avg_ticket"] = avg_ticket
        df["Total Charges Log"] = _safe_log1p(total_charges)
        df["avg_ticket_log"] = _safe_log1p(avg_ticket)
        df["is_new_customer"] = tenure_months.lt(6).fillna(False).astype(int)

        return df[
            list(TABULAR_RAW_NUMERIC_FEATURES)
            + list(TABULAR_RAW_CATEGORICAL_FEATURES)
            + list(TABULAR_DERIVED_FEATURES)
        ].copy()


def apply_feature_engineering(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aplica engenharia de atributos sobre o DataFrame e retorna o mesmo DataFrame com novas colunas.

    - Cria a coluna "Ternure Months" a partir de "Tenure Months", substituindo 0 por 1.
    - Cria "total_services" como soma das colunas de serviços existentes entre: "Online Security_Yes", "Online Backup_Yes", "Device Protection_Yes", "Tech Support_Yes", "Streaming TV_Yes" e "Streaming Movies_Yes".
    - Cria "fiber_price_impact" somente se "Internet Service_Fiber optic" existir, calculando a interação com "Monthly Charges".
    - Cria "avg_ticket" como "Total Charges" / "Tenure Months". Sobrescreve "Total Charges" com log1p de "Total Charges".
    - Cria "Monthly Charges_log" com log1p de "Monthly Charges".
    - Cria "is_new_customer" como flag inteira para "Tenure Months" < 6.
    Args:
    df (pd.DataFrame): DataFrame de entrada com as features necessárias para as
    transformações.

    Returns:
    pd.DataFrame: O próprio DataFrame de entrada, após mutações in-place, contendo
    as colunas originais e as colunas derivadas criadas.

    Raises:
    KeyError: Se alguma coluna obrigatória para as operações não existir, como
    "Tenure Months", "Total Charges" ou "Monthly Charges".
    TypeError: Se colunas usadas em operações matemáticas não tiverem tipo numérico.

    Notes:
    - A função modifica o DataFrame recebido (in-place) antes de retorná-lo.
    """

    needed_cols = list(REQUIRED_BASE_COLUMNS)
    service_cols = list(SERVICE_BINARY_COLUMNS)
    numeric_cols = ["Tenure Months", "Total Charges", "Monthly Charges"]

    missing_needed = [c for c in needed_cols if c not in df.columns]
    missing_service = [c for c in service_cols if c not in df.columns]
    not_numeric = [
        c
        for c in numeric_cols
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]

    if missing_needed or missing_service:
        raise KeyError(
            f"O Dataframe precisa desssas colunas para o tratamento: {needed_cols + service_cols}"
        )

    if not_numeric:
        raise TypeError(
            f"As seguintes colunas precisam ser numéricas para as operações matemáticas: {not_numeric}"
        )

    # 1. Ajuste de Tenure (evitar divisão por zero)
    # HACK O ideal seria criar uma coluna auxiliar para calcular o avg_ticket,
    # mas para evitar mudanças em outras partes do código, substituímos os zeros por 1 diretamente na coluna original.
    df['Tenure Months'] = df['Tenure Months'].replace(0, 1)

    # 2. Stickiness: Total de serviços adicionais ativos
    existing_services = [col for col in service_cols if col in df.columns]
    df["total_services"] = df[existing_services].sum(axis=1)

    # 3. Interação Fibra x Preço
    df["fiber_price_impact"] = (
        df["Internet Service_Fiber optic"] * df["Monthly Charges"]
    )

    # 4. Métricas Financeiras e Log para Normalização
    df["avg_ticket"] = df["Total Charges"] / df["Tenure Months"]
    df["Total Charges"] = np.log1p(df["Total Charges"])
    df["Monthly Charges_log"] = np.log1p(df["Monthly Charges"])

    # 5. Flag de cliente novo
    df["is_new_customer"] = (df["Tenure Months"] < 6).astype(int)

    return df
