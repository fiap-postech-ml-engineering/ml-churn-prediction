import pandas as pd
from collections.abc import Sequence

from src.config.settings import RAW_CATEGORICAL_FEATURES


def apply_one_hot_encoding(
    df: pd.DataFrame,
    categorical_features: Sequence[str] = RAW_CATEGORICAL_FEATURES,
) -> pd.DataFrame:
    """
    Aplica one-hot encoding nas colunas categóricas especificadas.
    """
    categorical_features = list(categorical_features)

    return pd.get_dummies(df, columns=categorical_features, drop_first=True)