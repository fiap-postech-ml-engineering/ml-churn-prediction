import pandas as pd

from src.config.settings import TABULAR_RAW_FEATURES, TARGET_COLUMN


def select_tabular_raw_features(
    df: pd.DataFrame,
    require_target: bool = False,
) -> pd.DataFrame:
    """Seleciona o contrato raw oficial usado pelo pipeline tabular final."""
    required_columns = list(TABULAR_RAW_FEATURES)

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing critical raw inference features: " f"{sorted(missing_columns)}"
        )

    selected_df = df[required_columns].copy()

    if require_target:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")
        selected_df[TARGET_COLUMN] = df[TARGET_COLUMN].copy()

    return selected_df
