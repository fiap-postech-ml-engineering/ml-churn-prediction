from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from src.data.load_data import load_csv_data
from src.data.preprocessing import (
    prepare_mlp_data,
    save_preprocessing_pipeline,
    TARGET_COLUMN,
    TARGET_SOURCE_COLUMN,
)


INPUT_DATASET_PATH = Path(
    "data/processed/telco_customer_churn_eda_pre-processed_encoded.csv"
)
OUTPUT_DATA_DIR = Path("data/processed")
OUTPUT_MODEL_DIR = Path("models")


def load_encoded_dataset(input_path: Path) -> pd.DataFrame:
    """
    Carrega dataset codificado usando o loader padrão do projeto.

    Se o loader retornar dataframe com separador incorreto (coluna única),
    aplica fallback para leitura padrão com vírgula.
    """
    df = load_csv_data(str(input_path))
    if df is None:
        raise ValueError(f"Failed to load dataset from: {input_path}")

    if TARGET_COLUMN in df.columns or TARGET_SOURCE_COLUMN in df.columns:
        return df

    print(
    "[INFO] Target column not found with project loader. "
    "Trying fallback CSV read with default comma separator..."
    )
    df_fallback = pd.read_csv(input_path)

    if (
        TARGET_COLUMN not in df_fallback.columns
        and TARGET_SOURCE_COLUMN not in df_fallback.columns
    ):
        raise ValueError(
            "Target column not found after loading dataset. "
            f"Expected '{TARGET_COLUMN}' or '{TARGET_SOURCE_COLUMN}'."
        )

    print(f"[INFO] Fallback load succeeded. Dataframe shape: {df_fallback.shape}")
    return df_fallback


def save_array_as_dataframe(
    data: np.ndarray,
    feature_names: list[str],
    output_path: Path,
) -> None:
    df = pd.DataFrame(data, columns=feature_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_target(y: pd.Series, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y.to_frame(name="target").to_csv(output_path, index=False)


def main() -> None:
    print("[START] Generating processed dataset artifacts...")

    df = load_encoded_dataset(INPUT_DATASET_PATH)

    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        scaler,
        feature_names,
    ) = prepare_mlp_data(df)

    save_array_as_dataframe(
        x_train,
        feature_names,
        OUTPUT_DATA_DIR / "X_train_processed.csv",
    )
    save_array_as_dataframe(
        x_val,
        feature_names,
        OUTPUT_DATA_DIR / "X_val_processed.csv",
    )
    save_array_as_dataframe(
        x_test,
        feature_names,
        OUTPUT_DATA_DIR / "X_test_processed.csv",
    )

    save_target(y_train, OUTPUT_DATA_DIR / "y_train.csv")
    save_target(y_val, OUTPUT_DATA_DIR / "y_val.csv")
    save_target(y_test, OUTPUT_DATA_DIR / "y_test.csv")

    pipeline_path = save_preprocessing_pipeline(
        scaler=scaler,
        feature_names=feature_names,
        output_path=OUTPUT_MODEL_DIR / "preprocessing_pipeline.joblib",
    )

    print("[DONE] Processed datasets generated successfully.")
    print(f"[INFO] X_train saved to: {OUTPUT_DATA_DIR / 'X_train_processed.csv'}")
    print(f"[INFO] X_val saved to: {OUTPUT_DATA_DIR / 'X_val_processed.csv'}")
    print(f"[INFO] X_test saved to: {OUTPUT_DATA_DIR / 'X_test_processed.csv'}")
    print(f"[INFO] y_train saved to: {OUTPUT_DATA_DIR / 'y_train.csv'}")
    print(f"[INFO] y_val saved to: {OUTPUT_DATA_DIR / 'y_val.csv'}")
    print(f"[INFO] y_test saved to: {OUTPUT_DATA_DIR / 'y_test.csv'}")
    print(f"[INFO] preprocessing pipeline saved to: {pipeline_path}")


if __name__ == "__main__":
    main()
