from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import TARGET_COLUMN
from src.data.load_data import load_csv_data
from src.data.preprocessing import (
    fit_tabular_preprocessing_pipeline,
    save_tabular_preprocessing_pipeline,
    select_tabular_raw_features,
    split_tabular_features_target,
    transform_tabular_features,
)

INPUT_DATASET_PATH = Path("data/raw/telco_customer_churn.csv")
OUTPUT_DATA_DIR = Path("data/processed")
OUTPUT_MODEL_DIR = Path("models")


def load_raw_dataset(input_path: Path) -> pd.DataFrame:
    """
    Carrega dataset codificado usando o loader padrão do projeto.

    Se o loader retornar dataframe com separador incorreto (coluna única),
    aplica fallback para leitura padrão com vírgula.
    """
    df = load_csv_data(str(input_path))
    if df is None:
        raise ValueError(f"Failed to load dataset from: {input_path}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    return df


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

    df = load_raw_dataset(INPUT_DATASET_PATH)
    df = select_tabular_raw_features(df, require_target=True)

    x, y = split_tabular_features_target(df)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.4,
        stratify=y,
        random_state=42,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    preprocessing_pipeline, feature_names = fit_tabular_preprocessing_pipeline(x_train)

    x_train = transform_tabular_features(preprocessing_pipeline, x_train)
    x_val = transform_tabular_features(preprocessing_pipeline, x_val)
    x_test = transform_tabular_features(preprocessing_pipeline, x_test)

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

    pipeline_path = save_tabular_preprocessing_pipeline(
        preprocessing_pipeline=preprocessing_pipeline,
        feature_names=feature_names,
        output_path=OUTPUT_MODEL_DIR
        / "preprocessing/churn_tabular_preprocessing_pipeline_v2.joblib",
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
