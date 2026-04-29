from pathlib import Path

import pandas as pd

from src.data.generate_processed_dataset import (
    load_raw_dataset,
    main,
)


RAW_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_customer_churn.csv"


def make_encoded_dataframe() -> pd.DataFrame:
    return pd.read_csv(RAW_DATASET_PATH, sep=";").iloc[:12].copy()


def test_load_raw_dataset_with_fallback_csv(tmp_path) -> None:
    csv_path = tmp_path / "encoded.csv"
    make_encoded_dataframe().to_csv(csv_path, index=False, sep=";")

    df = load_raw_dataset(csv_path)

    assert df.shape[0] == 12
    assert "Churn Value" in df.columns


def test_main_generates_processed_outputs(tmp_path, monkeypatch, capsys) -> None:
    input_csv = tmp_path / "encoded.csv"
    output_data_dir = tmp_path / "processed"
    output_model_dir = tmp_path / "models"

    make_encoded_dataframe().to_csv(input_csv, index=False, sep=";")

    monkeypatch.setattr(
        "src.data.generate_processed_dataset.INPUT_DATASET_PATH",
        input_csv,
    )
    monkeypatch.setattr(
        "src.data.generate_processed_dataset.OUTPUT_DATA_DIR",
        output_data_dir,
    )
    monkeypatch.setattr(
        "src.data.generate_processed_dataset.OUTPUT_MODEL_DIR",
        output_model_dir,
    )

    main()

    assert (output_data_dir / "X_train_processed.csv").exists()
    assert (output_data_dir / "X_val_processed.csv").exists()
    assert (output_data_dir / "X_test_processed.csv").exists()
    assert (output_data_dir / "y_train.csv").exists()
    assert (output_data_dir / "y_val.csv").exists()
    assert (output_data_dir / "y_test.csv").exists()
    assert (
        output_model_dir / "preprocessing/churn_tabular_preprocessing_pipeline_v2.joblib"
    ).exists()

    captured = capsys.readouterr()
    assert "[START] Generating processed dataset artifacts..." in captured.out
    assert "[DONE] Processed datasets generated successfully." in captured.out


def test_load_raw_dataset_raises_when_target_not_found(tmp_path) -> None:
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame(
        {
            "feat_1": [1, 2, 3],
            "feat_2": [4, 5, 6],
        }
    ).to_csv(csv_path, index=False, sep=";")

    import pytest

    with pytest.raises(ValueError, match="Target column not found"):
        load_raw_dataset(csv_path)
