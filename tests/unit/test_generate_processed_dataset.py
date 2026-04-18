from pathlib import Path

import pandas as pd

from src.data.generate_processed_dataset import (
    load_encoded_dataset,
    main,
)


def make_encoded_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feat_1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "feat_2": [10, 20, 15, 30, 18, 35, 12, 40, 14, 45, 16, 50],
            "feat_3": [100, 120, 90, 130, 110, 150, 95, 160, 105, 170, 115, 180],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_load_encoded_dataset_with_fallback_csv(tmp_path, capsys) -> None:
    csv_path = tmp_path / "encoded.csv"
    make_encoded_dataframe().to_csv(csv_path, index=False)

    df = load_encoded_dataset(csv_path)

    assert df.shape == (12, 4)
    assert "target" in df.columns

    captured = capsys.readouterr()
    assert "Trying fallback CSV read with default comma separator" in captured.out
    assert "Fallback load succeeded" in captured.out


def test_main_generates_processed_outputs(tmp_path, monkeypatch, capsys) -> None:
    input_csv = tmp_path / "encoded.csv"
    output_data_dir = tmp_path / "processed"
    output_model_dir = tmp_path / "models"

    make_encoded_dataframe().to_csv(input_csv, index=False)

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
        output_model_dir / "preprocessing/churn_preprocessing_pipeline_v1.joblib"
    ).exists()

    captured = capsys.readouterr()
    assert "[START] Generating processed dataset artifacts..." in captured.out
    assert "[DONE] Processed datasets generated successfully." in captured.out


def test_load_encoded_dataset_raises_when_target_not_found(tmp_path) -> None:
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame(
        {
            "feat_1": [1, 2, 3],
            "feat_2": [4, 5, 6],
        }
    ).to_csv(csv_path, index=False)

    import pytest

    with pytest.raises(ValueError, match="Target column not found"):
        load_encoded_dataset(csv_path)
