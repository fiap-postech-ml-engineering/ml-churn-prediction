import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import (
    load_preprocessing_pipeline,
    prepare_mlp_data,
    save_preprocessing_pipeline,
)


def make_valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feat_1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "feat_2": [10, 20, 15, 30, 18, 35, 12, 40, 14, 45, 16, 50],
            "feat_3": [100, 120, 90, 130, 110, 150, 95, 160, 105, 170, 115, 180],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_prepare_mlp_data_returns_consistent_shapes() -> None:
    df = make_valid_dataframe()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        feature_names,
    ) = prepare_mlp_data(df, seed=42)

    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert X_train.shape[1] == len(feature_names)

    assert len(y_train) == X_train.shape[0]
    assert len(y_val) == X_val.shape[0]
    assert len(y_test) == X_test.shape[0]

    assert isinstance(scaler, StandardScaler)


def test_prepare_mlp_data_is_deterministic_with_same_seed() -> None:
    df = make_valid_dataframe()

    result_1 = prepare_mlp_data(df, seed=42)
    result_2 = prepare_mlp_data(df, seed=42)

    X_train_1, X_val_1, X_test_1, y_train_1, y_val_1, y_test_1, _, feature_names_1 = result_1
    X_train_2, X_val_2, X_test_2, y_train_2, y_val_2, y_test_2, _, feature_names_2 = result_2

    np.testing.assert_allclose(X_train_1, X_train_2)
    np.testing.assert_allclose(X_val_1, X_val_2)
    np.testing.assert_allclose(X_test_1, X_test_2)

    assert y_train_1.tolist() == y_train_2.tolist()
    assert y_val_1.tolist() == y_val_2.tolist()
    assert y_test_1.tolist() == y_test_2.tolist()

    assert feature_names_1 == feature_names_2


def test_prepare_mlp_data_preserves_target_distribution_reasonably() -> None:
    df = make_valid_dataframe()

    _, _, _, y_train, y_val, y_test, _, _ = prepare_mlp_data(df, seed=42)

    overall_rate = df["target"].mean()
    train_rate = y_train.mean()
    val_rate = y_val.mean()
    test_rate = y_test.mean()

    assert abs(train_rate - overall_rate) <= 0.30
    assert abs(val_rate - overall_rate) <= 0.30
    assert abs(test_rate - overall_rate) <= 0.30


def test_prepare_mlp_data_raises_for_non_numeric_features() -> None:
    df = make_valid_dataframe()
    df["plan"] = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]

    with pytest.raises(ValueError, match="Non-numeric columns found"):
        prepare_mlp_data(df)


def test_prepare_mlp_data_raises_when_target_is_missing() -> None:
    df = make_valid_dataframe().drop(columns=["target"])

    with pytest.raises(ValueError, match="Target column 'target' not found"):
        prepare_mlp_data(df)


def test_prepare_mlp_data_raises_for_invalid_target_stratification() -> None:
    df = pd.DataFrame(
        {
            "feat_1": [0, 1, 0, 1],
            "feat_2": [10, 20, 15, 30],
            "feat_3": [100, 120, 90, 130],
            "target": [0, 0, 0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="Stratified split requires at least 2 samples in each target class",
    ):
        prepare_mlp_data(df)


def test_save_and_load_preprocessing_pipeline_allows_transform(tmp_path) -> None:
    df = make_valid_dataframe()

    (
        X_train,
        _X_val,
        _X_test,
        _y_train,
        _y_val,
        _y_test,
        scaler,
        feature_names,
    ) = prepare_mlp_data(df, seed=42)

    output_path = tmp_path / "preprocessing_pipeline.joblib"

    saved_path = save_preprocessing_pipeline(
        scaler=scaler,
        feature_names=feature_names,
        output_path=output_path,
    )

    artifact = load_preprocessing_pipeline(saved_path)

    assert artifact["pipeline_type"] == "mlp_preprocessing"
    assert artifact["feature_names"] == feature_names
    assert isinstance(artifact["scaler"], StandardScaler)

    X_batch = df[feature_names].iloc[:2]
    X_batch_transformed = artifact["scaler"].transform(X_batch)

    assert X_batch_transformed.shape == (2, len(feature_names))
    assert X_train.shape[1] == len(feature_names)


def test_load_preprocessing_pipeline_raises_for_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(
        FileNotFoundError,
        match="Preprocessing pipeline artifact not found",
    ):
        load_preprocessing_pipeline(missing_path)


def test_load_preprocessing_pipeline_raises_for_invalid_object(tmp_path) -> None:
    import joblib

    invalid_path = tmp_path / "invalid.joblib"
    joblib.dump(["not", "a", "dict"], invalid_path)

    with pytest.raises(
        ValueError,
        match="Expected a dictionary",
    ):
        load_preprocessing_pipeline(invalid_path)


def test_load_preprocessing_pipeline_raises_for_missing_required_keys(tmp_path) -> None:
    import joblib

    invalid_path = tmp_path / "invalid_keys.joblib"
    joblib.dump({"pipeline_type": "mlp_preprocessing"}, invalid_path)

    with pytest.raises(
        ValueError,
        match="Missing keys",
    ):
        load_preprocessing_pipeline(invalid_path)