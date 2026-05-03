"""Testes para o módulo tabular_pipeline.py"""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config.settings import (
    TABULAR_DERIVED_FEATURES,
    TABULAR_RAW_CATEGORICAL_FEATURES,
    TABULAR_RAW_NUMERIC_FEATURES,
)
from src.data.preprocessing_config import TABULAR_PIPELINE_TYPE
from src.data.tabular_pipeline import (
    build_tabular_preprocessing_artifact,
    build_tabular_preprocessing_pipeline,
    fit_tabular_preprocessing_pipeline,
    transform_tabular_features,
)


@pytest.fixture
def sample_raw_df():
    """Cria um dataframe bruto com features numéricas e categóricas."""
    numeric_values = {col: [1.0, 2.0] for col in TABULAR_RAW_NUMERIC_FEATURES}
    categorical_values = {
        col: ["cat_a", "cat_b"] for col in TABULAR_RAW_CATEGORICAL_FEATURES
    }

    return pd.DataFrame({**numeric_values, **categorical_values})


class TestBuildTabularPreprocessingPipeline:
    """Testes para build_tabular_preprocessing_pipeline."""

    def test_build_pipeline_returns_pipeline(self, sample_raw_df):
        """Verifica se a função retorna um Pipeline."""
        pipeline = build_tabular_preprocessing_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_build_pipeline_has_required_steps(self):
        """Verifica se o pipeline tem os passos necessários."""
        pipeline = build_tabular_preprocessing_pipeline()

        assert "feature_engineering" in pipeline.named_steps
        assert "preprocessor" in pipeline.named_steps

    def test_build_pipeline_with_custom_numeric_features(self):
        """Verifica se funciona com features numéricas customizadas."""
        custom_numeric = ["feat1", "feat2"]
        pipeline = build_tabular_preprocessing_pipeline(numeric_features=custom_numeric)

        assert isinstance(pipeline, Pipeline)

    def test_build_pipeline_with_custom_categorical_features(self):
        """Verifica se funciona com features categóricas customizadas."""
        custom_categorical = ["cat1", "cat2"]
        pipeline = build_tabular_preprocessing_pipeline(
            categorical_features=custom_categorical
        )

        assert isinstance(pipeline, Pipeline)

    def test_build_pipeline_with_both_custom_features(self):
        """Verifica se funciona com ambas as features customizadas."""
        custom_numeric = ["feat1", "feat2"]
        custom_categorical = ["cat1", "cat2"]

        pipeline = build_tabular_preprocessing_pipeline(
            numeric_features=custom_numeric, categorical_features=custom_categorical
        )

        assert isinstance(pipeline, Pipeline)


class TestFitTabularPreprocessingPipeline:
    """Testes para fit_tabular_preprocessing_pipeline."""

    def test_fit_pipeline_returns_tuple(self, sample_raw_df):
        """Verifica se a função retorna uma tupla."""
        result = fit_tabular_preprocessing_pipeline(sample_raw_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fit_pipeline_returns_pipeline_and_features(self, sample_raw_df):
        """Verifica se retorna Pipeline e lista de features."""
        pipeline, feature_names = fit_tabular_preprocessing_pipeline(sample_raw_df)

        assert isinstance(pipeline, Pipeline)
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

    def test_fit_pipeline_features_are_strings(self, sample_raw_df):
        """Verifica se os nomes das features são strings."""
        pipeline, feature_names = fit_tabular_preprocessing_pipeline(sample_raw_df)

        assert all(isinstance(name, str) for name in feature_names)

    def test_fit_pipeline_with_custom_features(self, sample_raw_df):
        """Verifica se funciona com features customizadas."""
        custom_numeric = list(TABULAR_RAW_NUMERIC_FEATURES)[:2]
        custom_categorical = list(TABULAR_RAW_CATEGORICAL_FEATURES)[:2]

        pipeline, feature_names = fit_tabular_preprocessing_pipeline(
            sample_raw_df,
            numeric_features=custom_numeric,
            categorical_features=custom_categorical,
        )

        assert isinstance(pipeline, Pipeline)
        assert len(feature_names) > 0


class TestTransformTabularFeatures:
    """Testes para transform_tabular_features."""

    def test_transform_returns_numpy_array(self, sample_raw_df):
        """Verifica se a função retorna um numpy array."""
        pipeline, _ = fit_tabular_preprocessing_pipeline(sample_raw_df)
        result = transform_tabular_features(pipeline, sample_raw_df)

        assert hasattr(result, "shape")
        assert len(result.shape) == 2

    def test_transform_preserves_row_count(self, sample_raw_df):
        """Verifica se o número de linhas é preservado."""
        pipeline, _ = fit_tabular_preprocessing_pipeline(sample_raw_df)
        result = transform_tabular_features(pipeline, sample_raw_df)

        assert result.shape[0] == len(sample_raw_df)

    def test_transform_returns_numeric_data(self, sample_raw_df):
        """Verifica se os dados retornados são numéricos."""
        pipeline, _ = fit_tabular_preprocessing_pipeline(sample_raw_df)
        result = transform_tabular_features(pipeline, sample_raw_df)

        assert result.dtype in [float, "float64", "float32"]


class TestBuildTabularPreprocessingArtifact:
    """Testes para build_tabular_preprocessing_artifact."""

    def test_build_artifact_returns_dict(self, sample_raw_df):
        """Verifica se a função retorna um dicionário."""
        pipeline, feature_names = fit_tabular_preprocessing_pipeline(sample_raw_df)

        artifact = build_tabular_preprocessing_artifact(
            preprocessing_pipeline=pipeline, feature_names=feature_names
        )

        assert isinstance(artifact, dict)

    def test_build_artifact_has_required_keys(self, sample_raw_df):
        """Verifica se o artefato possui as chaves necessárias."""
        pipeline, feature_names = fit_tabular_preprocessing_pipeline(sample_raw_df)

        artifact = build_tabular_preprocessing_artifact(
            preprocessing_pipeline=pipeline, feature_names=feature_names
        )

        required_keys = {
            "pipeline_type",
            "preprocessing_pipeline",
            "feature_names",
            "numeric_features",
            "categorical_features",
            "target_col",
        }
        assert required_keys.issubset(artifact.keys())

    def test_build_artifact_pipeline_type_is_correct(self, sample_raw_df):
        """Verifica se o tipo do pipeline está correto."""
        pipeline, feature_names = fit_tabular_preprocessing_pipeline(sample_raw_df)

        artifact = build_tabular_preprocessing_artifact(
            preprocessing_pipeline=pipeline, feature_names=feature_names
        )

        assert artifact["pipeline_type"] == TABULAR_PIPELINE_TYPE

    def test_build_artifact_feature_names_match(self, sample_raw_df):
        """Verifica se os nomes das features correspondem."""
        pipeline, feature_names = fit_tabular_preprocessing_pipeline(sample_raw_df)

        artifact = build_tabular_preprocessing_artifact(
            preprocessing_pipeline=pipeline, feature_names=feature_names
        )

        assert artifact["feature_names"] == feature_names

    def test_build_artifact_raises_error_for_empty_features(self, sample_raw_df):
        """Verifica se levanta erro para lista vazia de features."""
        pipeline, _ = fit_tabular_preprocessing_pipeline(sample_raw_df)

        with pytest.raises(ValueError, match="feature_names cannot be empty"):
            build_tabular_preprocessing_artifact(
                preprocessing_pipeline=pipeline, feature_names=[]
            )

    def test_build_artifact_with_custom_features(self, sample_raw_df):
        """Verifica se funciona com features customizadas."""
        pipeline, feature_names = fit_tabular_preprocessing_pipeline(sample_raw_df)

        custom_numeric = list(TABULAR_RAW_NUMERIC_FEATURES)[:2]
        custom_categorical = list(TABULAR_RAW_CATEGORICAL_FEATURES)[:2]

        artifact = build_tabular_preprocessing_artifact(
            preprocessing_pipeline=pipeline,
            feature_names=feature_names,
            numeric_features=custom_numeric,
            categorical_features=custom_categorical,
        )

        assert artifact["numeric_features"] == custom_numeric
        assert artifact["categorical_features"] == custom_categorical
