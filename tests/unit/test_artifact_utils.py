"""Testes para o módulo artifact_utils.py"""

import tempfile
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.artifact_utils import (
    build_preprocessing_artifact,
    load_preprocessing_pipeline,
    save_preprocessing_pipeline,
    try_load_preprocessing_pipeline,
)
from src.data.preprocessing_config import TARGET_COLUMN, PreprocessingConfig


@pytest.fixture
def sample_scaler():
    """Cria um StandardScaler ajustado."""
    import numpy as np

    x_train = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        }
    )

    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler


@pytest.fixture
def sample_feature_names():
    """Cria uma lista de nomes de features."""
    return ["feature1", "feature2", "feature3"]


class TestBuildPreprocessingArtifact:
    """Testes para build_preprocessing_artifact."""

    def test_build_artifact_returns_dict(self, sample_scaler, sample_feature_names):
        """Verifica se a função retorna um dicionário."""
        artifact = build_preprocessing_artifact(
            scaler=sample_scaler, feature_names=sample_feature_names
        )

        assert isinstance(artifact, dict)

    def test_build_artifact_has_required_keys(
        self, sample_scaler, sample_feature_names
    ):
        """Verifica se o artefato possui as chaves necessárias."""
        artifact = build_preprocessing_artifact(
            scaler=sample_scaler, feature_names=sample_feature_names
        )

        required_keys = {
            "pipeline_type",
            "scaler",
            "feature_names",
            "target_col",
            "seed",
            "test_size",
            "val_size",
        }
        assert required_keys.issubset(artifact.keys())

    def test_build_artifact_pipeline_type_is_correct(
        self, sample_scaler, sample_feature_names
    ):
        """Verifica se o tipo do pipeline está correto."""
        artifact = build_preprocessing_artifact(
            scaler=sample_scaler, feature_names=sample_feature_names
        )

        assert artifact["pipeline_type"] == "mlp_preprocessing"

    def test_build_artifact_feature_names_match(
        self, sample_scaler, sample_feature_names
    ):
        """Verifica se os nomes das features correspondem."""
        artifact = build_preprocessing_artifact(
            scaler=sample_scaler, feature_names=sample_feature_names
        )

        assert artifact["feature_names"] == sample_feature_names

    def test_build_artifact_scaler_is_preserved(
        self, sample_scaler, sample_feature_names
    ):
        """Verifica se o scaler é preservado."""
        artifact = build_preprocessing_artifact(
            scaler=sample_scaler, feature_names=sample_feature_names
        )

        assert artifact["scaler"] is sample_scaler

    def test_build_artifact_raises_error_for_empty_features(self, sample_scaler):
        """Verifica se levanta erro para lista vazia de features."""
        with pytest.raises(ValueError, match="feature_names cannot be empty"):
            build_preprocessing_artifact(scaler=sample_scaler, feature_names=[])

    def test_build_artifact_with_custom_values(
        self, sample_scaler, sample_feature_names
    ):
        """Verifica se funciona com valores customizados."""
        custom_config = PreprocessingConfig(
            target_column="custom_target", random_seed=123, test_size=0.2, val_size=0.15
        )

        artifact = build_preprocessing_artifact(
            scaler=sample_scaler,
            feature_names=sample_feature_names,
            target_col="custom_target",
            seed=123,
            test_size=0.2,
            val_size=0.15,
            config=custom_config,
        )

        assert artifact["target_col"] == "custom_target"
        assert artifact["seed"] == 123


class TestSavePreprocessingPipeline:
    """Testes para save_preprocessing_pipeline."""

    def test_save_pipeline_creates_file(self, sample_scaler, sample_feature_names):
        """Verifica se a função cria um arquivo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pipeline.joblib"

            saved_path = save_preprocessing_pipeline(
                scaler=sample_scaler,
                feature_names=sample_feature_names,
                output_path=output_path,
            )

            assert output_path.exists()
            assert saved_path == output_path

    def test_save_pipeline_creates_directories(
        self, sample_scaler, sample_feature_names
    ):
        """Verifica se cria diretórios se necessário."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "pipeline.joblib"

            save_preprocessing_pipeline(
                scaler=sample_scaler,
                feature_names=sample_feature_names,
                output_path=output_path,
            )

            assert output_path.exists()
            assert output_path.parent.exists()

    def test_save_pipeline_can_be_loaded(self, sample_scaler, sample_feature_names):
        """Verifica se o arquivo salvo pode ser carregado."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pipeline.joblib"

            save_preprocessing_pipeline(
                scaler=sample_scaler,
                feature_names=sample_feature_names,
                output_path=output_path,
            )

            loaded_artifact = joblib.load(output_path)
            assert isinstance(loaded_artifact, dict)
            assert loaded_artifact["feature_names"] == sample_feature_names


class TestLoadPreprocessingPipeline:
    """Testes para load_preprocessing_pipeline."""

    def test_load_pipeline_returns_dict(self, sample_scaler, sample_feature_names):
        """Verifica se a função retorna um dicionário."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pipeline.joblib"

            save_preprocessing_pipeline(
                scaler=sample_scaler,
                feature_names=sample_feature_names,
                output_path=output_path,
            )

            loaded = load_preprocessing_pipeline(input_path=output_path)
            assert isinstance(loaded, dict)

    def test_load_pipeline_has_required_keys(self, sample_scaler, sample_feature_names):
        """Verifica se o artefato carregado possui as chaves necessárias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pipeline.joblib"

            save_preprocessing_pipeline(
                scaler=sample_scaler,
                feature_names=sample_feature_names,
                output_path=output_path,
            )

            loaded = load_preprocessing_pipeline(input_path=output_path)

            required_keys = {
                "pipeline_type",
                "scaler",
                "feature_names",
                "target_col",
                "seed",
                "test_size",
                "val_size",
            }
            assert required_keys.issubset(loaded.keys())

    def test_load_pipeline_raises_error_for_missing_file(self):
        """Verifica se levanta erro para arquivo inexistente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "nonexistent.joblib"

            with pytest.raises(FileNotFoundError):
                load_preprocessing_pipeline(input_path=missing_path)

    def test_load_pipeline_raises_error_for_invalid_dict(
        self, sample_scaler, sample_feature_names
    ):
        """Verifica se levanta erro para arquivo inválido."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "invalid.joblib"

            # Salva um objeto que não é um dicionário
            joblib.dump("not a dict", invalid_path)

            with pytest.raises(ValueError, match="invalid"):
                load_preprocessing_pipeline(input_path=invalid_path)

    def test_load_pipeline_raises_error_for_missing_keys(
        self, sample_scaler, sample_feature_names
    ):
        """Verifica se levanta erro quando faltam chaves necessárias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "invalid.joblib"

            # Salva um dicionário com chaves incompletas
            joblib.dump({"pipeline_type": "mlp_preprocessing"}, invalid_path)

            with pytest.raises(ValueError, match="Missing keys"):
                load_preprocessing_pipeline(input_path=invalid_path)


class TestTryLoadPreprocessingPipeline:
    """Testes para try_load_preprocessing_pipeline."""

    def test_try_load_returns_dict_when_exists(
        self, sample_scaler, sample_feature_names
    ):
        """Verifica se retorna dicionário quando arquivo existe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pipeline.joblib"

            save_preprocessing_pipeline(
                scaler=sample_scaler,
                feature_names=sample_feature_names,
                output_path=output_path,
            )

            loaded = try_load_preprocessing_pipeline(input_path=output_path)
            assert isinstance(loaded, dict)

    def test_try_load_returns_none_when_missing(self):
        """Verifica se retorna None quando arquivo não existe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "nonexistent.joblib"

            loaded = try_load_preprocessing_pipeline(input_path=missing_path)
            assert loaded is None

    def test_try_load_never_raises_error(self):
        """Verifica se never levanta erro (retorna None em caso de falha)."""
        # Mesmo com um caminho inválido, não deve levantar erro
        result = try_load_preprocessing_pipeline(
            input_path="/nonexistent/path/file.joblib"
        )
        assert result is None


class TestRoundTrip:
    """Testes de round-trip: save e load."""

    def test_save_load_roundtrip(self, sample_scaler, sample_feature_names):
        """Verifica se save/load preserva os dados."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pipeline.joblib"

            # Save
            save_preprocessing_pipeline(
                scaler=sample_scaler,
                feature_names=sample_feature_names,
                output_path=output_path,
            )

            # Load
            loaded = load_preprocessing_pipeline(input_path=output_path)

            # Verify
            assert loaded["feature_names"] == sample_feature_names
            assert loaded["pipeline_type"] == "mlp_preprocessing"
            assert isinstance(loaded["scaler"], StandardScaler)
