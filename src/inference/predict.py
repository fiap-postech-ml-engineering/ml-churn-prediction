"""
Módulo de Carregamento e Inferência do Modelo MLP.
"""

import json
import logging

import joblib
import torch

from src.config.settings import MODELS_DIR, TABULAR_MLP_MODEL_PATH
from src.data.preprocessing import (
    try_load_preprocessing_pipeline,
    try_load_tabular_preprocessing_pipeline,
)
from src.models.mlp_model import MLPNetworkChurn

logger = logging.getLogger(__name__)


class ModelArtifacts:
    """
    Contém todos os artefatos necessários para inferência.
    """

    def __init__(
        self,
        model=None,
        scaler=None,
        feature_names=None,
        model_metrics=None,
        device=None,
    ):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.model_metrics = model_metrics
        self.device = device


def load_model_artifacts() -> ModelArtifacts:
    """
    Carrega modelo MLP, scaler e artefatos associados.

    Returns:
        ModelArtifacts: Objeto contendo modelo, scaler, features e device
    """

    logger.info("Iniciando carregamento do modelo MLP...")

    # Definir caminhos
    model_path = TABULAR_MLP_MODEL_PATH
    scaler_path = MODELS_DIR / "mlp/churn_mlp_input_scaler_v1.joblib"
    features_path = MODELS_DIR / "mlp/churn_mlp_input_features_v1.joblib"
    metrics_path = MODELS_DIR / "mlp/churn_mlp_metrics_v1.json"

    model = None
    scaler = None
    feature_names = None
    model_metrics = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        preprocessing_artifact = try_load_tabular_preprocessing_pipeline()
        if preprocessing_artifact is not None:
            scaler = preprocessing_artifact["preprocessing_pipeline"]
            feature_names = preprocessing_artifact["feature_names"]
            logger.info("✓ Tabular preprocessing pipeline carregado do bundle de treino")
            logger.info(f"✓ Features carregadas: {len(feature_names)} features")
        else:
            preprocessing_artifact = try_load_preprocessing_pipeline()
            if preprocessing_artifact is not None:
                scaler = preprocessing_artifact["scaler"]
                feature_names = preprocessing_artifact["feature_names"]
                logger.info("✓ Preprocessing legado carregado do bundle anterior")
                logger.info(f"✓ Features carregadas: {len(feature_names)} features")
            else:
                # Fallback para artefatos legados separados.
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
                    logger.info(f"✓ Scaler carregado de: {scaler_path}")
                else:
                    raise FileNotFoundError(f"Scaler não encontrado em: {scaler_path}")

                if features_path.exists():
                    feature_names = joblib.load(features_path)
                    logger.info(f"✓ Features carregadas: {len(feature_names)} features")
                else:
                    raise FileNotFoundError(f"Features não encontradas em: {features_path}")

        # Carregar métricas do modelo
        if metrics_path.exists():
            with open(metrics_path, 'r', encoding='utf-8') as f:
                model_metrics = json.load(f)
            logger.info(
                f"✓ Métricas do modelo carregadas. ROC-AUC: {model_metrics.get('ROC-AUC', 'N/A')}"
            )
        else:
            logger.warning(f"Métricas não encontradas em: {metrics_path}")

        # Carregar modelo PyTorch
        if model_path.exists():
            model = MLPNetworkChurn(input_size=len(feature_names))
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()  # Modo de inferência
            logger.info(f"✓ Modelo MLP carregado de: {model_path}")
            logger.info(f"  Device: {device}")
            logger.info("  Modo: Avaliação (inference mode)")
        else:
            legacy_model_path = MODELS_DIR / "mlp/churn_mlp_best_state_dict_v1.pth"
            if legacy_model_path.exists():
                model = MLPNetworkChurn(input_size=len(feature_names))
                model.load_state_dict(torch.load(legacy_model_path, map_location=device))
                model.to(device)
                model.eval()
                logger.info(f"✓ Modelo MLP legado carregado de: {legacy_model_path}")
                logger.info(f"  Device: {device}")
                logger.info("  Modo: Avaliação (inference mode)")
            else:
                raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

        logger.info("=" * 70)
        logger.info("✅ MODELO CARREGADO COM SUCESSO")
        logger.info("=" * 70)

    except FileNotFoundError as e:
        logger.error(f"❌ ERRO: {e}")
        logger.error(
            "   A API está rodando, mas sem modelo. Endpoints /health e / funcionarão."
        )
        logger.error("   Certifique-se de treinar o modelo antes de usar /predict")

    except Exception as e:
        logger.error(f"❌ ERRO INESPERADO ao carregar modelo: {e}")
        logger.error("   Verifique se todos os arquivos estão presentes e válidos.")

    return ModelArtifacts(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        model_metrics=model_metrics,
        device=device,
    )

