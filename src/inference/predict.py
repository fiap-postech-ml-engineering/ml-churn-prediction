"""
Módulo de Carregamento e Inferência do Modelo MLP.
"""

import json
import logging
from pathlib import Path

import joblib
import torch

from ..models.mlp_model import MLPNetworkChurn
from ..config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


class ModelArtifacts:
    """
    Contém todos os artefatos necessários para inferência.
    """
    
    def __init__(self, model=None, scaler=None, feature_names=None, 
                 model_metrics=None, device=None):
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
    MODEL_PATH = MODELS_DIR / "best_mlp_model.pth"
    SCALER_PATH = MODELS_DIR / "mlp_scaler.joblib"
    FEATURES_PATH = MODELS_DIR / "mlp_features.joblib"
    METRICS_PATH = MODELS_DIR / "mlp_metrics.json"
    
    model = None
    scaler = None
    feature_names = None
    model_metrics = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Carregar scaler
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"✓ Scaler carregado de: {SCALER_PATH}")
        else:
            raise FileNotFoundError(f"Scaler não encontrado em: {SCALER_PATH}")
        
        # Carregar nomes de features
        if FEATURES_PATH.exists():
            feature_names = joblib.load(FEATURES_PATH)
            logger.info(f"✓ Features carregadas: {len(feature_names)} features")
        else:
            raise FileNotFoundError(f"Features não encontradas em: {FEATURES_PATH}")
        
        # Carregar métricas do modelo
        if METRICS_PATH.exists():
            with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                model_metrics = json.load(f)
            logger.info(f"✓ Métricas do modelo carregadas. ROC-AUC: {model_metrics.get('ROC-AUC', 'N/A')}")
        else:
            logger.warning(f"Métricas não encontradas em: {METRICS_PATH}")
        
        # Carregar modelo PyTorch
        if MODEL_PATH.exists():
            model = MLPNetworkChurn(input_size=len(feature_names))
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()  # Modo de inferência
            logger.info(f"✓ Modelo MLP carregado de: {MODEL_PATH}")
            logger.info(f"  Device: {device}")
            logger.info(f"  Modo: Avaliação (inference mode)")
        else:
            raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
        
        logger.info("=" * 70)
        logger.info("✅ MODELO CARREGADO COM SUCESSO")
        logger.info("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"❌ ERRO: {e}")
        logger.error("   A API está rodando, mas sem modelo. Endpoints /health e / funcionarão.")
        logger.error("   Certifique-se de treinar o modelo antes de usar /predict")
        
    except Exception as e:
        logger.error(f"❌ ERRO INESPERADO ao carregar modelo: {e}")
        logger.error("   Verifique se todos os arquivos estão presentes e válidos.")
    
    return ModelArtifacts(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        model_metrics=model_metrics,
        device=device
    )
