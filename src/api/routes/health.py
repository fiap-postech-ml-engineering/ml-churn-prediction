import logging

import numpy as np
from fastapi import APIRouter

from src.inference.predict import load_model_artifacts

# ==================== CONFIGURAÇÃO DE LOGGING ====================
logger = logging.getLogger(__name__)

# ==================== IMPORT DO MODELO (global) ====================
# Esses imports virão de app.py como modelo compartilhado

model_artifacts = load_model_artifacts()
model = model_artifacts.model
scaler = model_artifacts.scaler
feature_names = model_artifacts.feature_names
device = model_artifacts.device

router = APIRouter()


@router.get("/health")
def health_check(model=model, scaler=scaler, feature_names=feature_names, device=device):
    """
    Verifica a Saúde da API e Disponibilidade do Modelo.

    Retorna status de operacionalidade dos componentes críticos:
    - API status
    - Disponibilidade do modelo
    - Disponibilidade do scaler
    - Device (CPU ou GPU)

    Args:
        model: Modelo MLP treinado (injetado de app.py)
        scaler: Scaler para normalização (injetado de app.py)
        feature_names: Lista de nomes de features (injetado de app.py)
        device: Device para PyTorch (injetado de app.py)

    Returns:
        dict: Status detalhado de todos os componentes
    """

    status = {
        "api_status": "operacional",
        "timestamp": str(np.datetime64('now')),
        "componentes": {
            "modelo_carregado": model is not None,
            "scaler_carregado": scaler is not None,
            "features_carregadas": feature_names is not None,
            "device": str(device),
        },
    }

    if model is None:
        status["aviso"] = "Modelo não foi carregado. Verifique os logs."

    return status
