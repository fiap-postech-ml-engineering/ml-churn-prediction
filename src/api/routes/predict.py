import logging

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from src.api.schemas import ChurnPrediction, ChurnRequest, ChurnResponse
from src.features.feature_engineering import apply_feature_engineering
from src.inference.predict import load_model_artifacts

logger = logging.getLogger(__name__)

router = APIRouter()

model_artifacts = load_model_artifacts()

model = model_artifacts.model
scaler = model_artifacts.scaler
feature_names = model_artifacts.feature_names
model_metrics = model_artifacts.model_metrics
device = model_artifacts.device


@router.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    """
    Realiza Predição de Churn para um Cliente.

    Recebe features RAW do cliente. Features derived (calculadas) são
    geradas automaticamente pela API através de feature engineering.

    Args:
        request: Objeto ChurnRequest com dicionário de features RAW

    Returns:
        ChurnResponse: Objeto com predição completa

    Raises:
        HTTPException: Se modelo não está carregado ou features inválidas
    """

    # Validar se modelo foi carregado
    if model is None:
        logger.error("❌ ERRO: Modelo não foi carregado")
        raise HTTPException(
            status_code=503,
            detail="Modelo não está disponível. Verifique os logs do servidor.",
        )

    try:
        # Aplicar Feature Engineering (calcula features derived)
        features_with_eng = apply_feature_engineering(request.features)

        # Validar se todas as features esperadas estão presentes (RAW + DERIVED)
        missing_features = set(feature_names) - set(features_with_eng.keys())
        extra_features = set(features_with_eng.keys()) - set(feature_names)

        if missing_features:
            logger.error(f"❌ Features faltando: {missing_features}")
            raise HTTPException(
                status_code=400,
                detail=f"Features faltando na requisição: {list(missing_features)}",
            )

        if extra_features:
            logger.warning(f"⚠️ Features extras (serão ignoradas): {extra_features}")

        # Reordenar features na ordem correta (conforme o treinamento)
        features_ordered = np.array(
            [features_with_eng[fname] for fname in feature_names], dtype=np.float32
        ).reshape(1, -1)

        # Normalizar usando o scaler treinado
        features_scaled = scaler.transform(features_ordered)

        # Converter para tensor PyTorch
        features_tensor = torch.from_numpy(features_scaled).float().to(device)

        # Realizar predição (disable gradientes para inference)
        with torch.no_grad():
            logits = model(features_tensor)
            probabilidade_churn = float(torch.sigmoid(logits).cpu().numpy()[0, 0])

        # Determinar classe (threshold = 0.5)
        classe = 1 if probabilidade_churn >= 0.5 else 0
        classe_descricao = "Churn Detectado" if classe == 1 else "Sem Churn"

        # Log da predição
        logger.info(
            f"✓ Predição realizada: Classe={classe}, Prob={probabilidade_churn:.4f}"
        )

        # Construir resposta
        predicao = ChurnPrediction(
            classe=classe,
            classe_descricao=classe_descricao,
            probabilidade_churn=probabilidade_churn,
        )

        resposta = ChurnResponse(
            sucesso=True,
            predicao=predicao,
            entrada_recebida=features_with_eng,  # Echo com features derivadas também
        )

        return resposta

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ ERRO durante predição: {e}")
        raise HTTPException(
            status_code=500, detail=f"Erro ao processar predição: {str(e)}"
        )
