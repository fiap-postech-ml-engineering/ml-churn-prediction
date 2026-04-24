import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import ChurnPrediction, ChurnRequest, ChurnResponse

from src.config.settings import APPROVAL_THRESHOLD

from src.features.select_features import select_model_features
from src.features.type_features import cast_feature_types
from src.features.missing_values import clean_missing_values
from src.features.apply_one_hot_encoding import apply_one_hot_encoding
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
        
        df = pd.DataFrame([request.features])

        df_selected_features = select_model_features(df)
        df_typed_features = cast_feature_types(df_selected_features)
        df_no_missing_values = clean_missing_values(df_typed_features)
        df_ohe = apply_one_hot_encoding(df_no_missing_values)
        df_feat_eng = apply_feature_engineering(df_ohe)

        # df_dataloaders = xxxxxxxxx(df_feat_eng)
        # probabilidade_churn = xxxxxxxxx(df_dataloaders) -> RESPOSTA DO MODELO 
        #
        # classe = 1 if probabilidade_churn >= APPROVAL_THRESHOLD else 0
        # classe_descricao = "Churn" if classe == 1 else "Não Churn"

        # Construir resposta
        predicao = ChurnPrediction(
            classe=classe, # Churn ou não
            classe_descricao=classe_descricao,
            probabilidade_churn=probabilidade_churn,
        )

        resposta = ChurnResponse(
            sucesso=True,
            predicao=predicao,
            entrada_recebida=df,  # Echo com features derivadas também
        )

        return resposta

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ ERRO durante predição: {e}")
        raise HTTPException(
            status_code=500, detail=f"Erro ao processar predição: {str(e)}"
        )
