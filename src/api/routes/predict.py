import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import ChurnPrediction, ChurnRequest, ChurnResponse
from src.config.settings import (
    APPROVAL_THRESHOLD,
    RAW_CATEGORICAL_FEATURES,
    RAW_FLOAT_FEATURES,
    RAW_INT_FEATURES,
    SELECTED_FEATURES,
    TARGET_COLUMN,
)
from src.features.apply_one_hot_encoding import apply_one_hot_encoding
from src.features.feature_engineering import apply_feature_engineering
from src.features.missing_values import clean_missing_values
from src.features.select_features import select_model_features
from src.features.type_features import cast_feature_types
from src.inference.feature_contract import (
    align_to_model_feature_contract,
    build_raw_inference_feature_names,
    ensure_feature_engineering_columns,
    ensure_model_one_hot_columns,
)
from src.inference.predict import load_model_artifacts
from src.inference.prepare_inference_data import prepare_inference_batch, run_inference

logger = logging.getLogger(__name__)

router = APIRouter()

model_artifacts = load_model_artifacts()

model = model_artifacts.model
scaler = model_artifacts.scaler
feature_names = model_artifacts.feature_names
model_metrics = model_artifacts.model_metrics
device = model_artifacts.device

RAW_INFERENCE_FEATURES = build_raw_inference_feature_names(
    selected_features=SELECTED_FEATURES,
    target_column=TARGET_COLUMN,
)
RAW_INFERENCE_INT_FEATURES = [
    feature for feature in RAW_INT_FEATURES if feature != TARGET_COLUMN
]


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

    def _run_step(step_name: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.exception(f"❌ Falha no passo '{step_name}'")
            raise RuntimeError(f"Erro em '{step_name}': {exc}") from exc

    # Validar se modelo foi carregado
    if model is None:
        logger.error("❌ ERRO: Modelo não foi carregado")
        raise HTTPException(
            status_code=503,
            detail="Modelo não está disponível. Verifique os logs do servidor.",
        )

    try:
        df = pd.DataFrame([request.features])

        df_selected_features = _run_step(
            "select_model_features",
            select_model_features,
            df,
            selected_features=RAW_INFERENCE_FEATURES,
        )
        df_typed_features = _run_step(
            "cast_feature_types",
            cast_feature_types,
            df_selected_features,
            int_features=RAW_INFERENCE_INT_FEATURES,
            float_features=RAW_FLOAT_FEATURES,
            categorical_features=RAW_CATEGORICAL_FEATURES,
        )
        df_no_missing_values = _run_step(
            "clean_missing_values",
            clean_missing_values,
            df_typed_features,
            selected_features=RAW_INFERENCE_FEATURES,
            int_features=RAW_INFERENCE_INT_FEATURES,
            float_features=RAW_FLOAT_FEATURES,
            categorical_features=RAW_CATEGORICAL_FEATURES,
        )
        df_ohe = apply_one_hot_encoding(df_no_missing_values)

        df_ohe_with_model_columns = ensure_model_one_hot_columns(
            df_encoded=df_ohe,
            df_raw_features=df_no_missing_values,
            model_feature_names=feature_names,
            categorical_features=RAW_CATEGORICAL_FEATURES,
        )

        df_ohe_ready_for_feat_eng = ensure_feature_engineering_columns(
            df_encoded=df_ohe_with_model_columns,
            df_raw_features=df_no_missing_values,
        )

        df_feat_eng = apply_feature_engineering(df_ohe_ready_for_feat_eng)

        df_model_ready = align_to_model_feature_contract(
            df_feat_eng,
            model_feature_names=feature_names,
        )

        inference_loader = _run_step(
            "prepare_inference_batch",
            prepare_inference_batch,
            df_features=df_model_ready,
            scaler=scaler,
            device=device,
        )

        probabilidades_churn, classes = _run_step(
            "run_inference",
            run_inference,
            inference_loader=inference_loader,
            model=model,
            device=device,
            approval_threshold=APPROVAL_THRESHOLD,
        )

        probabilidade_churn = float(probabilidades_churn[0])
        classe = int(classes[0])
        classe_descricao = "Churn" if classe == 1 else "Não Churn"
        predicao = ChurnPrediction(
            classe=classe,
            classe_descricao=classe_descricao,
            probabilidade_churn=probabilidade_churn,
        )

        resposta = ChurnResponse(
            sucesso=True,
            predicao=predicao,
            entrada_recebida=request.features,
        )

        return resposta

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ ERRO durante predição: {e}")
        raise HTTPException(
            status_code=500, detail=f"Erro ao processar predição: {str(e)}"
        )
