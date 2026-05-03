import logging

import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from src.api.schemas import (
    TABULAR_RAW_FEATURES_EXAMPLE_CHURNER,
    TABULAR_RAW_FEATURES_EXAMPLE_NON_CHURNER,
    ChurnPrediction,
    ChurnRequest,
    ChurnResponse,
)
from src.config.settings import APPROVAL_THRESHOLD
from src.data.preprocessing import select_tabular_raw_features
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


@router.post("/predict", response_model=ChurnResponse)
def predict(
    request: ChurnRequest = Body(
        ...,
        openapi_examples={
            "payload churner": {
                "summary": "Exemplo de Payload (Churner)",
                "value": {"features": TABULAR_RAW_FEATURES_EXAMPLE_CHURNER},
            },
            "payload não churner": {
                "summary": "Exemplo de Payload (Não Churner)",
                "value": {"features": TABULAR_RAW_FEATURES_EXAMPLE_NON_CHURNER},
            },
        },
    )
):
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
        except ValueError:
            raise
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

        df_model_ready = _run_step(
            "select_tabular_raw_features",
            select_tabular_raw_features,
            df,
            require_target=False,
        )

        inference_loader = _run_step(
            "prepare_inference_batch",
            prepare_inference_batch,
            df_features=df_model_ready,
            scaler=scaler,
            device=device,
            feature_names=feature_names,
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

    except ValueError as e:
        logger.warning(f"⚠️ Payload inválido para inferência: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"❌ ERRO durante predição: {e}")
        raise HTTPException(
            status_code=500, detail=f"Erro ao processar predição: {str(e)}"
        )
