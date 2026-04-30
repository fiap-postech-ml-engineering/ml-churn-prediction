from fastapi import APIRouter, HTTPException

from src.api.schemas import FeaturesResponse
from src.config.settings import TABULAR_RAW_FEATURES

router = APIRouter()


@router.get("/features", response_model=FeaturesResponse)
def get_features():
    """
    Retorna Lista de Todas as Features Esperadas para Predição.

    Útil para o cliente saber exatamente quais features enviar no endpoint /predict.
    As features devem ser enviadas como um dicionário com essas chaves.

    Returns:
        dict: Informações sobre as features (nomes, ordem, quantidade)
    """

    if TABULAR_RAW_FEATURES is None:
        raise HTTPException(
            status_code=503,
            detail="Features não estão disponíveis. Modelo não foi carregado corretamente.",
        )

    return FeaturesResponse(
        total_features=len(TABULAR_RAW_FEATURES),
        feature_names=list(TABULAR_RAW_FEATURES),
        descricao="Use essas chaves exatamente como aparecem no dicionário raw enviado ao endpoint /predict",
    )
