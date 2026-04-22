from fastapi import APIRouter, HTTPException

from src.config.settings import SELECTED_FEATURES

router = APIRouter()


@router.get("/features")
def get_features():
    """
    Retorna Lista de Todas as Features Esperadas para Predição.

    Útil para o cliente saber exatamente quais features enviar no endpoint /predict.
    As features devem ser enviadas como um dicionário com essas chaves.

    Returns:
        dict: Informações sobre as features (nomes, ordem, quantidade)
    """

    if SELECTED_FEATURES is None:
        raise HTTPException(
            status_code=503,
            detail="Features não estão disponíveis. Modelo não foi carregado corretamente.",
        )

    return {
        "total_features": len(SELECTED_FEATURES),
        "feature_names": SELECTED_FEATURES,
        "descricao": "Use essas chaves exatamente como aparecem aqui no dicionário de features do endpoint /predict",
    }
