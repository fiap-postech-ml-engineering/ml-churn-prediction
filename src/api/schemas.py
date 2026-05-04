# ==================== MODELOS PYDANTIC (Schemas de Requisição/Resposta) ====================


from pydantic import BaseModel, Field

from src.config.settings import TABULAR_RAW_FEATURES

TABULAR_RAW_FEATURES_EXAMPLE_CHURNER = {
    "Latitude": 34.425581,
    "Longitude": -119.813765,
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "Yes",
    "Online Backup": "No",
    "Device Protection": "Yes",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Credit card (automatic)",
    "Monthly Charges": 79.65,
    "Total Charges": 95.4,
    "CLTV": 5432,
}

TABULAR_RAW_FEATURES_EXAMPLE_NON_CHURNER = {
    "Latitude": 34.027337,
    "Longitude": -118.285150,
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "Yes",
    "Tenure Months": 0,
    "Phone Service": "Yes",
    "Multiple Lines": "Yes",
    "Internet Service": "Fiber optic",
    "Online Security": "Yes",
    "Online Backup": "Yes",
    "Device Protection": "Yes",
    "Tech Support": "Yes",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "One year",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 105.5,
    "Total Charges": 2686.05,
    "CLTV": 5822,
}

FEATURES_RESPONSE_EXAMPLE = {
    "total_features": len(TABULAR_RAW_FEATURES),
    "feature_names": TABULAR_RAW_FEATURES,
    "descricao": "Use essas chaves exatamente como aparecem no dicionário raw enviado ao endpoint /predict",
}


class PredictRequest(BaseModel):
    """
    Modelo de Entrada para Requisição de Predição.

    Representa os dados RAW de um cliente de telecom. A API recebe os nomes
    originais das colunas do dataset, não features já transformadas em one-hot.

    As features derivadas, como avg_ticket, is_new_customer, total_services e
    fiber_price_impact, são geradas automaticamente pela API.

    Não enviar:
        - Churn Value
        - Churn Label
        - colunas one-hot como Contract_Two year ou Payment Method_Electronic check
    """

    features: dict = Field(
        ...,
        description="Dicionário com features RAW do cliente (derivadas são calculadas automaticamente)",
        json_schema_extra={
            "example": TABULAR_RAW_FEATURES_EXAMPLE_CHURNER,
            "examples": [TABULAR_RAW_FEATURES_EXAMPLE_CHURNER],
        },
    )


class ChurnPrediction(BaseModel):
    """
    Resultado da Classificação de Churn.

    Atributos:
        classe: Classe predita (0 = Sem Churn, 1 = Churn)
        classe_descricao: Descrição legível da predição
        probabilidade_churn: Confiança da predição (0 a 1)
                            - Valores > 0.5 indicam tendência a churn
    """

    classe: int = Field(..., description="Classe predita (0=Sem Churn, 1=Churn)")
    classe_descricao: str = Field(..., description="Descrição da classe")
    probabilidade_churn: float = Field(
        ..., description="Probabilidade de Churn (0 a 1)"
    )


class PredictResponse(BaseModel):
    """
    Resposta Completa da API para Predições.

    Atributos:
        sucesso: Indicador de sucesso da predição
        predicao: Objeto com classe e probabilidades
        entrada_recebida: Dicionário das features enviadas (echo dos dados)
    """

    sucesso: bool = Field(..., description="Indicador de sucesso da operação")
    predicao: ChurnPrediction = Field(..., description="Resultado da predição")
    entrada_recebida: dict = Field(..., description="Echo das features recebidas")


# ==================== ALIASES PARA COMPATIBILIDADE ====================

# Aliases para manter compatibilidade com código legado
ChurnRequest = PredictRequest
ChurnResponse = PredictResponse
