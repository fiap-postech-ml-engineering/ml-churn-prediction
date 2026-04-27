# ==================== MODELOS PYDANTIC (Schemas de Requisição/Resposta) ====================


from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Modelo de Entrada para Requisição de Predição.

    Representa os dados RAW de um cliente de telecom. As features derived
    (calculadas) serão geradas automaticamente pela API.

    Atributos:
        features: Dicionário com features RAW do cliente (dados originais do dataset).
                  As features calculadas (avg_ticket, is_new_customer, etc)
                  são automaticamente derivadas pela API.

    Features RAW esperadas (16 no total):
        - Dependents, Phone Service, Multiple Lines, Internet Service
        - Online Security, Online Backup, Device Protection, Tech Support
        - Streaming TV, Streaming Movies, Contract, Paperless Billing
        - Payment Method, Tenure Months, Monthly Charges, Total Charges
    """

    features: dict = Field(
        ...,
        description="Dicionário com features RAW do cliente (derivadas são calculadas automaticamente)",
        json_schema_extra={
            "example": {
                "Dependents": "No",
                "Tenure Months": 12,
                "Phone Service": "Yes",
                "Multiple Lines": "Yes",
                "Internet Service": "Fiber optic",
                "Online Security": "No",
                "Online Backup": "Yes",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "No",
                "Streaming Movies": "No",
                "Contract": "Two year",
                "Paperless Billing": "Yes",
                "Payment Method": "Credit card (automatic)",
                "Monthly Charges": 65.0,
                "Total Charges": 780.0,
            }
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
