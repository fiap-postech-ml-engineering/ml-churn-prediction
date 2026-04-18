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

    Features RAW esperadas (25 no total):
        - Latitude, Longitude (localização)
        - Tenure Months, Monthly Charges, Total Charges (financeiro)
        - Gender_Male, Senior Citizen_Yes, Partner_Yes, Dependents_Yes (demográfico)
        - Phone Service_Yes, Multiple Lines_Yes, Multiple Lines_No phone service
        - Internet Service_Fiber optic, Internet Service_No
        - Online Security_Yes, Online Security_No internet service
        - Online Backup_Yes, Online Backup_No internet service
        - Device Protection_Yes, Device Protection_No internet service
        - Tech Support_Yes, Tech Support_No internet service
        - Streaming TV_Yes, Streaming TV_No internet service
        - Streaming Movies_Yes, Streaming Movies_No internet service
        - Contract_One year, Contract_Two year
        - Paperless Billing_Yes
        - Payment Method_Credit card (automatic), Payment Method_Electronic check, Payment Method_Mailed check
    """

    features: dict = Field(
        ...,
        description="Dicionário com features RAW do cliente (derivadas são calculadas automaticamente)",
        json_schema_extra={
            "example": {
                "Latitude": 47.0,
                "Longitude": -122.0,
                "Tenure Months": 12.0,
                "Monthly Charges": 65.0,
                "Total Charges": 780.0,
                "CLTV": 600.0,
                "Gender_Male": 1.0,
                "Senior Citizen_Yes": 0.0,
                "Partner_Yes": 1.0,
                "Dependents_Yes": 0.0,
                "Phone Service_Yes": 1.0,
                "Multiple Lines_No phone service": 0.0,
                "Multiple Lines_Yes": 1.0,
                "Internet Service_Fiber optic": 1.0,
                "Internet Service_No": 0.0,
                "Online Security_No internet service": 0.0,
                "Online Security_Yes": 1.0,
                "Online Backup_No internet service": 0.0,
                "Online Backup_Yes": 1.0,
                "Device Protection_No internet service": 0.0,
                "Device Protection_Yes": 0.0,
                "Tech Support_No internet service": 0.0,
                "Tech Support_Yes": 0.0,
                "Streaming TV_No internet service": 0.0,
                "Streaming TV_Yes": 0.0,
                "Streaming Movies_No internet service": 0.0,
                "Streaming Movies_Yes": 0.0,
                "Contract_One year": 0.0,
                "Contract_Two year": 1.0,
                "Paperless Billing_Yes": 1.0,
                "Payment Method_Credit card (automatic)": 1.0,
                "Payment Method_Electronic check": 0.0,
                "Payment Method_Mailed check": 0.0,
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
