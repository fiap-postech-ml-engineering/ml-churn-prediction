import logging

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from ..schemas import ChurnPrediction, PredictRequest, PredictResponse

# ==================== CONFIGURAÇÃO DE LOGGING ====================
logger = logging.getLogger(__name__)

# ==================== IMPORT DO MODELO (global) ====================
# Esses imports virão de app.py comomodelo compartilhado
# Para evitar circular imports, passamos como dependência

router = APIRouter(prefix="/predict", tags=["predictions"])


def apply_feature_engineering(features_dict: dict) -> dict:
    """
    Calcula features derived a partir das features RAW.

    Esse processamento espelha exatamente o que foi feito durante o treinamento
    no notebook 02_baselines.

    Features DERIVED criadas:
        - total_services: Soma de serviços adicionais contratados
        - fiber_price_impact: Interação Fibra Óptica x Monthly Charges
        - Total Charges Log: Transformação log de Total Charges
        - avg_ticket: Receita média (Total Charges / Tenure Months)
        - is_new_customer: Flag para clientes novos (Tenure Months < 6)

    Args:
        features_dict: Dicionário com features RAW

    Returns:
        dict: Dicionário com features RAW + derived
    """

    features_eng = features_dict.copy()

    # 1. Ajuste de Tenure (evitar divisão por zero)
    tenure = max(features_eng.get("Tenure Months", 1), 1)

    # 2. Stickiness: Total de serviços adicionais ativos
    service_cols = [
        'Online Security_Yes',
        'Online Backup_Yes',
        'Device Protection_Yes',
        'Tech Support_Yes',
        'Streaming TV_Yes',
        'Streaming Movies_Yes',
    ]
    total_services = sum(features_eng.get(col, 0) for col in service_cols)
    features_eng["total_services"] = float(total_services)

    # 3. Interação Fibra x Preço (O "vilão" do Churn)
    fiber_optic = features_eng.get("Internet Service_Fiber optic", 0)
    monthly_charges = features_eng.get("Monthly Charges", 0)
    features_eng["fiber_price_impact"] = float(fiber_optic * monthly_charges)

    # 4. Transformação Log de Total Charges (normalização)
    total_charges = features_eng.get("Total Charges", 0)
    features_eng["Total Charges Log"] = float(np.log1p(total_charges))

    # 5. Ticket médio (financeiro por mês)
    features_eng["avg_ticket"] = float(total_charges / tenure)

    # 6. Transformação Log de avg_ticket
    features_eng["avg_ticket_log"] = float(np.log1p(features_eng["avg_ticket"]))

    # 7. Flag de cliente novo
    features_eng["is_new_customer"] = 1.0 if tenure < 6 else 0.0

    return features_eng


@router.post("", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    model: torch.nn.Module = None,
    scaler: object = None,
    feature_names: list = None,
    device: torch.device = None,
):
    """
    Realiza Predição de Churn para um Cliente.

    Recebe features RAW do cliente. Features derived (calculadas) são
    geradas automaticamente pela API através de feature engineering.

    Args:
        request: Objeto PredictRequest com dicionário de features RAW
        model: Modelo MLP treinado (injetado de app.py)
        scaler: Scaler para normalização (injetado de app.py)
        feature_names: Lista de nomes de features esperadas (injetado de app.py)
        device: Device para PyTorch (injetado de app.py)

    Returns:
        PredictResponse: Objeto com predição completa

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

        resposta = PredictResponse(
            sucesso=True, predicao=predicao, entrada_recebida=features_with_eng
        )

        return resposta

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ ERRO durante predição: {e}")
        raise HTTPException(
            status_code=500, detail=f"Erro ao processar predição: {str(e)}"
        )
