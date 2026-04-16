"""
FastAPI Application para Predição de Churn - MLP PyTorch
"""

import logging
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from .schemas import ChurnRequest, ChurnResponse, ChurnPrediction
from ..config.settings import API_TITLE, API_VERSION, API_DESCRIPTION
from ..features.feature_engineering import apply_feature_engineering
from ..inference.predict import load_model_artifacts

# ==================== CONFIGURAÇÃO DE LOGGING ====================
logger = logging.getLogger(__name__)

# ==================== INICIALIZAÇÃO DA APLICAÇÃO ====================
app = FastAPI(
    title=API_TITLE or "API de Predição de Churn",
    version=API_VERSION or "1.0.0",
    description=API_DESCRIPTION or "API para classificação binária de churn em clientes de telecom",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==================== CARREGAMENTO GLOBAL DO MODELO ====================
# Carrega artefatos do modelo na inicialização
model_artifacts = load_model_artifacts()

model = model_artifacts.model
scaler = model_artifacts.scaler
feature_names = model_artifacts.feature_names
model_metrics = model_artifacts.model_metrics
device = model_artifacts.device


# ==================== ENDPOINTS DA API ====================

@app.get("/")
def home():
    """
    Retorna Informações Gerais sobre a API.
    
    Returns:
        dict: Nome, versão, descrição e endpoints disponíveis
    """
    return {
        "nome": "API de Predição de Churn",
        "versao": "1.0.0",
        "descricao": "API para classificação binária de churn em clientes de telecom",
        "endpoints": {
            "GET /": "Informações da API",
            "GET /features": "Lista de todas as features esperadas",
            "GET /health": "Status de saúde da API",
            "GET /docs": "Documentação interativa (Swagger)",
            "GET /redoc": "Documentação ReDoc",
            "POST /predict": "Fazer predição de churn"
        },
        "modelo_carregado": model is not None,
        "modelo_metricas": model_metrics if model_metrics else "Não disponível"
    }


@app.get("/health")
def health():
    """
    Verifica a Saúde da API e Disponibilidade do Modelo.
    
    Retorna status de operacionalidade dos componentes críticos:
    - API status
    - Disponibilidade do modelo
    - Disponibilidade do scaler
    - Device (CPU ou GPU)
    
    Returns:
        dict: Status detalhado de todos os componentes
    """
    status = {
        "api_status": "operacional",
        "timestamp": str(np.datetime64('now')),
        "componentes": {
            "modelo_carregado": model is not None,
            "scaler_carregado": scaler is not None,
            "features_carregadas": feature_names is not None,
            "device": str(device),
        }
    }
    
    if model is None:
        status["aviso"] = "Modelo não foi carregado. Verifique os logs."
    
    return status


@app.get("/features")
def get_features():
    """
    Retorna Lista de Todas as Features Esperadas para Predição.
    
    Útil para o cliente saber exatamente quais features enviar no endpoint /predict.
    As features devem ser enviadas como um dicionário com essas chaves.
    
    Returns:
        dict: Informações sobre as features (nomes, ordem, quantidade)
    """
    
    if feature_names is None:
        raise HTTPException(
            status_code=503,
            detail="Features não estão disponíveis. Modelo não foi carregado corretamente."
        )
    
    return {
        "total_features": len(feature_names),
        "feature_names": feature_names,
        "descricao": "Use essas chaves exatamente como aparecem aqui no dicionário de features do endpoint /predict"
    }


@app.post("/predict", response_model=ChurnResponse)
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
            detail="Modelo não está disponível. Verifique os logs do servidor."
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
                detail=f"Features faltando na requisição: {list(missing_features)}"
            )
        
        if extra_features:
            logger.warning(f"⚠️ Features extras (serão ignoradas): {extra_features}")
        
        # Reordenar features na ordem correta (conforme o treinamento)
        features_ordered = np.array(
            [features_with_eng[fname] for fname in feature_names],
            dtype=np.float32
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
        logger.info(f"✓ Predição realizada: Classe={classe}, Prob={probabilidade_churn:.4f}")
        
        # Construir resposta
        predicao = ChurnPrediction(
            classe=classe,
            classe_descricao=classe_descricao,
            probabilidade_churn=probabilidade_churn
        )
        
        resposta = ChurnResponse(
            sucesso=True,
            predicao=predicao,
            entrada_recebida=features_with_eng  # Echo com features derivadas também
        )
        
        return resposta
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"❌ ERRO durante predição: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar predição: {str(e)}"
        )
