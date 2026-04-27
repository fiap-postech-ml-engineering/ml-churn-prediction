"""
FastAPI Application para Predição de Churn - MLP PyTorch
"""

import logging

from fastapi import FastAPI

from src.api.middleware import register_observability_middleware
from src.api.routes.features import router as features_route
from src.api.routes.health import router as health_route
from src.api.routes.predict import router as predict_route
from src.config.logging_config import setup_logging
from src.config.settings import API_DESCRIPTION, API_TITLE, API_VERSION
from src.inference.predict import load_model_artifacts

# ==================== CONFIGURAÇÃO DE LOGGING ====================
setup_logging()
logger = logging.getLogger(__name__)

# ==================== INICIALIZAÇÃO DA APLICAÇÃO ====================
app = FastAPI(
    title=API_TITLE or "API de Predição de Churn",
    version=API_VERSION or "1.0.0",
    description=API_DESCRIPTION
    or "API para classificação binária de churn em clientes de telecom",
    docs_url="/docs",
    redoc_url="/redoc",
)

register_observability_middleware(app)

app.include_router(router=health_route)
app.include_router(router=predict_route)
app.include_router(router=features_route)

model_artifacts = load_model_artifacts()

model = model_artifacts.model
model_metrics = model_artifacts.model_metrics


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
            "POST /predict": "Fazer predição de churn",
        },
        "modelo_carregado": model is not None,
        "modelo_metricas": model_metrics if model_metrics else "Não disponível",
    }
