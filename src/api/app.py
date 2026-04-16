"""
API FastAPI para Predição de Churn - MLP PyTorch

Descripción:
    API REST que utiliza um modelo MLP treinado em PyTorch para prever 
    se um cliente de telecom fará churn (sairá da empresa).
    
Componentes:
    - PredictRequest: Modelo de entrada para requisições de predição
    - PredictResponse: Resposta completa da API
    - Carregamento de modelo com tratamento de erros
    - 3 endpoints: home, health, predict
"""

import json
import logging
from pathlib import Path
from typing import List

import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException

from .schemas import PredictRequest, PredictResponse
from .routes import predict as predict_router
from .routes import health as health_router

# ==================== CONFIGURAÇÃO DE LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== INICIALIZAÇÃO DA APLICAÇÃO ====================
app = FastAPI(
    title="API de Predição de Churn",
    version="1.0.0",
    description="API para classificação binária de churn em clientes de telecom",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==================== DEFINIÇÃO DE CAMINHOS ====================
PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/api -> src -> ml-churn-prediction
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "best_mlp_model.pth"
SCALER_PATH = MODELS_DIR / "mlp_scaler.joblib"
FEATURES_PATH = MODELS_DIR / "mlp_features.joblib"
METRICS_PATH = MODELS_DIR / "mlp_metrics.json"

# ==================== DEFINIÇÃO DA ARQUITETURA MLP ====================
class MLPNetworkChurn(nn.Module):
    """
    Rede Neural MLP para Classificação de Churn.
    
    Arquitetura:
        Input (35) → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
                  → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
                  → Dense(64)  → BatchNorm → ReLU → Dropout(0.2)
                  → Dense(32)  → BatchNorm → ReLU → Dropout(0.1)
                  → Output(1) [logit]
    """
    
    def __init__(self, input_size=35, hidden_dims=[256, 128, 64, 32],
                 dropout_rates=[0.3, 0.3, 0.2, 0.1]):
        super(MLPNetworkChurn, self).__init__()
        
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        
        layers = []
        prev_dim = input_size
        
        # Construir camadas ocultas
        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim
        
        # Camada de output
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ==================== CARREGAMENTO DO MODELO ====================
logger.info("Iniciando carregamento do modelo MLP...")

model = None
scaler = None
feature_names = None
model_metrics = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Carregar scaler
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"✓ Scaler carregado de: {SCALER_PATH}")
    else:
        raise FileNotFoundError(f"Scaler não encontrado em: {SCALER_PATH}")
    
    # Carregar nomes de features
    if FEATURES_PATH.exists():
        feature_names = joblib.load(FEATURES_PATH)
        logger.info(f"✓ Features carregadas: {len(feature_names)} features")
    else:
        raise FileNotFoundError(f"Features não encontradas em: {FEATURES_PATH}")
    
    # Carregar métricas do modelo
    if METRICS_PATH.exists():
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            model_metrics = json.load(f)
        logger.info(f"✓ Métricas do modelo carregadas. ROC-AUC: {model_metrics.get('ROC-AUC', 'N/A')}")
    else:
        logger.warning(f"Métricas não encontradas em: {METRICS_PATH}")
    
    # Carregar modelo PyTorch
    if MODEL_PATH.exists():
        model = MLPNetworkChurn(input_size=len(feature_names))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()  # Modo de inferência
        logger.info(f"✓ Modelo MLP carregado de: {MODEL_PATH}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Modo: Avaliação (inference mode)")
    else:
        raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
    
    logger.info("=" * 70)
    logger.info("✅ MODELO CARREGADO COM SUCESSO")
    logger.info("=" * 70)
    
except FileNotFoundError as e:
    logger.error(f"❌ ERRO: {e}")
    logger.error("   A API está rodando, mas sem modelo. Endpoints /health e / funcionarão.")
    logger.error("   Certifique-se de treinar o modelo antes de usar /predict")
    
except Exception as e:
    logger.error(f"❌ ERRO INESPERADO ao carregar modelo: {e}")
    logger.error("   Verifique se todos os arquivos estão presentes e válidos.")


# ==================== INCLUSÃO DOS ROUTERS ====================
app.include_router(predict_router.router)
app.include_router(health_router.router)


# ==================== ENDPOINTS DA API ====================

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


# ==================== INICIALIZAÇÃO DO SERVIDOR ====================
if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 70)
    logger.info("Iniciando servidor FastAPI...")
    logger.info("=" * 70)
    logger.info("Documentação disponível em: http://localhost:8000/docs")
    logger.info("Redoc disponível em: http://localhost:8000/redoc")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
