"""application configs.."""

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Base File Paths
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Criar diretórios se não existirem
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Configurações do modelo
RANDOM_SEED=42
APPROVAL_THRESHOLD = float(os.getenv("APPROVAL_THRESHOLD", "0.6"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "0.1.0")

# Configurações de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DRIFT_LOG_PATH = LOGS_DIR / "input_samples.jsonl"

# Configurações da API
API_TITLE = "Churn API"
API_VERSION = "0.1.0"
API_DESCRIPTION = "Churn Prediction API using FastAPI"

# Configurações de ambiente
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"