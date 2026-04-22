"""Application configs.."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv()

# Base File Paths
BASE_DIR = Path(__file__).parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MLFLOW_TRACKING_PATH = BASE_DIR / "mlruns"
MLFLOW_ARTIFACTS_PATH = BASE_DIR / "models" / "mlflow_artifacts"

# Criar diretórios se não existirem
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
MLFLOW_TRACKING_PATH.mkdir(exist_ok=True)
MLFLOW_ARTIFACTS_PATH.mkdir(exist_ok=True)

# MLFlow Configurações
EXPERIMENT_NAME = "ml-churn-prediction"
EXPERIMENT_TAGS = {
    "project": "ml-churn-prediction",
    "business_domain": "telecom",
    "problem_type": "binary_classification",
    "target": "churn",
    "primary_metric": "recall",
    "dataset_name": "telco_customer_churn",
}

# Configurações do modelo
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
VALUATION_SIZE = float(os.getenv("VALUATION_SIZE", "0.2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "200"))
TARGET_COLUMN = "Churn Value"
SELECTED_FEATURES = [
        "Dependents", "Tenure Months", "Phone Service",
        "Multiple Lines", "Internet Service", "Online Security",
        "Online Backup", "Device Protection", "Tech Support",
        "Streaming TV", "Streaming Movies", "Contract",
        "Paperless Billing", "Payment Method", "Monthly Charge",
        "Total Charges", "Churn Value"
]
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
