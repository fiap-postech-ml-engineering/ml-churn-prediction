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
VALIDATION_SIZE = float(
    os.getenv("VALIDATION_SIZE", os.getenv("VALUATION_SIZE", "0.2"))
)
# Backward compatibility for legacy misspelled setting.
VALUATION_SIZE = VALIDATION_SIZE
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "200"))
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "Churn Value")
PREPROCESSING_TARGET_COLUMN = os.getenv("PREPROCESSING_TARGET_COLUMN", "target")
TOTAL_CHARGES_COLUMN = os.getenv("TOTAL_CHARGES_COLUMN", "Total Charges")
PREPROCESSING_COLUMNS_TO_DROP = tuple(
    col.strip()
    for col in os.getenv("PREPROCESSING_COLUMNS_TO_DROP", "Churn Score,Count").split(",")
    if col.strip()
)
PREPROCESSING_PIPELINE_PATH = (
    MODELS_DIR / "preprocessing" / "churn_preprocessing_pipeline_v1.joblib"
)
SELECTED_FEATURES = [
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charge",
    "Total Charges",
    "Churn Value",
]
# Schema de features RAW (etapas 2 e 3 do escopo de refatoracao)
RAW_INT_FEATURES = [
    "Tenure Months",
    "Churn Value",
]
RAW_FLOAT_FEATURES = [
    "Monthly Charge",
    "Total Charges",
]
RAW_CATEGORICAL_FEATURES = [
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]
FEATURE_TYPE_STRICT = os.getenv("FEATURE_TYPE_STRICT", "true").lower() == "true"
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
