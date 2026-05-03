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
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0001"))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "20"))
MLP_HIDDEN_DIMS = [256, 128, 64, 32]
MLP_DROPOUT_RATES = [0.3, 0.3, 0.2, 0.1]
PREPROCESSING_COLUMNS_TO_DROP = ["Churn Score", "Count"]
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "Churn Value")
PREPROCESSING_TARGET_COLUMN = os.getenv("PREPROCESSING_TARGET_COLUMN", "target")
TOTAL_CHARGES_COLUMN = os.getenv("TOTAL_CHARGES_COLUMN", "Total Charges")
PREPROCESSING_PIPELINE_PATH = (
    MODELS_DIR / "preprocessing" / "churn_preprocessing_pipeline_v1.joblib"
)
TABULAR_PREPROCESSING_PIPELINE_PATH = (
    MODELS_DIR / "preprocessing" / "churn_tabular_preprocessing_pipeline_v1.joblib"
)
TABULAR_MODEL_FEATURE_NAMES_PATH = (
    MODELS_DIR / "mlp" / "churn_mlp_input_features_v1.joblib"
)
TABULAR_MLP_MODEL_PATH = MODELS_DIR / "mlp" / "churn_mlp_best_state_dict_v1.pth"
TABULAR_MLP_METRICS_PATH = MODELS_DIR / "mlp" / "churn_mlp_metrics_v1.json"
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
    "Monthly Charges",
    "Total Charges",
    "Churn Value",
]
# Schema de features RAW (etapas 2 e 3 do escopo de refatoracao)
RAW_INT_FEATURES = [
    "Tenure Months",
    "Churn Value",
]
RAW_FLOAT_FEATURES = [
    "Monthly Charges",
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

TABULAR_RAW_NUMERIC_FEATURES = [
    "Latitude",
    "Longitude",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
]
TABULAR_RAW_CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
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
TABULAR_RAW_FEATURES = TABULAR_RAW_NUMERIC_FEATURES + TABULAR_RAW_CATEGORICAL_FEATURES
TABULAR_DERIVED_FEATURES = [
    "total_services",
    "fiber_price_impact",
    "avg_ticket",
    "Total Charges Log",
    "avg_ticket_log",
    "is_new_customer",
]
FEATURE_TYPE_STRICT = os.getenv("FEATURE_TYPE_STRICT", "true").lower() == "true"
APPROVAL_THRESHOLD = float(os.getenv("APPROVAL_THRESHOLD", "0.45"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "0.1.0")

# Configurações de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
SERVICE_NAME = os.getenv("SERVICE_NAME", "ml-churn-api")
REQUEST_ID_HEADER = os.getenv("REQUEST_ID_HEADER", "X-Request-ID")
LATENCY_WARN_MS = float(os.getenv("LATENCY_WARN_MS", "500"))
DRIFT_LOG_PATH = LOGS_DIR / "input_samples.jsonl"

# Configurações da API
API_TITLE = "Churn API"
API_VERSION = "0.1.0"
API_DESCRIPTION = "Churn Prediction API using FastAPI"

# Configurações de ambiente
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"
