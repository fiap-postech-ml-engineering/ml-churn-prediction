import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing_config import ProcessedArray


def fit_scaler(x_train: pd.DataFrame) -> StandardScaler:
    """
    Ajusta o scaler somente nos dados de treino.
    """
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler


def transform_features(
    scaler: StandardScaler,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[ProcessedArray, ProcessedArray, ProcessedArray]:
    """
    Aplica o scaler já ajustado em treino, validação e teste.
    """
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_val_scaled, x_test_scaled