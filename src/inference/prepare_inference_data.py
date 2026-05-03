"""
Módulo para preparação de dados para inferência.

Responsável por:
- Normalizar dados com StandardScaler
- Tensorizar dados com PyTorch
- Criar DataLoaders para inferência
"""

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def prepare_inference_batch(
    df_features: pd.DataFrame,
    scaler,
    device: torch.device,
    feature_names: list[str] | None = None,
    batch_size: int = 32,
) -> DataLoader:
    """
    Prepara um batch de dados para inferência.

    Realiza as seguintes etapas:
    1. Normaliza dados usando o scaler pré-treinado
    2. Converte para tensores PyTorch
    3. Cria DataLoader para iteração em batches

    Args:
        df_features: DataFrame com features engineered preparadas
        scaler: StandardScaler já treinado (carregado do modelo)
        device: Torch device (cpu ou cuda)
        batch_size: Tamanho dos batches para inferência (padrão: 32)

    Returns:
        DataLoader: DataLoader contendo os dados preparados

    Raises:
        ValueError: Se DataFrame estiver vazio ou scaler for None
        RuntimeError: Se houver erro na conversão para tensor
    """

    if df_features.empty:
        raise ValueError("❌ DataFrame de features não pode estar vazio")

    if scaler is None:
        raise ValueError("❌ Scaler não foi carregado corretamente")

    try:
        logger.info(f"📊 Preparando batch para inferência: {df_features.shape}")

        # 1. Transformar dados usando o pipeline de preprocessing fitado
        x_scaled = scaler.transform(df_features)
        if hasattr(x_scaled, "toarray"):
            x_scaled = x_scaled.toarray()
        if feature_names and "CLTV" in feature_names:
            cltv_idx = feature_names.index("CLTV")
            x_scaled = np.delete(x_scaled, cltv_idx, axis=1)
        zero_ratio = float((x_scaled == 0).sum().sum()) / float(x_scaled.size)
        if zero_ratio > 0.8:
            logger.warning(
                "⚠️ Alta proporção de zeros nas features pós-preprocessing: %.2f%%",
                zero_ratio * 100,
            )
        logger.info(
            f"✓ Dados transformados | mean={x_scaled.mean():.4f}, std={x_scaled.std():.4f}"
        )

        # 2. Converter para tensor PyTorch
        x_tensor = torch.from_numpy(x_scaled).float().to(device)
        logger.info(f"✓ Tensor criado: {x_tensor.shape} | Device: {device}")

        # 3. Criar TensorDataset
        inference_dataset = TensorDataset(x_tensor)

        # 4. Criar DataLoader (sem shuffle para inferência)
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        logger.info(f"✓ DataLoader criado com {len(inference_loader)} batch(es)")

        return inference_loader

    except Exception as e:
        logger.error(f"❌ Erro ao preparar dados para inferência: {e}")
        raise RuntimeError(f"Erro na preparação de dados: {str(e)}")


def run_inference(
    inference_loader: DataLoader,
    model,
    device: torch.device,
    approval_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Executa inferência no modelo MLP.

    Args:
        inference_loader: DataLoader com dados preparados
        model: Modelo PyTorch carregado
        device: Torch device
        approval_threshold: Threshold para classificação (padrão: 0.5)

    Returns:
        Tupla contendo:
        - probabilidades_churn: ndarray com probabilidades (0-1)
        - classes: ndarray com classes preditas (0 ou 1)

    Raises:
        RuntimeError: Se modelo for None ou houver erro na predição
    """

    if model is None:
        raise RuntimeError("❌ Modelo não foi carregado corretamente")

    try:
        logger.info("🔮 Executando inferência...")

        model.eval()  # Modo de avaliação
        probabilidades_churn = []

        with torch.no_grad():
            for batch in inference_loader:
                x_batch = batch[0].to(device)

                # Predição - modelo retorna logits
                logits = model(x_batch)

                # Converter logits para probabilidades usando sigmoid
                proba = torch.sigmoid(logits).cpu().numpy()
                probabilidades_churn.extend(proba.flatten())

        probabilidades_churn = np.array(probabilidades_churn)

        # Classificar baseado no threshold
        classes = (probabilidades_churn >= approval_threshold).astype(int)

        logger.info(
            f"✓ Inferência concluída | "
            f"Probabilidade média: {probabilidades_churn.mean():.4f}"
        )

        return probabilidades_churn, classes

    except Exception as e:
        logger.error(f"❌ Erro durante inferência: {e}")
        raise RuntimeError(f"Erro na execução de inferência: {str(e)}")
