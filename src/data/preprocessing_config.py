from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuração de preprocessing com defaults vindos de settings."""

    target_source_column: str = settings.TARGET_COLUMN
    target_column: str = settings.PREPROCESSING_TARGET_COLUMN
    total_charges_column: str = settings.TOTAL_CHARGES_COLUMN
    columns_to_drop: tuple[str, ...] = field(
        default_factory=lambda: tuple(settings.PREPROCESSING_COLUMNS_TO_DROP)
    )
    pipeline_path: Path = settings.PREPROCESSING_PIPELINE_PATH
    random_seed: int = settings.RANDOM_SEED
    test_size: float = settings.TEST_SIZE
    val_size: float = settings.VALIDATION_SIZE


DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()

# Backward compatibility for modules importing these constants directly.
TARGET_SOURCE_COLUMN = DEFAULT_PREPROCESSING_CONFIG.target_source_column
TARGET_COLUMN = DEFAULT_PREPROCESSING_CONFIG.target_column
TOTAL_CHARGES_COLUMN = DEFAULT_PREPROCESSING_CONFIG.total_charges_column
COLUMNS_TO_DROP = DEFAULT_PREPROCESSING_CONFIG.columns_to_drop
DEFAULT_PREPROCESSING_PIPELINE_PATH = DEFAULT_PREPROCESSING_CONFIG.pipeline_path

TABULAR_TARGET_COLUMN = settings.TARGET_COLUMN
TABULAR_PIPELINE_TYPE = "tabular_mlp_preprocessing"

# Backward compatibility: ProcessedArray
import numpy as np
ProcessedArray = np.ndarray