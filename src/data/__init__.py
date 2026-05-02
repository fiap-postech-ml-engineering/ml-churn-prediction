"""
Init do módulo data.

Contém apenas métricas de negócio usadas pelos notebooks.
"""

from .business_metrics import weighted_recall

__all__ = ["weighted_recall"]