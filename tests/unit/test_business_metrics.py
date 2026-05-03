"""Testes para o módulo business_metrics.py"""

import numpy as np
import pytest

from src.data.business_metrics import weighted_recall


class TestWeightedRecall:
    """Testes para a função weighted_recall."""

    @pytest.fixture
    def sample_data(self):
        """Cria dados de exemplo para testes."""
        # 10 amostras: 4 churn (1), 6 não-churn (0)
        y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        # Predições: 3 verdadeiros positivos, 1 falso negativo, etc.
        y_pred = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
        # CLTV correspondente
        cltv = np.array([100, 200, 150, 300, 120, 180, 250, 140, 220, 160])
        return y_true, y_pred, cltv

    def test_weighted_recall_returns_float_for_valid_data(self, sample_data):
        """Verifica se retorna float para dados válidos."""
        y_true, y_pred, cltv = sample_data
        result = weighted_recall(y_true, y_pred, cltv)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_weighted_recall_calculates_correct_value(self, sample_data):
        """Verifica se calcula o valor correto."""
        y_true, y_pred, cltv = sample_data

        # Cálculo manual:
        # True positives: índices 1, 6, 8 (y_true=1 e y_pred=1)
        # CLTV dos TP: 200 + 250 + 220 = 670
        # Total churn CLTV: índices 1, 3, 6, 8 = 200 + 300 + 250 + 220 = 970
        # Weighted recall: 670 / 970 ≈ 0.6907

        result = weighted_recall(y_true, y_pred, cltv)
        expected = 670 / 970  # ≈ 0.6907216494845361
        assert np.isclose(result, expected, rtol=1e-10)

    def test_weighted_recall_returns_none_when_cltv_is_none(self):
        """Verifica se retorna None quando cltv é None."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = weighted_recall(y_true, y_pred, None)
        assert result is None

    def test_weighted_recall_returns_none_when_cltv_is_scalar(self):
        """Verifica se retorna None quando cltv é escalar."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        cltv = 100  # escalar
        result = weighted_recall(y_true, y_pred, cltv)
        assert result is None

    def test_weighted_recall_returns_none_when_cltv_wrong_size(self):
        """Verifica se retorna None quando cltv tem tamanho errado."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        cltv = np.array([100, 200])  # tamanho errado
        result = weighted_recall(y_true, y_pred, cltv)
        assert result is None

    def test_weighted_recall_returns_zero_when_no_true_positives(self):
        """Verifica se retorna 0 quando não há churn real."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        cltv = np.array([100, 200, 150, 300])
        result = weighted_recall(y_true, y_pred, cltv)
        assert result == 0.0

    def test_weighted_recall_returns_zero_when_no_actual_churn(self):
        """Verifica se retorna 0 quando não há churn real."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])  # prediz churn mas não há
        cltv = np.array([100, 200, 150, 300])
        result = weighted_recall(y_true, y_pred, cltv)
        assert result == 0.0

    def test_weighted_recall_returns_one_when_perfect_predictions(self):
        """Verifica se retorna 1 quando todas as predições são perfeitas."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        cltv = np.array([100, 200, 150, 300])
        result = weighted_recall(y_true, y_pred, cltv)
        assert result == 1.0

    def test_weighted_recall_handles_empty_arrays(self):
        """Verifica comportamento com arrays vazios."""
        y_true = np.array([])
        y_pred = np.array([])
        cltv = np.array([])
        result = weighted_recall(y_true, y_pred, cltv)
        assert result == 0.0  # divisão por zero resulta em 0

    def test_weighted_recall_works_with_lists(self):
        """Verifica se funciona com listas em vez de arrays."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        cltv = [100, 200, 150, 300]
        result = weighted_recall(y_true, y_pred, cltv)
        assert result == 1.0

    def test_weighted_recall_ignores_false_positives(self):
        """Verifica se falsos positivos não afetam o cálculo."""
        # Mesmo predizendo churn onde não há, só TP contam
        y_true = np.array([0, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])  # FP no índice 0 e 2
        cltv = np.array([100, 200, 150, 300])

        # Apenas TP no índice 1: CLTV = 200
        # Total churn CLTV: índice 1 = 200
        # Resultado: 200/200 = 1.0

        result = weighted_recall(y_true, y_pred, cltv)
        assert result == 1.0

    def test_weighted_recall_with_different_cltv_weights(self):
        """Verifica cálculo com diferentes pesos CLTV."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])  # acerta apenas um dos dois churn

        # Cenário 1: CLTV igual
        cltv_equal = np.array([100, 100, 100, 100])
        result_equal = weighted_recall(y_true, y_pred, cltv_equal)
        assert result_equal == 0.5  # 100/200

        # Cenário 2: CLTV diferente - acerta o de maior valor
        cltv_weighted = np.array([100, 500, 100, 200])
        result_weighted = weighted_recall(y_true, y_pred, cltv_weighted)
        expected = 500 / (500 + 200)  # 500/700 ≈ 0.714
        assert np.isclose(result_weighted, expected)

    def test_weighted_recall_with_zero_cltv(self):
        """Verifica comportamento com CLTV zero."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        cltv = np.array([0, 200, 0, 300])  # alguns zeros

        # CLTV capturado: 200 + 300 = 500
        # CLTV total risco: 200 + 300 = 500
        result = weighted_recall(y_true, y_pred, cltv)
        assert result == 1.0

    def test_weighted_recall_with_all_zero_cltv(self):
        """Verifica comportamento quando todos CLTV são zero."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        cltv = np.array([0, 0, 0, 0])
        result = weighted_recall(y_true, y_pred, cltv)
        assert result == 0.0  # divisão por zero resulta em 0
