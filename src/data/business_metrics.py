import numpy as np


def weighted_recall(y_true, y_pred, cltv):
    """
    y_true → array (n,) com labels reais (0/1)
    y_pred → array (n,) com predições binárias (0/1) — depois de aplicar threshold
    cltv   → array (n,) com CLTV de cada cliente
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Se cltv não foi fornecido, retornar None para indicar indisponibilidade
    if cltv is None:
        return None

    cltv = np.asarray(cltv, dtype=float)

    # Se cltv for escalar ou não tiver o mesmo tamanho de y_true, não podemos computar
    if cltv.ndim == 0 or cltv.shape[0] != y_true.shape[0]:
        return None

    tp_mask = (y_true == 1) & (y_pred == 1)
    actual_pos = y_true == 1

    cltv_capturado = cltv[tp_mask].sum()
    cltv_total_risco = cltv[actual_pos].sum()

    if cltv_total_risco == 0:
        return 0.0
    return cltv_capturado / cltv_total_risco
