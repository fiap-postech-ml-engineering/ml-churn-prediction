"""
Engenharia de Features para Predição de Churn.
"""

import numpy as np


def apply_feature_engineering(features_dict: dict) -> dict:
    """
    Calcula features derived a partir das features RAW.

    Esse processamento espelha exatamente o que foi feito durante o treinamento
    no notebook 02_baselines.

    Features DERIVED criadas:
        - total_services: Soma de serviços adicionais contratados
        - fiber_price_impact: Interação Fibra Óptica x Monthly Charges
        - Total Charges Log: Transformação log de Total Charges
        - avg_ticket: Receita média (Total Charges / Tenure Months)
        - avg_ticket_log: Log da receita média
        - is_new_customer: Flag para clientes novos (Tenure Months < 6)

    Args:
        features_dict: Dicionário com features RAW

    Returns:
        dict: Dicionário com features RAW + derived
    """

    features_eng = features_dict.copy()

    # 1. Ajuste de Tenure (evitar divisão por zero)
    tenure = max(features_eng.get("Tenure Months", 1), 1)

    # 2. Stickiness: Total de serviços adicionais ativos
    service_cols = [
        'Online Security_Yes',
        'Online Backup_Yes',
        'Device Protection_Yes',
        'Tech Support_Yes',
        'Streaming TV_Yes',
        'Streaming Movies_Yes',
    ]
    total_services = sum(features_eng.get(col, 0) for col in service_cols)
    features_eng["total_services"] = float(total_services)

    # 3. Interação Fibra x Preço (O "vilão" do Churn)
    fiber_optic = features_eng.get("Internet Service_Fiber optic", 0)
    monthly_charges = features_eng.get("Monthly Charges", 0)
    features_eng["fiber_price_impact"] = float(fiber_optic * monthly_charges)

    # 4. Transformação Log de Total Charges (normalização)
    total_charges = features_eng.get("Total Charges", 0)
    features_eng["Total Charges Log"] = float(np.log1p(total_charges))

    # 5. Ticket médio (financeiro por mês)
    features_eng["avg_ticket"] = float(total_charges / tenure)

    # 6. Transformação Log de avg_ticket
    features_eng["avg_ticket_log"] = float(np.log1p(features_eng["avg_ticket"]))

    # 7. Flag de cliente novo
    features_eng["is_new_customer"] = 1.0 if tenure < 6 else 0.0

    return features_eng
