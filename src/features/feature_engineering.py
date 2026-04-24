import pandas as pd
import numpy as np

def apply_feature_engineering(
        df : pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Aplica engenharia de atributos sobre o DataFrame e retorna o mesmo DataFrame com novas colunas.

    - Cria a coluna "Ternure Months" a partir de "Tenure Months", substituindo 0 por 1.
    - Cria "total_services" como soma das colunas de serviços existentes entre: "Online Security_Yes", "Online Backup_Yes", "Device Protection_Yes", "Tech Support_Yes", "Streaming TV_Yes" e "Streaming Movies_Yes".
    - Cria "fiber_price_impact" somente se "Internet Service_Fiber optic" existir, calculando a interação com "Monthly Charges".
    - Cria "avg_ticket" como "Total Charges" / "Tenure Months". Sobrescreve "Total Charges" com log1p de "Total Charges".
    - Cria "Monthly Charges_log" com log1p de "Monthly Charges".
    - Cria "is_new_customer" como flag inteira para "Tenure Months" < 6.
    Args:
    df (pd.DataFrame): DataFrame de entrada com as features necessárias para as
    transformações.

    Returns:
    pd.DataFrame: O próprio DataFrame de entrada, após mutações in-place, contendo
    as colunas originais e as colunas derivadas criadas.

    Raises:
    KeyError: Se alguma coluna obrigatória para as operações não existir, como
    "Tenure Months", "Total Charges" ou "Monthly Charges".
    TypeError: Se colunas usadas em operações matemáticas não tiverem tipo numérico.

    Notes:
    - A função modifica o DataFrame recebido (in-place) antes de retorná-lo.
    """

    needed_cols = ["Tenure Months", "Total Charges", "Monthly Charges", "Internet Service_Fiber optic"]
    service_cols = [
        'Online Security_Yes', 'Online Backup_Yes', 'Device Protection_Yes',
        'Tech Support_Yes', 'Streaming TV_Yes', 'Streaming Movies_Yes'
    ]
    numeric_cols = ["Tenure Months", "Total Charges", "Monthly Charges"]

    missing_needed = [c for c in needed_cols if c not in df.columns]
    missing_service = [c for c in service_cols if c not in df.columns]
    not_numeric = [c for c in numeric_cols if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]

    if missing_needed or missing_service:
       raise KeyError(f"O Dataframe precisa desssas colunas para o tratamento: {needed_cols + service_cols}")
    
    if not_numeric:
        raise TypeError(f"As seguintes colunas precisam ser numéricas para as operações matemáticas: {not_numeric}")

    # 1. Ajuste de Tenure (evitar divisão por zero)
    # HACK O ideal seria criar uma coluna auxiliar para calcular o avg_ticket, 
    # mas para evitar mudanças em outras partes do código, substituímos os zeros por 1 diretamente na coluna original.
    df['Tenure Months'] = df['Tenure Months'].replace(0, 1)

    # 2. Stickiness: Total de serviços adicionais ativos
    existing_services = [col for col in service_cols if col in df.columns]
    df["total_services"] = df[existing_services].sum(axis=1)

    # 3. Interação Fibra x Preço
    df["fiber_price_impact"] = df["Internet Service_Fiber optic"] * df["Monthly Charges"]

    # 4. Métricas Financeiras e Log para Normalização
    df["avg_ticket"] = df["Total Charges"] / df["Tenure Months"]
    df["Total Charges"] = np.log1p(df["Total Charges"])
    df["Monthly Charges_log"] = np.log1p(df["Monthly Charges"])

    # 5. Flag de cliente novo
    df["is_new_customer"] = (df["Tenure Months"] < 6).astype(int)

    return df
