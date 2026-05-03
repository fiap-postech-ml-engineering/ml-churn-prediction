# 🧾 Ficha técnica

- **Nome** - Churn Prediction Model
- **Data de treinamento** - Maio/2026
- **Versão do modelo** - 1.0.0
- **Tipo de modelo** - MLP em PyTorch

- **Para quais casos o modelo foi projetado**
    - Previsão de possíveis churners com base em dados financeiros e serviços contratados
    - Suporte a ações de retenção focadas em clientes com alto CLTV e risco de churn

## 🤖 Tipo de modelo

- Regressão binária com MLP em PyTorch
- Preprocessing e feature engineering via pipeline sklearn unificado
- Saída: probabilidade de churn e classe binária por threshold

## 🎯 Saída

- `probabilidade_churn` entre 0 e 1
- `classe` binária com threshold atual de 0.45

# 📊 Métricas

Para a avaliação do modelo adotamos as seguintes métricas, com foco principal em `PR AUC`, `F2 Score` e `Weighted Recall` para alinhamento com os objetivos de negócio de retenção de clientes de alto valor.

- O **PR-AUC** nos dá uma visão mais realista da performance do modelo em cenários de churn, onde a classe positiva é minoritária, o que condiz com o nosso cenário. 
- O **F2 Score** é utilizado para verificar a relação entre precisão e recall, dando mais peso ao recall, o que é crucial para retenção e servindo de critério de desempate entre modelos com PR AUC similar.
- O **Weighted Recall** nos ajuda a entender a recuperação de valor em termos de CLTV dos churners identificados e foi utilizado para nos ajudar a escolher o threshold ideal.

## 🏆 Métricas principais em ordem de importância
| **Métrica** | **Valor** |
|-------------|-----------|
|**`PR AUC`** | 0.65
|**`F2 Score`** | 0.75
|**`Weighted Recall`** | 0.86
|**`Recall`**  | 0.87
|**`Precision`**  | 0.49

## 🔎 Métricas secundárias para comparação
| **Métrica** | **Valor** |
|-------------|-----------|
|**`F1 Score`** | 0.63
|**`ROC AUC`** | 0.85
|**`Accuracy`** | 0.72

## ⚖️ Threshold escolhido e trade-off de custo

Para o nosso caso, temos a seguinte interpreação dos resultados:

| **Classe** | **Interpretação** |
|------------------|---------------------|
| **Falso negativo** | cliente de churn não identificado e possivelmente perdido |
| **Falso positivo** | cliente sem churn priorizado em campanha de retenção |
| **Verdadeiro positivo** | cliente de churn identificado e possível retenção |
| **Verdadeiro negativo** | cliente sem churn não priorizado, o que é aceitável

Especificamente para o nosso caso, em Telecomunicações, entendemos que é muito mais vantajoso abordar um cliente que não tem risco de churn _(Falso positivo)_ do que perder um cliente de alto valor _(Falso negativo)_, o que nos levou a priorizar o **Recall** _`(dos clientes que realmente vão churnar, quantos o modelo conseguiu capturar)`_ em troca de um **Precision** mais baixo _`(dos clientes que o modelo sinalizou como churn, quantos realmente vão churnar)`_.

Com isso, chegamos a um **threshold de 0.45**, onde maximizamos a recuperação de CLTV (Weighted Recall), mas ainda sim, mantemos o Precision em um patamar razoável perto dos 50%, o que é uma estratégia agressiva, mas controlada.

| Métrica | Weighted Recall | PR AUC | F2 Score | Recall | Precision |
|---------|---|---|---|---|---|
| `Threshold 0.30` | 0.92 | 0.70 | 0.75 | 0.93 | 0.43 |
| `Threshold 0.40` | 0.88 | 0.70 | 0.77 | 0.90 | 0.48 |
| **`Threshold 0.45 (escolhido)`** | **0.85** | **0.70** | **0.76** | **0.87** | **0.50** |
| `Threshold 0.50` | 0.82 | 0.70 | 0.75 | 0.84 | 0.53 |
| `Threshold 0.55` | 0.78 | 0.70 | 0.73 | 0.80 | 0.55 |
| `Threshold 0.60` | 0.72 | 0.70 | 0.70 | 0.74 | 0.59 |
| `Threshold 0.65` | 0.68 | 0.70 | 0.68 | 0.70 | 0.64 |

Para resultados mais conservadores, podemos aumentar o threshold em troca da diminuição do Weighted Recall e aumento do Precision, 

# 🗂️ Dataset utilizado

O dataset utilizado foi o [Telco customer churn: IBM dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) que contém informações de clientes de uma empresa de telecomunicações, incluindo dados demográficos, serviços contratados, dados financeiros e se o cliente churnou ou não.

As features presentes no dataset puro são:

- **CustomerID:** Um ID único que identifica cada cliente.
- **Count:** Um valor usado em relatórios/dashboards para somar o número de clientes em um conjunto filtrado.
- **Country:** País de residência principal do cliente.
- **State:** Estado de residência principal do cliente.
- **City:** Cidade de residência principal do cliente.
- **Zip Code:** CEP da residência principal do cliente.
- **Lat Long:** Combinação de latitude e longitude da residência principal do cliente.
- **Latitude:** Latitude da residência principal do cliente.
- **Longitude:** Longitude da residência principal do cliente.
- **Gender:** Gênero do cliente: Masculino, Feminino.
- **Senior Citizen:** Indica se o cliente tem 65 anos ou mais: Sim, Não.
- **Partner:** Indica se o cliente tem parceiro(a): Sim, Não.
- **Dependents:** Indica se o cliente vive com dependentes: Sim, Não. Dependentes podem ser filhos, pais, avós etc.
- **Tenure Months:** Total de meses em que o cliente permaneceu na empresa até o fim do trimestre de referência.
- **Phone Service:** Indica se o cliente assina serviço de telefone residencial com a empresa: Sim, Não.
- **Multiple Lines:** Indica se o cliente assina múltiplas linhas telefônicas: Sim, Não.
- **Internet Service:** Indica se o cliente assina serviço de internet: Não, DSL, Fibra Óptica, Cabo.
- **Online Security:** Indica se o cliente assina serviço adicional de segurança online: Sim, Não.
- **Online Backup:** Indica se o cliente assina serviço adicional de backup online: Sim, Não.
- **Device Protection:** Indica se o cliente assina plano adicional de proteção de dispositivo para equipamentos de internet: Sim, Não.
- **Tech Support:** Indica se o cliente assina plano adicional de suporte técnico com menor tempo de espera: Sim, Não.
- **Streaming TV:** Indica se o cliente usa internet para assistir TV por streaming de terceiros: Sim, Não. A empresa não cobra taxa adicional por esse serviço.
- **Streaming Movies:** Indica se o cliente usa internet para assistir filmes por streaming de terceiros: Sim, Não. A empresa não cobra taxa adicional por esse serviço.
- **Contract:** Tipo de contrato atual do cliente: Mensal, Um Ano, Dois Anos.
- **Paperless Billing:** Indica se o cliente optou por fatura digital: Sim, Não.
- **Payment Method:** Forma de pagamento da fatura: Débito em Conta, Cartão de Crédito, Cheque por Correio.
- **Monthly Charge:** Valor total mensal atual cobrado pelos serviços contratados.
- **Total Charges:** Valor total acumulado cobrado do cliente até o fim do trimestre de referência.
- **Churn Label:** Sim = cliente saiu da empresa neste trimestre. Não = cliente permaneceu. Diretamente relacionado a Churn Value.
- **Churn Value:** 1 = cliente saiu da empresa neste trimestre. 0 = cliente permaneceu. Diretamente relacionado a Churn Label.
- **Churn Score:** Valor de 0 a 100 calculado pelo IBM SPSS Modeler. O modelo considera múltiplos fatores associados ao churn. Quanto maior o score, maior a probabilidade de churn.
- **CLTV:** Customer Lifetime Value. Um CLTV previsto é calculado com fórmulas corporativas e dados existentes. Quanto maior o valor, mais valioso é o cliente. Clientes de alto valor devem ser monitorados para churn.
- **Churn Reason:** Motivo específico de saída do cliente. Diretamente relacionado à categoria de churn.

Para o nosso estudo mantivemos apenas as features mais relevantes, sendo elas:
- **Dependents**
- **Tenure Months**
- **Phone Service**
- **Multiple Lines**
- **Internet Service**
- **Online Security**
- **Online Backup**
- **Device Protection**
- **Tech Support**
- **Streaming TV**
- **Streaming Movies**
- **Contract**
- **Paperless Billing**
- **Payment Method**
- **Monthly Charge**
- **Total Charges**

E adicionando/modificando algumas features, como:
- **Stickiness** (Total de serviços ativos)
- **fiber_price_impact** (Interação Fibra vs. Preço)
- **avg_ticket** (Ticket médio)
- **is_new_customer** (Menos de 6 meses de contrato)
- **Normalização logarítmica** de Total Charges e Monthly Charges

# 🤔 O por que da escolha do MLP

A escolha do MLP foi definida após uma analise comparativa com o modelo de Regressão Logística (O melhor modelo dentre os baselines), onde o MLP apresentou uma performance superior entregando um ganho de em Recall e Weighted Recall, mantendo o mesmo PR AUC

| Métrica | MLP PyTorch | Logistic Regression | Diferença |
|---------|---|---|---|
| **Melhor Threshold** | 0.45 | 0.50 | - |
| **PR AUC** | 0.66 | 0.66 | +0.00 |
| **F2 Score** | 0.76 | 0.72 | +0.04 |
| **Weighted Recall** | 0.86 | 0.79 | +0.07 |
| **Recall** | 0.87 | 0.80 | +0.07 |
| **Precision** | 0.49 | 0.51 | -0.02 |
| - | - | - | - |
| **ROC AUC** | 0.85 | 0.85 | +0.00 |
| **Accuracy** | 0.73 | 0.74 | -0.01 |
| **F1 Score** | 0.63 | 0.62 | +0.01 |

O MLP se mostra vantajoso para esse caso em epecífico já que esse resultado foi atingido sem grandes testes de hiperparâmetros, o que indica que o modelo tem um potencial de melhoria ainda maior, enquanto a Regressão Logística já se mostrou mais limitada mesmo com ajustes de regularização e feature engineering.

Testes de arquitetura diferentes com outros otimizadores e funções de perda podem maximizar os resultados ainda mais

## 🏗️ Arquitetura utilizada

A arquitetura utilizada segue o seguinte direcionamento:
```
35 features
    ↓
  Dense(256) + BatchNorm + ReLU + Dropout(0.3)
    ↓
  Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
   Dense(64) + BatchNorm + ReLU + Dropout(0.2)
    ↓
   Dense(32) + BatchNorm + ReLU + Dropout(0.1)
    ↓
  Dense(1) → LOGIT
    ↓
σ(logit) → Prob [0,1]
```

**Total de parâmetros treináveis: 53.697**

- **Loss** - `BCEWithLogitsLoss` com `pos_weight=2.77` — penaliza erros em churners proporcionalmente ao desbalanceamento de classes (73%/27%), forçando o modelo a priorizar recall sobre precision
- **Optimizer** - Adam com `lr=1e-4` — taxa de aprendizado adaptativa por parâmetro; valor conservador escolhido para estabilidade com BatchNorm
- **LR Scheduler** - `ReduceLROnPlateau(factor=0.5, patience=5)` — reduz o learning rate pela metade quando a PR-AUC de validação estagna, evitando oscilações tardias no treinamento
- **Regularização** - Dropout progressivo (0.3 → 0.1) e BatchNorm — dropout mais agressivo nas camadas iniciais (maior capacidade, maior risco de overfitting) e mais suave nas finais; BatchNorm estabiliza as ativações entre camadas
- **Função de ativação** - ReLU — evita o problema de vanishing gradient em redes profundas
- **Early Stopping** - Monitora PR-AUC no validation set com `patience=20` — PR-AUC é mais informativa que ROC-AUC em datasets desbalanceados pois foca no desempenho sobre a classe positiva (churners)

# ⚠️ Limitações

- O modelo depende da qualidade e completude das features RAW enviadas para inferência.
- O comportamento pode degradar se a distribuição de dados em produção mudar significativamente.
- O modelo não substitui regra de negócio; ele apenas prioriza casos com maior risco.

## 🧭 Vieses e cuidados

- O dataset inclui variáveis sensíveis ou proxy de segmentação, como `Senior Citizen`, `Gender` e `Partner`.
- Essas variáveis são usadas apenas como preditores estatísticos, não como regra operacional isolada.
- Decisões de retenção devem ser revisadas por contexto comercial.

## 🚨 Cenários de falha

- Payload incompleto ou com colunas RAW faltantes
- Mudança forte de distribuição entre treino e produção
- Clientes com perfis muito raros fora da base histórica