# 🧾 Ficha técnica

- **Nome** - Churn Prediction Model
- **Data de treinamento** - Maio/2026
- **Versão do modelo** - 1.0.0
- **Tipo de modelo** - Classificação binária com MLP em PyTorch
- **Pipeline** - Preprocessing e feature engineering via sklearn unificado

- **Para quais casos o modelo foi projetado**
    - Previsão de possíveis churners com base em dados financeiros e serviços contratados
    - Suporte a ações de retenção focadas em clientes com alto CLTV e risco de churn

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

Para o nosso caso, temos a seguinte interpretação dos resultados:

| **Classe** | **Interpretação** |
|------------------|---------------------|
| **Falso negativo** | cliente de churn não identificado e possivelmente perdido |
| **Falso positivo** | cliente sem churn priorizado em campanha de retenção |
| **Verdadeiro positivo** | cliente de churn identificado e possível retenção |
| **Verdadeiro negativo** | cliente sem churn não priorizado, o que é aceitável |

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

Para resultados mais conservadores, podemos aumentar o threshold em troca da diminuição do Weighted Recall e aumento do Precision.

# 🗂️ Dataset utilizado

O dataset utilizado foi o [Telco customer churn: IBM dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset), contendo ~7.000 registros de clientes de uma empresa de telecomunicações americana, com dados demográficos, serviços contratados, dados financeiros e indicador de churn. A descrição completa das features originais está disponível na documentação do Kaggle.

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

A escolha do MLP foi definida após uma análise comparativa com o modelo de Regressão Logística (O melhor modelo dentre os baselines), onde o MLP apresentou uma performance superior entregando um ganho em Recall e Weighted Recall, mantendo o mesmo PR AUC.

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

O MLP se mostra vantajoso para esse caso em específico já que esse resultado foi atingido sem grandes testes de hiperparâmetros, o que indica que o modelo tem um potencial de melhoria ainda maior, enquanto a Regressão Logística já se mostrou mais limitada mesmo com ajustes de regularização e feature engineering.

Testes de arquitetura diferentes com outros otimizadores e funções de perda podem maximizar os resultados ainda mais.

## 🏗️ Arquitetura utilizada

A arquitetura utilizada segue o seguinte direcionamento:
```
36 features
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

- O modelo depende da qualidade e completude das features RAW enviadas para inferência, sendo algumas mais críticas do que outras, como:
    - Tenure Months
    - Total Charges
    - Monthly Charges
    - Internet Service
    - Online Security
    - Online Backup
    - Device Protection
    - Tech Support
    - Streaming TV
    - Streaming Movies
- O dataset de treino contém ~7.000 registros de uma empresa de telecom americana — generalização para outros mercados, perfis demográficos ou estruturas de serviço distintas não foi validada.
- Ainda não foi implementado um mecanismo de monitoramento de drift em produção.
- O modelo não substitui regra de negócio; ele apenas prioriza casos com maior risco.

## 🧭 Vieses e cuidados

- O dataset inclui variáveis sensíveis como `Senior Citizen`, `Gender` e `Partner`. Essas variáveis têm poder preditivo estatístico no dataset de treino, mas seu uso direto em critérios de retenção pode introduzir discriminação — decisões de campanha devem passar por revisão comercial.
- O dataset é de origem americana (IBM/Kaggle). Padrões de comportamento de churn podem diferir em outros países por fatores culturais, regulatórios ou de estrutura de mercado — o modelo não deve ser aplicado a outros contextos sem revalidação.

## 🚨 Cenários de falha

- **Payload incompleto** — features críticas ausentes são detectadas pela validação do feature contract antes da inferência; a API retorna 422 com descrição do campo faltante.
- **Drift de distribuição** — sem alertas automatizados, a degradação do modelo pode passar despercebida. Sinal de alerta: proporção de predições positivas divergindo do baseline de treino (~26.5% de churners).
- **Recalibração de threshold necessária** — se o perfil de clientes mudar, o threshold atual pode gerar recall insuficiente ou volume excessivo de falsos positivos, exigindo novo tuning no validation set.
- **Clientes novos (Tenure < 6 meses)** — essa faixa está sub-representada na base de treino e é a mais volátil para churn. Probabilidades geradas para esse grupo devem ser interpretadas com cautela.
- **Perfis fora da distribuição de treino** — clientes com combinações de features muito raras (ex: alto CLTV + contrato mensal + todos os serviços ativos) podem receber probabilidades pouco confiáveis.