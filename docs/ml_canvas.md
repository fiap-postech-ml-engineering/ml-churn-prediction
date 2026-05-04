# 📃ML Canvas

Neste documento trouxemos um levantamento de informações focado nos conceitos de _"Business Understanding"_ e _"Data Understanding"_

## 🎯 Qual problema queremos resolver?

Desacelerar a perda de clientes da empresa através de métricas que direcionem a atenção dos stakeholders para clientes com padrões e comportamentos que indicam um possível cancelamento (Churn)

## 📐 Métrica do negócio que consideraremos para a resolução do problema

- **Weighted Recall:** A proporção de CLTV de churners recuperada com o modelo

$$\text{Weighted Recall} = \frac{\sum_{i \in TP} CLTV_i}{\sum_{i \in \text{churners reais}} CLTV_i}$$

Além do Weighted Recall, também consideramos:

- **CLTV** - Queremos recuperar a maior quantidade de valor de CLTV possível
- **% Churn:** - Queremos reduzir a taxa de churn geral da empresa

## 🏹 O que queríamos atingir

Esperamos atingir uma recuperação de **+80% do CLTV total** dos churners nos testes, exemplo:
- De uma lista de 1000 churners, temos um CLTV somado de R$100k
- O objetivo é recuperar em valor de CLTV o equivalente a R$80k desses churners, independente da quantidade de churners 

## ✅ O que atingimos

No conjunto de testes o modelo atingiu um Weighted Recall de 86% do CLTV dos churners, fixando o Precision do modelo em 50% para garantir que metade dos clientes priorizados fossem realmente churners. Vide [model_card.md] para detalhes sobre o trade-off das métricas.

## 🔁 O que fazemos com essa informação?

A partir da classificação gerada pelo modelo entendemos que o time interessado pela segmentação dos clientes para ações de retenção (CRM, Marketing e Operações) pode priorizar os clientes com maior risco de churn, direcionando esforços e recursos para os casos com maior potencial de recuperação de CLTV.

**Fluxo de uso:**

1. **Inferência** — A API realiza inferência em tempo real por cliente. Para a classificação trimestral da base completa, um job externo (cron ou script) itera sobre os clientes e chama o endpoint `/predict` para cada um, persistindo os resultados no banco de dados *(BigQuery, EC2, etc.)*. A API funciona em realtime, mas a classificação pode ocorrer em batch de acordo com a necessidade do time de negócios

2. **Cadência** — Classificação ocorre todo quarter *(cadência definida pelo time de negócios)*
3. **Análise** — Time de negócios organiza as informações em dashboards e relatórios para acompanhamento *(Power BI, Looker, etc.)*
4. **Ação** — Time responsável pela retenção (CRM, Marketing) extrai a lista de clientes que serão alvo de:
   - Ações diretas: promoções, abordagens via telefone
   - Segmentação de campanhas: mídia paga digital *(Google Ads, Meta Ads e afins)*


## 🧪 Como atestar a assertividade das ações tomadas a partir do modelo?

As ações tomadas pelo time responsável pela retenção devem ser abordadas em grupos de clientes separados definidos a partir a predição do modelo, sempre com um grupo controle para comparação.
- **Grupo tratado**: clientes com risco _(Acima do threshold do modelo)_ → recebem ação de retenção
- **Grupo controle**: clientes com risco _(Acima do threshold do modelo)_ → não recebem ação (holdout)

Nessa análise, o KPI a ser considerado seria a **Share de churners**, dado por 

$$\text{Share de Churners} = \frac{\text{Churners no grupo}}{\text{Total do grupo}}$$

Se o Share de churners do grupo prioritário (onde tomamos as ações de retenção) for menor do que o do grupo controle, temos a confirmação de que as ações tomadas a partir do modelo conseguiram reduzir o churn mantendo a maior parte do valor de CLTV.

## 🛠️ Recursos necessários

- **Base de dados** para treinamento dos modelos cruzando os dados analíticos do negócio a nível de usuário com informações demográficas e de comportamento do usuário
- **Apoio das equipes** de Marketing, CRM e Negócios para que possamos traduzir o comportamento dos usuários em dados assertivos
- **Infraestrutura** em nuvem para CI/CD/CT do modelo e hospedagem da API onde os dados vão ser servidos

## 🗂️ Dados e variáveis relevantes
- **CLTV:** - O quanto o cliente vale para a empresa em todo o seu ciclo de vida
- **Tenure Months** - Total de meses em que a pessoa é cliente da empresa
- **Monthly Charge** - Valor mensal cobrado do cliente somando todos os serviços
- **Total Charges** - Valor total acumulado já cobrado ao cliente
- **Aderência aos Serviços** - Quantos serviços o cliente tem contratado e quais são eles?
