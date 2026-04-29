# Model Card

## Nome do modelo

Churn Prediction Model v2

## Objetivo

Estimativa da probabilidade de churn para apoiar ações de retenção em clientes de telecom.

## Tipo de modelo

- Regressão binária com MLP em PyTorch
- Preprocessing e feature engineering via pipeline sklearn unificado
- Saída: probabilidade de churn e classe binária por threshold

## Dados de entrada

- Features RAW oficiais do contrato v2
- Dados numéricos: `Latitude`, `Longitude`, `Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV`
- Dados categóricos: `Gender`, `Senior Citizen`, `Partner`, `Dependents`, serviços contratados e tipo de pagamento

## Saída

- `probabilidade_churn` entre 0 e 1
- `classe` binária com threshold atual de 0.6

## Métricas v2

- `test_roc_auc`: 0.8535250200211838
- `test_accuracy`: 0.7402413058907026
- `test_precision`: 0.5064308681672026
- `test_recall`: 0.8422459893048129
- `test_f1`: 0.6325301204819277
- `best_val_roc_auc`: 0.8652845591464517

## Threshold e trade-off

O threshold atual de 0.6 favorece recall alto, o que é coerente para retenção: perder um cliente de churn tende a ser mais caro do que abordar um cliente que ficaria.
Na validação, o modelo mostrou boa capacidade de recuperação de churn e uma taxa aceitável de falso positivo para um fluxo de retenção.

## Limitações

- O modelo depende da qualidade e completude das features RAW enviadas para inferência.
- O comportamento pode degradar se a distribuição de dados em produção mudar significativamente.
- O modelo não substitui regra de negócio; ele apenas prioriza casos com maior risco.

## Vieses e cuidados

- O dataset inclui variáveis sensíveis ou proxy de segmentação, como `Senior Citizen`, `Gender` e `Partner`.
- Essas variáveis são usadas apenas como preditores estatísticos, não como regra operacional isolada.
- Decisões de retenção devem ser revisadas por contexto comercial.

## Cenários de falha

- Payload incompleto ou com colunas RAW faltantes
- Mudança forte de distribuição entre treino e produção
- Clientes com perfis muito raros fora da base histórica

## Uso recomendado

- Priorizar campanhas de retenção quando `probabilidade_churn >= 0.6`
- Ajustar o threshold apenas após análise de custo operacional e orçamento de retenção