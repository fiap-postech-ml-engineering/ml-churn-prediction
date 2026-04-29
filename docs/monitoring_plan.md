# Plano de Monitoramento

## Objetivo

Garantir que o comportamento do modelo em produção permaneça coerente com o treino v2.

## Métricas a monitorar

- Volume de requisições por período
- Latência p50/p95/p99 da API
- Taxa de erro por endpoint
- Proporção de classes preditas
- Média e quantis das probabilidades previstas
- Distribuição das features RAW de entrada
- Drift de dados e de saída do modelo
- Taxa de payload inválido

## Alertas sugeridos

- Aumento de 422 ou 5xx acima do baseline
- Queda abrupta no volume de predições
- Desvio relevante na distribuição de `Monthly Charges`, `Tenure Months` e `Total Charges`
- Aumento sustentado de predições extremas em uma única classe
- Drift de features críticas acima do limiar definido pelo time

## Sinais de drift

- Mudança na média, mediana ou quartis das features RAW
- Aumento de valores ausentes ou valores fora de faixa
- Mudança na taxa de churn prevista versus histórico
- Divergência entre probabilidade média atual e baseline do treino

## Playbook de resposta

1. validar se houve mudança no schema de entrada
2. revisar logs da API e exemplos de payload com falha
3. comparar distribuição atual com o conjunto de treino
4. revisar threshold e orçamento de retenção
5. acionar re-treino se houver drift persistente ou degradação de métricas

## O que seria monitorado em produção

- Logs estruturados da API
- Estatísticas resumidas de entrada e saída
- Taxa de decisões por faixa de probabilidade
- Amostras de previsões para auditoria posterior