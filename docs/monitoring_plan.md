# Plano de Monitoramento

## 🎯 Objetivo

Garantir que o comportamento do modelo em produção permaneça dentro dos limites esperados em relação ao treino — tanto em performance técnica (latência, erros) quanto em comportamento do modelo (distribuição de predições, drift de features).

## 📊 Métricas a monitorar

### Infraestrutura e API
- Volume de requisições por período
- Latência p50 / p95 / p99 da API (baseline esperado: p95 < 500ms)
- Taxa de erro por endpoint (422 e 5xx)
- Taxa de payload inválido (indicador de mudança de contrato pelo cliente)

### Comportamento do modelo
- Proporção de predições positivas (baseline de treino: ~26.5% de churners)
- Média e quantis das probabilidades previstas
- Distribuição das features RAW de entrada, com foco em `Monthly Charges`, `Tenure Months` e `Total Charges`

### Negócio
- Weighted Recall realizado nas campanhas de retenção (comparando churners previstos com churners confirmados no período)

## 🚨 Alertas sugeridos

| Sinal | Critério de alerta |
|---|---|
| Erros de payload (422) | Aumento > 20% em relação ao baseline do período anterior |
| Erros de servidor (5xx) | Qualquer ocorrência acima de zero em janela de 1h |
| Queda de volume | Redução > 50% no volume de predições em relação à média do período |
| Drift de proporção de churn | Proporção de predições positivas fora do intervalo 15%–40% |
| Drift de features | Desvio > 2 desvios padrão na média de `Monthly Charges`, `Tenure Months` ou `Total Charges` em relação ao conjunto de treino |
| Probabilidade média extrema | Média das probabilidades previstas abaixo de 0.15 ou acima de 0.60 |

## 🔍 Sinais de drift

- Mudança na média, mediana ou quartis das features RAW em relação ao conjunto de treino
- Aumento de valores ausentes ou valores fora da faixa histórica
- Proporção de predições positivas divergindo do baseline de treino (~26.5%)
- Divergência entre a probabilidade média atual e a probabilidade média registrada no treino

Quando qualquer um desses sinais for detectado, seguir o playbook abaixo antes de acionar re-treino.

## 📋 Playbook de resposta

1. **Validar schema de entrada** — verificar se houve mudança no contrato de features enviado pelo cliente
2. **Revisar logs da API** — identificar exemplos de payload com erro e padrões de falha recentes
3. **Comparar distribuições** — contrastar a distribuição atual das features com o conjunto de treino
4. **Revisar threshold** — avaliar se o threshold atual (0.45) ainda é adequado para o perfil de clientes em produção
5. **Acionar re-treino** — se o drift for persistente ou as métricas de negócio degradarem, iniciar novo ciclo de treinamento com dados atualizados

## 🛠️ O que precisa ser implementado para produção

| Componente | Status atual | O que falta |
|---|---|---|
| Logs estruturados da API | ✅ Implementado | — |
| Estatísticas de entrada e saída por request | ✅ Implementado (middleware) | — |
| Agregação de métricas por período | ❌ Não implementado | Ferramenta de APM (Datadog, Grafana, CloudWatch) |
| Alertas automáticos | ❌ Não implementado | Regras configuradas na ferramenta de APM |
| Monitoramento de drift automatizado | ❌ Não implementado | Pipeline de comparação de distribuições (ex: Evidently AI) |
| Auditoria de predições | ❌ Não implementado | Armazenamento de amostras de input/output para revisão posterior |
