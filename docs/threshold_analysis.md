# Análise de Threshold e Custo FP/FN

## Interpretação de custo

- Falso negativo: cliente de churn não identificado e possivelmente perdido
- Falso positivo: cliente sem churn priorizado em campanha de retenção

Em churn, falso negativo tende a ser mais caro que falso positivo, porque a perda do cliente pode gerar impacto direto de receita.

## Threshold atual

O threshold adotado no v2 é 0.6.
Essa escolha favorece recall alto sem perder totalmente a precisão.

## Evidência v2

- `precision`: 0.5064308681672026
- `recall`: 0.8422459893048129
- `f1`: 0.6325301204819277

Essa combinação indica que o modelo encontra a maior parte dos churns, ainda que gere uma quantidade relevante de alertas.

## Recomendação

- Manter 0.6 como threshold operacional inicial
- Revisar para baixo somente se o time tiver orçamento para mais campanhas
- Revisar para cima se o custo de abordagem for alto e o funil de retenção estiver saturado

## Regra prática

- Threshold menor: mais recall, mais campanhas, mais falsos positivos
- Threshold maior: menos campanhas, menos recall, menos falsos positivos