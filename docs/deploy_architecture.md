# Arquitetura de Deploy

## Escolha

Deploy real-time via FastAPI.

## Justificativa

- O caso de uso pede resposta por cliente em tempo de consulta
- A API já está integrada ao pipeline v2 e aos artefatos persistidos
- A inferência é leve o suficiente para execução online
- O fluxo permite validação imediata do payload e retorno de probabilidade em tempo real

## Como a API se encaixa

- A API recebe features RAW
- O pipeline v2 transforma os dados internamente
- O modelo MLP executa a predição
- A resposta retorna classe e probabilidade

## Complemento batch

O mesmo pipeline pode ser reutilizado em lote para campanhas e análise offline.
Isso reduz risco de divergência entre treino e inferência.

## Limitações atuais

- Não há autoscaling implementado no repositório
- Não há orquestração de containers ou registry configurado como parte da entrega atual
- O foco da entrega é funcionalidade reprodutível e documentação técnica