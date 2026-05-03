# Arquitetura de Deploy

## Visão geral

Para esse projeto escolhemos uma arquitetura de microsserviço provisionado através de uma API real-time utilizando FastAPI Nela temos um total de 3 endpoints principais, sendo eles
- ` GET /` - Endpoint de boas-vindas para verificar se a API está rodando 
- ` GET /health` - Endpoint de health check para monitoramento da API
- ` GET /predict` - Recebe um payload JSON com as features RAW, processa os dados e retorna a predição de churn

A documentação dos endpoints é gerada automaticamente pelo FastAPI e pode ser acessada em `/docs` ou `/redoc` quando a API estiver rodando.

## O por quer da escolha dessa arquitetura

A escolha por FastAPI se deu pela sua simplicidade, performance e facilidade de integração com modelos de ML em Python

- A análise é simples a ponto de poder ser realizada on demand via request a API ou agendada via CRON JOB  
- A inferência é leve e simples o suficiente para execução online
- O fluxo permite validação imediata do payload e retorno de probabilidade em tempo real

## Qual o funcionamento da API

- A API carrega o modelo em memória no startup, garantindo baixa latência para as predições subsequentes
- Recebe features RAW através de um request JSON 
- O pipeline transforma os dados internamente para a saída necessária
- O modelo MLP realiza a predição
- A resposta retorna classe e probabilidade

## Configurações e ambiente

- O ambiente é configurado via `settings.py` onde definimos parâmetros como `APPROVAL_THRESHOLD`, `LATENCY_WARN_MS`, `ENVIRONMENT` e `LOG_LEVEL`, o que permite ajustes rápidos sem necessidade de mudanças no código
- O modelo é carregado a partir de um caminho definido em `MODEL_PATH`, o que facilita a atualização do modelo sem necessidade de redeploy da API

## Limitações atuais

- Não há autoscaling implementado no repositório
- Não há orquestração de containers ou registry configurado como parte da entrega atual