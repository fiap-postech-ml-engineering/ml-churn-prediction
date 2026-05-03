# Arquitetura de Deploy

## 🗺️ Visão geral

Para esse projeto escolhemos uma arquitetura de microsserviço stateless exposto via API REST síncrona, implementada com **FastAPI + Uvicorn**. O serviço encapsula o pipeline completo de inferência — desde a recepção de features RAW até a resposta com classe e probabilidade de churn — sem exigir que o cliente conheça os detalhes internos de pré-processamento ou engenharia de features.

A API expõe 3 endpoints principais:

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Informações da API e status dos artefatos carregados |
| `GET` | `/health` | Health check individual por componente (modelo, scaler, features) |
| `POST` | `/predict` | Inferência de churn para um cliente |

A documentação interativa é gerada automaticamente pelo FastAPI e pode ser acessada em `/docs` (Swagger) ou `/redoc` quando a API estiver rodando.

## 🤔 O porquê da escolha dessa arquitetura

A escolha por FastAPI se deu pela sua performance assíncrona nativa, validação automática de payload via Pydantic e facilidade de integração com pipelines de ML em Python — sem overhead de frameworks mais pesados.

- A inferência via MLP com 36 features em CPU é sub-milissegundo; o custo dominante é I/O de rede, o que torna a execução online viável
- O fluxo permite validação imediata do payload e retorno de probabilidade em tempo real
- O mesmo endpoint pode ser consumido on demand ou orquestrado em batch via CRON JOB externo, sem necessidade de uma arquitetura separada para cada modo

## 🔁 Qual o funcionamento da API

Os artefatos do modelo são carregados **uma única vez no startup** e mantidos em memória durante toda a vida do processo, eliminando latência de I/O por request. Se os artefatos v2 não forem encontrados, o serviço faz fallback automático para v1 — sem crash do processo.

O ciclo de vida de uma requisição `POST /predict`:

1. **Middleware de observabilidade** — gera ou propaga o header `X-Request-ID` (UUID), inicia contador de latência e injeta o request_id em todos os logs da requisição
2. **Validação Pydantic** — verifica estrutura e tipos do payload JSON (retorna 422 se inválido)
3. **Validação do feature contract** — verifica presença das features críticas antes de chegar ao modelo (retorna 422 com descrição do campo faltante)
4. **Pipeline de pré-processamento** — aplica feature engineering (6 features derivadas), imputação, one-hot encoding e StandardScaler internamente, sem intervenção do cliente
5. **Inferência MLP** — forward pass em `eval mode` com `torch.no_grad()`, sigmoid sobre o logit e aplicação do threshold configurável
6. **Resposta** — retorna classe, probabilidade e echo do payload recebido

Ao final, o middleware loga latência total e emite `WARNING` se ultrapassar `LATENCY_WARN_MS` (500ms).

## ⚙️ Configurações e ambiente

Todas as configurações são gerenciadas via variáveis de ambiente (`.env`), sem valores hardcoded nas rotas:

| Variável | Descrição |
|---|---|
| `APPROVAL_THRESHOLD` | Limiar de classificação (0.45 atual) |
| `LATENCY_WARN_MS` | Limiar para alerta de latência (padrão: 500ms) |
| `ENVIRONMENT` | Ativa logging em arquivo e desativa debug em produção |
| `LOG_LEVEL` | Nível de verbosidade dos logs |
| `TABULAR_MLP_MODEL_PATH` | Caminho dos pesos do modelo PyTorch |
| `TABULAR_PREPROCESSING_PIPELINE_PATH` | Caminho do pipeline sklearn serializado |

Todos os logs são emitidos em **JSON estruturado**, enriquecidos com `request_id`, `latency_ms`, `status_code` e `client_ip`, facilitando ingestão em ferramentas de observabilidade.

## ⚠️ Limitações atuais

| Limitação | Impacto |
|---|---|
| Sem containerização | Deploy manual, portabilidade não garantida entre ambientes |
| Sem CI/CD configurado | Testes e deploy são executados manualmente |
| Processo único (sem workers paralelos) | Sem paralelismo real de CPU em picos de carga |
| Sem autoscaling | Gargalo sob alta demanda |
| Sem versionamento de endpoint (`/v1/`) | Breaking changes afetam todos os clientes |
| Sem autenticação | API aberta, sem controle de acesso |
| Sem rate limiting | Vulnerável a abuso ou sobrecarga acidental |
| Sem registry de modelos ativo | Troca de versão do modelo é feita manualmente |
