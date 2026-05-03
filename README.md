# ML Churn Prediction

Projeto do Tech Challenge focado em previsão de churn para clientes de telecom. O contrato oficial do projeto é o pipeline, usado tanto no treino
quanto na inferência.

---
Badges.....
---

## Requisitos

- Python 3.12
- `uv` instalado para gerenciamento das dependencias

## Setup
Realize o clone do repositório com

```bash
git clone https://github.com/fiap-postech-ml-engineering/ml-churn-prediction

cd ml-churn-prediction
```
Crie o ambiente virtual
```bash
uv venv
```
Para acessar o ambiente virtual

```bash
Para Windows 		-> .venv\Scripts\activate
Para Linux / macOS 	-> source .venv/bin/activate
```
Instale as dependências com 
```
uv sync
```
Ou se quiser as dependencias de desenvolvimento (incluindo testes, lint e formatação):
```bash
uv pip install -e .[dev]
```

## Visão geral

O fluxo do projeto é:

1. Carregar o dataset bruto
2. Aplicar o pipeline de feature engineering + preprocessing
3. Treinar a MLP com PyTorch
4. Salvar os artefatos
5. Servir predições pela API FastAPI

## Estrutura do projeto

- `src/`: código-fonte do pipeline, treino, inferência e API
- `data/raw/`: dataset bruto original
- `data/processed/`: datasets processados e gerados pelo pipeline
- `models/`: pesos, pipeline e artefatos versionados
- `tests/`: testes automatizados
- `docs/`: Model Card, plano de monitoramento e demais documentos finais
- `notebooks/`: exploração e validação histórica do projeto

## Documentações importantes
- [ML Canvas](docs/ml_canvas.md) — visão de negócio, métricas, objetivos e plano de ação a partir do modelo
- [Model Card](docs/model_card.md) — detalhes técnicos do modelo, métricas, arquitetura e limitações
- [Arquitetura de Deploy](docs/deploy_architecture.md) — detalhes da implementação da API, configuração e limitações atuais
- [Plano de Monitoramento](docs/monitoring_plan.md) — métricas, alertas e playbook para monitoramento em produção

## Executar a API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints principais:

- `GET /health`
- `POST /predict`
- `GET /docs`

## Treinar novamente

O treino reprodutível fora do notebook usa o pipeline v2 e regenera os artefatos.

```bash
python -m src.training.train_model
```

Artefatos gerados:

- `models/preprocessing/churn_tabular_preprocessing_pipeline_v2.joblib`
- `models/mlp/churn_mlp_best_state_dict_v2.pth`
- `models/mlp/churn_mlp_input_features_v2.joblib`
- `models/mlp/churn_mlp_metrics_v2.json`

## Testes e validação

```bash
make test
```

```bash
make lint
```

```bash
make format
```

```bash
make check
```

O projeto também mantém testes de API, schema, smoke e paridade offline vs API.

## Como reproduzir uma inferência

1. subir a API com `uvicorn main:app --reload`
2. enviar um JSON com as features RAW oficiais do dataset
3. a API aplica o pipeline v2 persistido e retorna classe + probabilidade

Exemplo de payload:

```json
{
	"features": {
		"Gender": "Female",
		"Senior Citizen": 0,
		"Partner": "Yes",
		"Dependents": "No",
		"Tenure Months": 12,
		"Phone Service": "Yes",
		"Multiple Lines": "No",
		"Internet Service": "Fiber optic",
		"Online Security": "No",
		"Online Backup": "Yes",
		"Device Protection": "No",
		"Tech Support": "No",
		"Streaming TV": "Yes",
		"Streaming Movies": "Yes",
		"Contract": "Month-to-month",
		"Paperless Billing": "Yes",
		"Payment Method": "Electronic check",
		"Monthly Charges": 79.95,
		"Total Charges": 900.5,
		"Latitude": 0.0,
		"Longitude": 0.0,
		"CLTV": 0
	}
}
```

## MLflow

O treino v2 registra parâmetros, métricas e artefatos essenciais no MLflow local.
O tracking fica em `mlruns/`.

## Pontos de melhoria
-
-
-
