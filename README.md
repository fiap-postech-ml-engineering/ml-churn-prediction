# ML Churn Prediction

Projeto desenvolvido para o Tech Challenge da Pós Tech, com foco em previsão de churn de clientes de telecom usando pipeline de Machine Learning end-to-end.

## Estrutura

- `data/`: dados utilizados no projeto
- `data/raw`: dados brutos
- `data/processed`: dados processados
- `notebooks/`: análises exploratórias e experimentos
- `src/`: código-fonte do projeto
- `tests/`: testes automatizados
- `models/`: modelos treinados
- `docs/`: documentação técnica

## Objetivo

Construir um modelo preditivo de churn com boas práticas de engenharia de ML.

---

# Replicando o projeto

## Clone o repositório

```
git clone https://github.com/fiap-postech-ml-engineering/ml-churn-prediction
cd ml-churn-prediction
```

## Crie o ambiente virtual

```
python -m venv .venv
```

## Ative o ambiente:

**Caso use Windows:**

```
.venv\Scripts\activate
```

**Caso use Linux/Mac**

```
source .venv/bin/activate
```

**Caso use Git Bash no Windows**

```
source .venv\Scripts\activate
```

## Instale as dependências

```
pip install -e ".[dev]"
```

Esse comando instala as dependências do projeto e as de desenvolvimento
necessárias para notebooks (incluindo ipykernel), evitando erro ao abrir
ou selecionar kernel no VS Code/Jupyter.

Se quiser instalar apenas as dependências mínimas de execução (sem notebook):

```
pip install -e .
```
