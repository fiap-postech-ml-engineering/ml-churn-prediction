# 🚀 Tech Challenge - Fase 1 (ML Engineering)

## 🎯 Objetivo
Construir um pipeline completo de ML para previsão de churn:
- Do entendimento → até API em produção

---

# 🧱 Etapas do Projeto

## 🔹 Etapa 1 — Entendimento + EDA + Baselines

### Entregas obrigatórias:
- ML Canvas preenchido
- Notebook de EDA
- Baselines (Dummy + Regressão Logística)
- Experimentos registrados no MLflow

### Pontos de atenção:
- Definir:
  - Problema de negócio
  - KPI (ex: churn rate)
  - Métrica técnica (AUC, F1, etc.)
- EDA precisa cobrir:
  - Volume
  - Qualidade
  - Distribuição
- Já versionar dataset no MLflow

### Impacto na nota:
- Clareza do problema
- Qualidade da análise
- Uso correto de métricas

---

## 🔹 Etapa 2 — Modelagem (MLP com PyTorch)

### Entregas obrigatórias:
- Modelo MLP funcional
- Comparação com baselines
- Registro no MLflow
- Tabela comparativa (≥ 4 métricas)

### Pontos de atenção:
- Implementar:
  - Early stopping
  - Batching
- Avaliar trade-off:
  - Falso positivo vs falso negativo
- Comparar com modelos simples (não só deep learning)

### Impacto na nota:
- Qualidade do modelo (25%)
- Justificativa técnica
- Comparação bem feita

---

## 🔹 Etapa 3 — Engenharia + API

### Entregas obrigatórias:
- Código refatorado (`src/`)
- Pipeline reprodutível (sklearn)
- API FastAPI com:
  - `/predict`
  - `/health`
- Testes automatizados (mínimo 3)
- Logging estruturado
- `pyproject.toml` configurado

### Pontos de atenção:
- NÃO deixar código só em notebook
- Separar:
  - dados
  - features
  - modelo
- Testes obrigatórios:
  - Smoke test
  - Schema (pandera)
  - API

### Impacto na nota:
- Qualidade do código (20%)
- API funcional (15%)
- Reprodutibilidade (15%)

---

## 🔹 Etapa 4 — Documentação + Entrega

### Entregas obrigatórias:
- README completo
- Model Card
- Arquitetura (batch vs real-time)
- Plano de monitoramento
- Vídeo (5 min - STAR)

### Pontos de atenção:
- Model Card deve incluir:
  - Performance
  - Limitações
  - Viés
- Monitoramento:
  - Métricas
  - Alertas
  - Drift
- README tem que rodar o projeto do zero

### Impacto na nota:
- Documentação (10%)
- Vídeo (10%)

---

# 📦 Requisitos obrigatórios (geral)

## Estrutura do repositório:
src/
data/
models/
tests/
notebooks/
docs/


## Boas práticas obrigatórias:
- Seeds fixados
- Validação cruzada estratificada
- MLflow (tracking completo)
- Logging (sem print)
- Lint (ruff sem erro)
- Testes automatizados
- `.gitignore` correto

---

# 🧪 Testes mínimos exigidos
- Smoke test
- Validação de schema
- Teste da API

---

# 📊 Critérios de avaliação

| Critério                     | Peso |
|----------------------------|------|
| Código e estrutura         | 20%  |
| Modelo (MLP)               | 25%  |
| Pipeline e reprodutibilidade | 15% |
| API                        | 15%  |
| Documentação               | 10%  |
| Vídeo                      | 10%  |
| Deploy (bônus)             | 5%   |

---

# ⚠️ Erros comuns (evitar)

- ❌ Só notebook (sem `src/`)
- ❌ Sem MLflow
- ❌ Sem testes
- ❌ API quebrada
- ❌ README incompleto
- ❌ Um único commit gigante
- ❌ Não comparar modelos

---

# ✅ Checklist final

- [ ] ML Canvas feito
- [ ] EDA completo
- [ ] Baselines + MLP
- [ ] MLflow ok
- [ ] Código modularizado
- [ ] API rodando
- [ ] Testes passando
- [ ] README completo
- [ ] Model Card pronto
- [ ] Vídeo gravado