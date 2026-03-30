# Plano Detalhado para 03_training.ipynb

## Objetivo
Implementar um MLP em PyTorch para previsão de churn, incluindo:
- Definição da arquitetura de rede neural (4 camadas ocultas + BatchNorm + Dropout)
- Loop de treinamento com validação estratificada e early stopping baseado em ROC-AUC
- Batching eficiente usando DataLoaders do PyTorch
- Log de métricas no MLflow e comparação com baseline de Logistic Regression

---

## Fase 1: Setup e Carregamento de Dados

### 1.1 Imports e Configuração Inicial
- Importar bibliotecas:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`
  - `sklearn` (`train_test_split`, `StratifiedKFold`, `StandardScaler`, métricas)
  - `mlflow`, `mlflow.pytorch`
- Configurar seed: `np.random.seed(42)`, `torch.manual_seed(42)`
  - Razão: garantir reprodutibilidade em splits, inicialização de pesos e shuffling.
- Detectar device: `cuda` ou `cpu`
  - Razão: usar GPU quando disponível para treinamento mais rápido.
- Configurar experimento MLflow: `mlflow.set_experiment("stg_notebook_training_V.1")`
  - Razão: centralizar logs e facilitar comparativo com baselines.

### 1.2 Carregar dados pré-processados
- Ler `data/processed/telco_customer_churn_eda_pre-processed_encoded.csv`
- Identificar coluna target (`target` ou equivalente)
- Separar `X` e `y`
- Checar shape e distribuição de classes (`value_counts`, proporção)
  - Razão: entender desbalanceamento antes de definir pos_weight/oversampling.
- Converter `X` para `float32`, `y` para `float32`
  - Razão: PyTorch opera com float32, evita conversão em tempo de execução.

### 1.3 Split estratificado
- Split 1: 80% train/val + 20% test (estratificado)
- Split 2: 80% train + 20% val (estratificado) a partir do trainval
  - Razão: manter proporção de classes em todos os conjuntos para metricas válidas.
- Salvar `n_features`, `n_pos`, `n_neg`, `pos_weight`
  - Razão: pos_weight para BCEWithLogitsLoss compensa classe minoritária.

---

## Fase 2: Preparação para PyTorch

### 2.1 Normalizar dados
- Instanciar `StandardScaler`
- Ajustar/scalar em `X_train`; transformar `X_val` e `X_test`
- Validar média ~0 e desvio ~1
  - Razão: a normalização ajuda convergência mais rápida e evita saturação de Neurônios devido a diferentes escalas.

### 2.2 Tensor conversion
- Converter para tensores e mover para device:
  - `X_train_tensor`, `y_train_tensor`
  - `X_val_tensor`, `y_val_tensor`
  - `X_test_tensor`, `y_test_tensor`
- Manter dtype correto (`float32`) e target 0.0/1.0.
  - Razão: evitar erro de tipos e garantir compatibilidade com `BCEWithLogitsLoss`.

### 2.3 DataLoaders
- Criar `TensorDataset` para train, val, test
- `DataLoader`:
  - Train: `batch_size=32`, `shuffle=True`
  - Val/Test: `batch_size=64`, `shuffle=False`
- Razão: "batching" reduz custo de memória e permite gradientes mais estáveis; shuffle evita aprendizados de ordem.

---

## Fase 3: Arquitetura MLP

### 3.1 Definição da classe
```python
class MLPNetworkChurn(nn.Module):
    def __init__(self, input_size, hidden_dims, dropout_rates):
        super().__init__()
        layers = []
        prev_dim = input_size
        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
```

### 3.2 Instanciar e mover device
- `input_size = n_features`
- `hidden_dims = [128, 64, 32, 16]`
- `dropout_rates = [0.3, 0.3, 0.2, 0.1]`
- `model.to(device)`
- Contar parâmetros tot/treináveis
- Razões:
  - Camadas densas crescentes/decrementais criam boa expressividade sem overfit extremo.
  - BatchNorm acelera convergência e permite maiores taxas de aprendizado por normalizar ativações.
  - Dropout regulariza para evitar overfitting (especialmente importante com <10k amostras).
  - Contagem de parâmetros ajuda avaliar viabilidade de implantação e inferência.

---

## Fase 4: Configuração de Treinamento

### 4.1 Loss e optimizer
- `optimizer = optim.Adam(model.parameters(), lr=0.001)`
- `criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))`

### 4.2 Métricas
- Função `compute_metrics(y_true, y_pred_proba)` inclui:
  - `roc_auc`, `accuracy`, `precision`, `recall`, `f1`
- `predicted_class = (proba >= 0.5)`

### 4.3 Early stopping
- `early_stopping_patience = 15`
- `best_val_roc_auc = -inf`
- `epochs_without_improvement = 0`
- Razões:
  - Early stopping evita overfitting ao interromper quando a métrica de validação não melhora.
  - ROC-AUC é boa para conjunto desbalanceado (foca no ranking de scores).
  - Patience 15 permite estabilidade e evita stop prematuro devido a pequenas flutuações.

---

## Fase 5: Loop de Treinamento com Early Stopping e MLflow

1. `mlflow.start_run()`
2. Log params e tags (arquitetura, optimizer, pos_weight, etc.)
   - Razão: garantir histórico de escolhas de hiperparâmetros para reprodução e comparação.
3. Loop para cada época em `range(200)`:
   - Treino por batch:
     - `model.train()`
     - Forward, loss, backward, optimizer.step()
     - somar `train_loss_total`
   - Validação sem grad (`model.eval()`, `torch.no_grad()`)
     - Calcular probabilidades: `torch.sigmoid(logits)`
     - Concatenar predições e targets
     - Calcular métricas com `compute_metrics`
   - Log métricas no MLflow (`mlflow.log_metrics(..., step=epoch)`)
   - Early stopping: salvar melhor `state_dict` e zerar contador
   - Parar se `epochs_without_improvement >= early_stopping_patience`
   - Razão: monitorar métricas a cada época permite avaliar estabilidade e detectar overfitting precoce.
4. Recarregar `best_model_state` em `model`
   - Razão: usar o melhor checkpoint obtido em validação para teste final.
5. `mlflow.pytorch.log_model(model, 'model')`
   - Razão: persistir o modelo treinado para deploy/reprodutibilidade.
6. `mlflow.end_run()`

---

## Fase 6: Avaliação no Test Set

1. `model.eval()` e `torch.no_grad()`
2. Calcular `test_preds_all` e `test_targets_all`
3. Métricas de teste com `compute_metrics`
   - Razão: o teste final deve ser feito com dados mantidos fora do treino/validação para estimativa honestamente generalizável.
4. Log no MLflow (`test_roc_auc`, `test_accuracy`, etc.)
   - Razão: manter histórico completo de métricas finais para comparações históricas e relatórios.
5. Criar `test_results` DataFrame com actual/proba, predicted_class
   - Razão: criar artefato para análise de erros (falsos positivos/negativos), threshold tuning.

---

## Fase 7: Visualização e Comparação

- Plot 1: Training Loss vs epochs
  - Razão: verificar convergência e possíveis sinais de overfitting/underfitting.
- Plot 2: Val ROC-AUC vs epochs + best line
  - Razão: confirmar melhoria da métrica principal e estabilidade;
- Plot 3: Val Accuracy, Precision, Recall, F1 vs epochs
  - Razão: avaliar trade-offs, especialmente recall/precision em churn class imbalance.
- Plot 4: Matriz de confusão no test set
  - Razão: entender erros de classificação (falsos positivos e falsos negativos) no contexto de negócio.
- Tabela comparativa com baseline:
  - Baseline: LogisticRegression + class_weight (ROC-AUC ~0.84, Recall ~0.8175)
- Análise de gap e se MLP supera baseline
  - Razão: extrair conclusão prática: se a nova abordagem justifica complexidade e custo computacional.

---

## Fase 8: Resumo e Próximos Passos

- Resumo final da execução (epochs, best ROC-AUC, test ROC-AUC)
- Observações de regularização, early stopping e saldo de detecção
- Próximos passos sugeridos:
  - Ajustar hiperparâmetros, testar scheduler, aumentar arquitetura
    - Razão: explorar melhorias potenciais e evitar local minima.
  - Integrar SelectKBest da fase 02
    - Razão: reduzir dimensionalidade e melhorar interpretabilidade/tempo de inferência.
  - Validar em novas partições/testes robustos
    - Razão: certificar que resultado é consistente e não dependente de um split específico.

