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
- Detectar device: `cuda` ou `cpu`
- Configurar experimento MLflow: `mlflow.set_experiment("stg_notebook_training_V.1")`

### 1.2 Carregar dados pré-processados
- Ler `data/processed/telco_customer_churn_eda_pre-processed_encoded.csv`
- Identificar coluna target (`target` ou equivalente)
- Separar `X` e `y`
- Checar shape e distribuição de classes
- Converter `X` para `float32`, `y` para `float32`

### 1.3 Split estratificado
- Split 1: 80% train/val + 20% test (estratificado)
- Split 2: 80% train + 20% val (estratificado) a partir do trainval
- Salvar `n_features`, `n_pos`, `n_neg`, `pos_weight`

---

## Fase 2: Preparação para PyTorch

### 2.1 Normalizar dados
- Instanciar `StandardScaler`
- Ajustar/scalar em `X_train`; transformar `X_val` e `X_test`
- Validar média ~0 e desvio ~1

### 2.2 Tensor conversion
- Converter para tensores e mover para device:
  - `X_train_tensor`, `y_train_tensor`
  - `X_val_tensor`, `y_val_tensor`
  - `X_test_tensor`, `y_test_tensor`

### 2.3 DataLoaders
- Criar `TensorDataset` para train, val, test
- `DataLoader`:
  - Train: `batch_size=32`, `shuffle=True`
  - Val/Test: `batch_size=64`, `shuffle=False`

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

---

## Fase 5: Loop de Treinamento com Early Stopping e MLflow

1. `mlflow.start_run()`
2. Log params e tags (arquitetura, optimizer, pos_weight, etc.)
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
4. Recarregar `best_model_state` em `model`
5. `mlflow.pytorch.log_model(model, 'model')`
6. `mlflow.end_run()`

---

## Fase 6: Avaliação no Test Set

1. `model.eval()` e `torch.no_grad()`
2. Calcular `test_preds_all` e `test_targets_all`
3. Métricas de teste com `compute_metrics`
4. Log no MLflow (`test_roc_auc`, `test_accuracy`, etc.)
5. Criar `test_results` DataFrame com actual/proba, predicted_class

---

## Fase 7: Visualização e Comparação

- Plot 1: Training Loss vs epochs
- Plot 2: Val ROC-AUC vs epochs + best line
- Plot 3: Val Accuracy, Precision, Recall, F1 vs epochs
- Plot 4: Matriz de confusão no test set
- Tabela comparativa com baseline:
  - Baseline: LogisticRegression + class_weight (ROC-AUC ~0.84, Recall ~0.8175)
- Análise de gap e se MLP supera baseline

---

## Fase 8: Resumo e Próximos Passos

- Resumo final da execução (epochs, best ROC-AUC, test ROC-AUC)
- Observações de regularização, early stopping e saldo de detecção
- Próximos passos sugeridos:
  - Ajustar hiperparâmetros, testar scheduler, aumentar arquitetura
  - Integrar SelectKBest da fase 02
  - Validar em novas partições/testes robustos
