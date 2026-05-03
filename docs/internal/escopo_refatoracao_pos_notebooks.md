# Contexto
Este documento tem como intuito organizar as etapas necessárias para a produtização dos estudos realizados dentro dos notebooks.
---

## 1. Carregar o CSV
- Verifica shape para validar carregamento correto

## 2. Selecionar features

- Dependents
- Tenure Months
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charge
- Total Charges
- Churn Value -> [TARGET]
     
## 3. Tipar features

| Feature | Tipo |
|---------|------|
| Churn Value [TARGET] | int64 |
| Tenure Months | int64 |
| Monthly Charge | float64 |
| Total Charges | float64 |
|- |- | 
| Dependents | string |
| Phone Service | string |
| Multiple Lines | string |
| Internet Service | string |
| Online Security | string |
| Online Backup | string |
| Device Protection | string |
| Tech Support | string |
| Streaming TV | string |
| Streaming Movies | string |
| Contract | string |
| Paperless Billing | string |
| Payment Method | string |

## 4. Tratamento de Missing Values

- Padronizar valores faltantes

## 5. One Hot Encoding

- Aplicar no dataset com as features já selecionadas

## 6. Feature Engineering

### 6.1. Ajuste de Tenure
- Replace: 0 → 1 (Evita divisão por 0)

### 6.2. Stickiness (Total de serviços ativos)
- Nova feature: `total_services`
- Contagem das colunas de serviço:
  - Online Security
  - Online Backup
  - Device Protection
  - Tech Support
  - Streaming TV
  - Streaming Movies

### 6.3. Interação Fibra vs. Preço
- Nova feature: `fiber_price_impact`
- Multiplicação: Internet Service Fiber Optic × Monthly Charge

### 6.4. Métricas Financeiras
- Nova feature: `avg_ticket` (Ticket médio)
- Normalização com log na Total Charges
- Normalização com log na Monthly Charges

### 6.5. Segmentação de Clientes
- Nova feature: `is_new_customer` (Tenure Months < 6)

## 7. Separar dataset em treino, teste e validação (ESTRATIFICADO)
### NÃO PRECISAMOS PARA A PREDIÇÃO
- Primeira divisão: 80% treino_full | 20% teste
- Segunda divisão: 60% treino | 20% validação (do treino_full)
- Calcula `pos_weight` para usar com loss function (BCEWithLogitsLoss)

## 8. Normalizar dados com StandardScaler
- Fit somente no `x_train`
- Transform no `x_val` e `x_test`

## 9. Tensorizar dados com o pytorch
- Todos os subsets de dados, depois do scaler

## 10. Cria os dataloaders
- Cria o TensorDataset com os subsets x e y
- Cria os dataloaders com o batch definido no settings.py e shuffle = True no subset de treino

## 11. Predict
- Colocar o modelo em modo de inferência
- Inferir sem gradiente com `with torch.no_grad():`
- Fazer foward pass nos dados
- Converter logit em probabilidade com sigmoid `probs = torch.sigmoid(logits).cpu().numpy()`

## 12. Carrega arquitetura do MLP
## SOMENTE PARA REFERENCIA, NÃO PRECISAMOS IMPLEMENTAR NA API
`classe MLPNetworkChurn em src.models.mlp_model.py`

```
INPUT (x features)
  ↓
Dense(256) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Dense(64)  + BatchNorm + ReLU + Dropout(0.2)
  ↓
Dense(32)  + BatchNorm + ReLU + Dropout(0.1)
  ↓
Dense(1) → LOGIT (sem ativação, pois usaremos BCEWithLogitsLoss)
```

| Parâmetro | Valor |
|-----------|-------|
| **Optimizer** | Adam(lr=1e-4) |
| **Loss Function** | BCEWithLogitsLoss(pos_weight) |
| **Max Epochs** | 200 |
| **Batch Size** | 32 |
