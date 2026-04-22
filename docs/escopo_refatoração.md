## 1. Carregar o CSV
### **`src.data.load_data.load_csv_data()`**

- Verifica shape para validar carregamento correto

## 2. Selecionar features
### **`SEM FUNÇÃO DEFINIDA`**

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
### **`SEM FUNÇÃO DEFINIDA`**

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

## 4. One Hot Encoding
### **`SEM FUNÇÃO DEFINIDA`**
  - Aplicar no dataset com as features já selecionadas 

## 5. Feature Engineering
### **`src.features.feature_engineering.apply_feature_engineering()`**

### 5.1. Ajuste de Tenure
- Replace: 0 → 1 (Evita divisão por 0)

### 5.2. Stickiness (Total de serviços ativos)
- Nova feature: `total_services`
- Contagem das colunas de serviço:
  - Online Security
  - Online Backup
  - Device Protection
  - Tech Support
  - Streaming TV
  - Streaming Movies

### 5.3. Interação Fibra vs. Preço
- Nova feature: `fiber_price_impact`
- Multiplicação: Internet Service Fiber Optic × Monthly Charge

### 5.4. Métricas Financeiras
- Nova feature: `avg_ticket` (Ticket médio)
- Normalização com log na Total Charges
- Normalização com log na Monthly Charges

### 5.5. Segmentação de Clientes
- Nova feature: `is_new_customer` (Tenure Months < 6)

## 6. Selecionar 30% das features (ANOVA F-Value) (12 no notebook)
## NÃO VAMOS APLICAR NO MLP, PODE DESCONSIDERAR

> ⚠️ Critério: Poder discriminativo entre churners e não-churners

| # | Feature | Tipo |
|----|---------|------|
| 1 | Tenure Months | int64 |
| 2 | Contract_Two year | int64 |
| 3 | fiber_price_impact | float64 |
| 4 | Total Charges | float64 |
| 5 | Dependents_Yes | int64 |
| 6 | Internet Service_Fiber optic | int64 |
| 7 | Payment Method_Electronic check | int64 |
| 8 | is_new_customer | int64 |
| 9 | Online Security_No internet service | int64 |
| 10 | Streaming Movies_No internet service | int64 |
| 11 | Online Backup_No internet service | int64 |
| 12 | Device Protection_No internet service | int64 |

## 7. Separar dataset em treino, teste e validação (ESTRATIFICADO)
### **`SEM FUNÇÃO DEFINIDA`**
- Primeira divisão: 80% treino_full | 20% teste
- Segunda divisão: 60% treino | 20% validação (do treino_full)
- Calcula `pos_weight` para usar com loss function (BCEWithLogitsLoss)

## 8. Normalizar dados com StandardScaler
### **`SEM FUNÇÃO DEFINIDA`**
- Fit somente no `x_train`
- Transform no `x_val` e `x_test`

## 9. Tensorizar dados com o pytorch
### **`SEM FUNÇÃO DEFINIDA`**
- Todos os subsets de dados, depois do scaler

## 10. Cria os dataloaders
### **`SEM FUNÇÃO DEFINIDA`**
- Cria o TensorDataset com os subsets x e y
- Cria os dataloaders com o batch definido no settings.py e shuffle = True no subset de treino

## 11. Predict
### **`SEM FUNÇÃO DEFINIDA`**
- Colocar o modelo em modo de inferência
- Inferir sem gradiente com `with torch.no_grad():`
- Fazer foward pass nos dados
- Converter logit em probabilidade com sigmoid `probs = torch.sigmoid(logits).cpu().numpy()`

## 12. Carrega arquitetura do MLP
## SOMENTE PARA REFERENCIA, NÃO PRECISAMOS IMPLEMENTAR NA API
### **`SEM FUNÇÃO DEFINIDA`**
`classe MLPNetworkChurn em src.models.mlp_model.py`

```
INPUT (12 features)
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
| **Batch Size** | 32 |d
