"""
Arquitetura da Rede Neural MLP para Classificação de Churn.
"""

import torch.nn as nn


class MLPNetworkChurn(nn.Module):
    """
    Rede Neural MLP para Classificação de Churn.

    Arquitetura:
        Input (35) → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
                  → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
                  → Dense(64)  → BatchNorm → ReLU → Dropout(0.2)
                  → Dense(32)  → BatchNorm → ReLU → Dropout(0.1)
                  → Output(1) [logit]
    """

    def __init__(
        self,
        input_size=35,
        hidden_dims=[256, 128, 64, 32],
        dropout_rates=[0.3, 0.3, 0.2, 0.1],
    ):
        super(MLPNetworkChurn, self).__init__()

        if len(hidden_dims) != len(dropout_rates):
            raise ValueError("hidden_dims and dropout_rates must have the same length.")

        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates

        layers = []
        prev_dim = input_size

        # Construir camadas ocultas
        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        # Camada de output
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
