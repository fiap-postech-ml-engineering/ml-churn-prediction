import pytest
import torch
import torch.nn as nn

from src.models.mlp_model import MLPNetworkChurn


def test_mlp_network_churn_instantiates_with_defaults() -> None:
    model = MLPNetworkChurn()

    assert isinstance(model, nn.Module)
    assert model.input_size == 35
    assert model.hidden_dims == [256, 128, 64, 32]
    assert model.dropout_rates == [0.3, 0.3, 0.2, 0.1]


def test_mlp_network_churn_forward_returns_expected_shape() -> None:
    model = MLPNetworkChurn(input_size=35)
    x = torch.randn(4, 35)

    output = model(x)

    assert output.shape == (4, 1)


def test_mlp_network_churn_custom_architecture() -> None:
    model = MLPNetworkChurn(
        input_size=10,
        hidden_dims=[64, 32],
        dropout_rates=[0.2, 0.1],
    )
    x = torch.randn(3, 10)

    output = model(x)

    assert model.input_size == 10
    assert model.hidden_dims == [64, 32]
    assert model.dropout_rates == [0.2, 0.1]
    assert output.shape == (3, 1)


def test_mlp_network_churn_raises_for_mismatched_hidden_dims_and_dropout_rates() -> None:
    with pytest.raises(ValueError, match="same length"):
        MLPNetworkChurn(
            input_size=10,
            hidden_dims=[64, 32, 16],
            dropout_rates=[0.2, 0.1],
        )


def test_mlp_network_contains_expected_final_layer() -> None:
    model = MLPNetworkChurn()

    assert isinstance(model.network, nn.Sequential)
    assert isinstance(model.network[-1], nn.Linear)
    assert model.network[-1].out_features == 1