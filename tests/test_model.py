"""Tests for the LSTM model."""

import torch

from fred_forecasting.models.lstm import LSTM


def test_forward_output_shape():
    model = LSTM(input_size=1, hidden_size=8, num_layers=1, output_size=1)
    x = torch.randn(4, 5, 1)  # (batch, window, features)
    out = model(x)
    assert out.shape == (4, 1)


def test_forward_multifeature_multioutput():
    model = LSTM(input_size=3, hidden_size=16, num_layers=2, output_size=2)
    x = torch.randn(2, 7, 3)
    out = model(x)
    assert out.shape == (2, 2)


def test_parameters_are_trainable():
    model = LSTM(hidden_size=8, num_layers=1)
    assert any(p.requires_grad for p in model.parameters())


def test_backward_produces_gradients():
    model = LSTM(input_size=1, hidden_size=8, num_layers=1, output_size=1)
    x = torch.randn(3, 4, 1)
    out = model(x).sum()
    out.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
