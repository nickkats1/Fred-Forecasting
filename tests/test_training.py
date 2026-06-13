"""Tests for the training loop."""

import pytest
import torch
from torch import nn

from fred_forecasting.models.lstm import LSTM
from fred_forecasting.training import EpochRecord, set_seed, train_validate


def _make_data():
    torch.manual_seed(0)
    x_train = torch.randn(16, 5, 1)
    y_train = torch.randn(16, 1)
    x_val = torch.randn(4, 5, 1)
    y_val = torch.randn(4, 1)
    return x_train, y_train, x_val, y_val


def test_train_validate_returns_history():
    x_train, y_train, x_val, y_val = _make_data()
    model = LSTM(hidden_size=8, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = train_validate(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        epochs=5,
    )
    assert len(history) == 5
    assert all(isinstance(r, EpochRecord) for r in history)
    assert history[0].epoch == 1
    assert history[-1].epoch == 5


def test_training_reduces_train_loss():
    x_train, y_train, x_val, y_val = _make_data()
    model = LSTM(hidden_size=16, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    history = train_validate(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        epochs=30,
    )
    assert history[-1].train_rmse < history[0].train_rmse


def test_invalid_epochs():
    x_train, y_train, x_val, y_val = _make_data()
    model = LSTM(hidden_size=4, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError, match="epochs"):
        train_validate(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            optimizer=optimizer,
            loss_fn=nn.MSELoss(),
            epochs=0,
        )


def test_log_every_throttles_without_error():
    x_train, y_train, x_val, y_val = _make_data()
    model = LSTM(hidden_size=4, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = train_validate(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        epochs=4,
        log_every=10,
    )
    # still records every epoch even when logging is throttled
    assert len(history) == 4


def test_set_seed_is_reproducible():
    set_seed(123)
    a = torch.randn(3)
    set_seed(123)
    b = torch.randn(3)
    assert torch.equal(a, b)
