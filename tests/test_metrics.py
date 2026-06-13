"""Tests for regression metrics."""

import numpy as np
import pytest

from fred_forecasting.metrics import Metrics, compute_metrics


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_metrics(y, y)
    assert m.r2 == pytest.approx(1.0)
    assert m.rmse == pytest.approx(0.0)
    assert m.mae == pytest.approx(0.0)
    assert m.mape == pytest.approx(0.0)


def test_known_errors():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 3.0])  # off by 1 on first element
    m = compute_metrics(y_true, y_pred)
    assert m.mae == pytest.approx(1 / 3)
    assert m.rmse == pytest.approx(np.sqrt(1 / 3))


def test_metrics_as_dict_keys():
    m = compute_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert set(m.as_dict()) == {"r2", "rmse", "mae", "mape"}


def test_metrics_returns_floats():
    m = compute_metrics(np.array([1.0, 2.0]), np.array([1.1, 1.9]))
    assert isinstance(m, Metrics)
    assert all(isinstance(v, float) for v in m.as_dict().values())
