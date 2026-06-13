"""Tests for sliding-window construction and tensor conversion."""

import numpy as np
import pytest
import torch

from fred_forecasting.windowing import convert_array_to_tensor, sliding_window


def test_sliding_window_shapes(ascending_values):
    x, y = sliding_window(ascending_values, window_size=3)
    assert x.shape == (7, 3, 1)
    assert y.shape == (7, 1)


def test_sliding_window_contents(ascending_values):
    x, y = sliding_window(ascending_values, window_size=2)
    # first window is [0, 1] -> target 2
    assert list(x[0].flatten()) == [0.0, 1.0]
    assert y[0, 0] == 2.0
    # last window is [7, 8] -> target 9
    assert list(x[-1].flatten()) == [7.0, 8.0]
    assert y[-1, 0] == 9.0


def test_sliding_window_accepts_1d():
    x, y = sliding_window(np.arange(5, dtype=float), window_size=2)
    assert x.shape == (3, 2, 1)
    assert y.shape == (3, 1)


@pytest.mark.parametrize("bad_window", [0, -1])
def test_sliding_window_invalid_window(ascending_values, bad_window):
    with pytest.raises(ValueError, match="window_size"):
        sliding_window(ascending_values, bad_window)


def test_sliding_window_series_too_short():
    with pytest.raises(ValueError, match="must exceed window_size"):
        sliding_window(np.arange(3, dtype=float), window_size=3)


def test_convert_array_to_tensor():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor = convert_array_to_tensor(arr)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == (2, 2)
