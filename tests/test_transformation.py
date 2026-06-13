"""Tests for split/scale transformations."""

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from fred_forecasting.data.transformation import inverse_transform, split_data, transform


def test_split_data_chronological(sample_series_df):
    train, val = split_data(sample_series_df, 0.8)
    assert len(train) == 48
    assert len(val) == 12
    # order preserved: train comes before val
    assert train[-1, 0] < val[0, 0]
    assert train.shape[1] == 1


@pytest.mark.parametrize("bad", [0.0, 1.0, -1, 5])
def test_split_invalid_train_size(sample_series_df, bad):
    with pytest.raises(ValueError, match="train_size"):
        split_data(sample_series_df, bad)


def test_split_empty_partition_raises():
    df = __import__("pandas").DataFrame({"v": [1.0, 2.0]})
    with pytest.raises(ValueError, match="empty split"):
        split_data(df, 0.01)


def test_transform_fits_on_train_only():
    train = np.array([[0.0], [10.0]])
    val = np.array([[20.0]])
    train_scaled, val_scaled, scaler = transform(train, val)
    # train scaled to [0, 1]
    assert train_scaled.min() == pytest.approx(0.0)
    assert train_scaled.max() == pytest.approx(1.0)
    # val extrapolates beyond 1 since fit on train only
    assert val_scaled[0, 0] > 1.0
    assert isinstance(scaler, MinMaxScaler)


def test_inverse_transform_roundtrip():
    train = np.array([[1.0], [2.0], [3.0], [4.0]])
    val = np.array([[2.5]])
    _, val_scaled, scaler = transform(train, val)
    recovered = inverse_transform(scaler, val_scaled)
    assert recovered[0, 0] == pytest.approx(2.5)


def test_inverse_transform_handles_1d():
    train = np.array([[0.0], [4.0]])
    _, _, scaler = transform(train, train)
    out = inverse_transform(scaler, np.array([0.5]))
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(2.0)
