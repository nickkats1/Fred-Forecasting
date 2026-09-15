"""Splitting, scaling, and windowing helpers for FRED time-series data."""

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler


def split_data(
    dataframe: pd.DataFrame,
    train_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split time-series data into train and validation arrays."""
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")

    values = dataframe.iloc[:, 0:1].values
    train_length = int(len(values) * train_size)

    df_train = values[:train_length]
    df_val = values[train_length:]
    return df_train, df_val


def transform(
    df_train: np.ndarray,
    df_val: np.ndarray,
    scaler: MinMaxScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Fit the scaler on train only, then scale both splits.

    Returns the fitted scaler so predictions can be mapped back to real units.
    """
    scaler = scaler or MinMaxScaler()
    scaler.fit(df_train)
    return scaler.transform(df_train), scaler.transform(df_val), scaler


def sliding_window(
    series: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised pairs where each window predicts the next observation.

    Args:
        series: Ordered values shaped ``(n,)`` or ``(n, features)``.
        window_size: Timesteps per input sequence.

    Returns:
        ``x`` shaped ``(n - window_size, window_size, features)`` and ``y``
        shaped ``(n - window_size, features)``.

    Raises:
        ValueError: If ``window_size < 1`` or the series is too short.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 1:
        series = series.reshape(-1, 1)

    if len(series) <= window_size:
        raise ValueError(
            f"series length ({len(series)}) must exceed window_size ({window_size})."
        )

    windows = sliding_window_view(series, window_size, axis=0).transpose(0, 2, 1)
    return np.ascontiguousarray(windows[:-1]), series[window_size:]


def convert_array_to_tensor(
    array: np.ndarray,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert an array to a float32 ``torch.Tensor``, optionally on ``device``."""
    return torch.as_tensor(array, dtype=torch.float32, device=device)



