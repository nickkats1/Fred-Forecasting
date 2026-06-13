"""Sliding-window construction and tensor conversion."""

import numpy as np
import torch


def sliding_window(
    series: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised input/target pairs from a 1D-ish value array.

    Each input ``X[i]`` is ``window_size`` consecutive observations and the
    target ``y[i]`` is the observation immediately following the window.

    Args:
        series: Array of shape ``(n, 1)`` (or ``(n,)``) of ordered values.
        window_size: Number of timesteps per input sequence; must be >= 1.

    Returns:
        ``(X, y)`` where ``X`` has shape ``(n - window_size, window_size, 1)``
        and ``y`` has shape ``(n - window_size, 1)``.

    Raises:
        ValueError: If ``window_size < 1`` or the series is too short.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    series = np.asarray(series, dtype=float)
    if series.ndim == 1:
        series = series.reshape(-1, 1)

    if len(series) <= window_size:
        raise ValueError(f"series length ({len(series)}) must exceed window_size ({window_size}).")

    x_list, y_list = [], []
    for i in range(len(series) - window_size):
        x_list.append(series[i : i + window_size])
        y_list.append(series[i + window_size])
    return np.asarray(x_list), np.asarray(y_list)


def convert_array_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a float32 ``torch.Tensor``."""
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)
