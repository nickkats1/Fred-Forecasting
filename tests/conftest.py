"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_series_df() -> pd.DataFrame:
    """A small, deterministic FRED-like frame: Date + one value column."""
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    values = np.linspace(1.0, 2.0, num=60)
    return pd.DataFrame({"Date": dates, "TESTSERIES": values})


@pytest.fixture
def ascending_values() -> np.ndarray:
    """A simple ascending 2D array of shape (n, 1)."""
    return np.arange(10, dtype=float).reshape(-1, 1)
