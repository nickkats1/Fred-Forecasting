"""Shared fixtures for the test suite."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_series() -> pd.Series:
    """A FRED-shaped series carrying one missing value and one duplicated row."""
    index = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-03-01"])
    return pd.Series([1.5, np.nan, 2.5, 2.5], index=index)


@pytest.fixture
def fake_fred(raw_series: pd.Series) -> Mock:
    """Stand in for the Fred client so no request ever leaves the machine."""
    client = Mock()
    client.get_series.return_value = raw_series
    with patch("src.data.Fred", return_value=client):
        yield client
