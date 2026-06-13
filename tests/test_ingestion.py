"""Tests for FRED data ingestion (fredapi is mocked — no network)."""

import numpy as np
import pandas as pd
import pytest
import requests

from fred_forecasting.data import ingestion


class _FakeFred:
    """Stand-in for fredapi.Fred returning a preset series."""

    def __init__(self, series, api_key=None):
        self._series = series
        self.api_key = api_key

    def get_series(self, series_id):
        return self._series


def _patch_fred(monkeypatch, series):
    def factory(api_key=None):
        return _FakeFred(series, api_key=api_key)

    monkeypatch.setattr(ingestion, "Fred", factory)


def test_fetch_data_returns_clean_frame(monkeypatch):
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    series = pd.Series([1.0, 2.0, np.nan, 2.0, 4.0], index=idx)
    _patch_fred(monkeypatch, series)

    df = ingestion.fetch_data("TESTSERIES")

    assert list(df.columns) == ["Date", "TESTSERIES"]
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    # NaN row dropped
    assert df["TESTSERIES"].isna().sum() == 0
    assert len(df) == 4
    # sorted by date
    assert df["Date"].is_monotonic_increasing


def test_fetch_data_passes_api_key(monkeypatch):
    captured = {}

    def factory(api_key=None):
        captured["api_key"] = api_key
        return _FakeFred(pd.Series([1.0], index=pd.date_range("2021-01-01", periods=1)))

    monkeypatch.setattr(ingestion, "Fred", factory)
    ingestion.fetch_data("X", api_key="secret")
    assert captured["api_key"] == "secret"


def test_fetch_data_empty_raises(monkeypatch):
    series = pd.Series([np.nan, np.nan], index=pd.date_range("2021-01-01", periods=2))
    _patch_fred(monkeypatch, series)
    with pytest.raises(ValueError, match="no usable observations"):
        ingestion.fetch_data("EMPTY")


def test_fetch_data_propagates_value_error(monkeypatch):
    def factory(api_key=None):
        raise ValueError("Bad Request; Check Series ID")

    monkeypatch.setattr(ingestion, "Fred", factory)
    with pytest.raises(ValueError, match="Bad Request"):
        ingestion.fetch_data("NOPE")


def test_fetch_data_propagates_network_error(monkeypatch):
    class _RaisingFred:
        def __init__(self, api_key=None):
            pass

        def get_series(self, series_id):
            raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(ingestion, "Fred", _RaisingFred)
    with pytest.raises(requests.exceptions.RequestException):
        ingestion.fetch_data("X")
