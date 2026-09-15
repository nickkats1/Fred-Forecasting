import os

import pandas as pd
import pytest

from src.data import fetch_data


def test_requests_the_series_that_was_asked_for(fake_fred):
    fetch_data("FEDFUNDS")

    fake_fred.get_series.assert_called_once_with("FEDFUNDS")


def test_returns_a_value_column_named_after_the_series_and_a_date_column(fake_fred):
    dataframe = fetch_data("FEDFUNDS")

    assert list(dataframe.columns) == ["FEDFUNDS", "Date"]
    assert pd.api.types.is_datetime64_any_dtype(dataframe["Date"])


def test_drops_missing_and_duplicated_rows(fake_fred):
    dataframe = fetch_data("FEDFUNDS")

    assert dataframe["FEDFUNDS"].tolist() == [1.5, 2.5]
    assert dataframe["Date"].tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-01")]


def test_returns_none_when_the_series_id_is_rejected(fake_fred):
    fake_fred.get_series.side_effect = ValueError("Bad Request")

    assert fetch_data("NOT_A_SERIES") is None


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("FRED_API_KEY"), reason="needs a real FRED API key")
def test_really_downloads_from_fred():
    dataframe = fetch_data("FEDFUNDS")

    assert not dataframe.empty
    assert list(dataframe.columns) == ["FEDFUNDS", "Date"]
