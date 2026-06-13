"""Tests for Settings configuration."""

import pytest

from fred_forecasting.config import Settings


def test_defaults():
    s = Settings()
    assert s.series_id == "DEXUSEU"
    assert 0 < s.train_size < 1
    assert s.window_size >= 1


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("FRED_SERIES_ID", "FEDFUNDS")
    monkeypatch.setenv("FRED_EPOCHS", "7")
    monkeypatch.setenv("FRED_API_KEY", "abc123")
    s = Settings()
    assert s.series_id == "FEDFUNDS"
    assert s.epochs == 7
    assert s.fred_api_key == "abc123"


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 2.0])
def test_invalid_train_size(bad):
    with pytest.raises(ValueError, match="train_size"):
        Settings(train_size=bad)


def test_invalid_window_size():
    with pytest.raises(ValueError, match="window_size"):
        Settings(window_size=0)


def test_invalid_learning_rate():
    with pytest.raises(ValueError, match="learning_rate"):
        Settings(learning_rate=0)


@pytest.mark.parametrize("field", ["hidden_size", "num_layers", "epochs"])
def test_invalid_positive_int_fields(field):
    with pytest.raises(ValueError, match=field):
        Settings(**{field: 0})


def test_replace_overrides_and_is_frozen():
    s = Settings(epochs=10)
    s2 = s.replace(epochs=20, series_id="GDP")
    assert s.epochs == 10  # original unchanged
    assert s2.epochs == 20
    assert s2.series_id == "GDP"


def test_replace_rejects_unknown_field():
    with pytest.raises(TypeError, match="Unknown settings"):
        Settings().replace(not_a_field=1)


def test_frozen_is_immutable():
    s = Settings()
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raises FrozenInstanceError
        s.epochs = 5
