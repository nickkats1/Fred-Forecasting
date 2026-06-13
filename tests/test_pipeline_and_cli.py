"""End-to-end pipeline and CLI tests with FRED ingestion mocked."""

import numpy as np
import pandas as pd
import pytest

from fred_forecasting import cli, pipeline
from fred_forecasting.config import Settings
from fred_forecasting.metrics import Metrics


@pytest.fixture
def fake_fetch(monkeypatch):
    """Patch fetch_data everywhere it is used to return a deterministic frame."""
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    # A smooth sine wave is easy for the LSTM to fit in a few epochs.
    values = np.sin(np.linspace(0, 6 * np.pi, num=120)) + 5.0
    df = pd.DataFrame({"Date": dates, "TESTSERIES": values})

    def _fetch(series_id, api_key=None):
        return df.copy()

    monkeypatch.setattr(pipeline, "fetch_data", _fetch)
    return df


def _fast_settings() -> Settings:
    return Settings(
        series_id="TESTSERIES",
        train_size=0.8,
        window_size=10,
        hidden_size=8,
        num_layers=1,
        epochs=3,
        seed=0,
    )


def test_run_pipeline_end_to_end(fake_fetch):
    result = pipeline.run_pipeline(_fast_settings())

    assert isinstance(result.metrics, Metrics)
    assert len(result.history) == 3
    # validation set has len(val) - window_size rows
    n_val = 120 - int(120 * 0.8)
    assert len(result.predictions) == n_val - 10
    assert list(result.predictions.columns) == ["Date", "Actual", "Predicted"]
    assert result.predictions["Actual"].notna().all()


def test_run_pipeline_is_reproducible(fake_fetch):
    r1 = pipeline.run_pipeline(_fast_settings())
    r2 = pipeline.run_pipeline(_fast_settings())
    assert r1.metrics.rmse == pytest.approx(r2.metrics.rmse, rel=1e-5)


def test_cli_runs_and_writes_csv(fake_fetch, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "run_pipeline", pipeline.run_pipeline)
    out = tmp_path / "preds.csv"
    rc = cli.main(
        [
            "--series-id",
            "TESTSERIES",
            "--epochs",
            "2",
            "--window-size",
            "10",
            "--hidden-size",
            "8",
            "--num-layers",
            "1",
            "--output-csv",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    written = pd.read_csv(out)
    assert {"Date", "Actual", "Predicted"} <= set(written.columns)
    captured = capsys.readouterr()
    assert "Validation metrics" in captured.out


def test_settings_from_args_only_overrides_provided():
    parser = cli.build_parser()
    args = parser.parse_args(["--epochs", "11"])
    settings = cli.settings_from_args(args)
    assert settings.epochs == 11
    # untouched fields keep defaults
    assert settings.series_id == Settings().series_id


def test_select_device_returns_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pipeline.select_device().type == "cpu"
