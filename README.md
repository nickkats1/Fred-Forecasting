# Fred-Forecasting

[![CI](https://github.com/nickkats1/Fred-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/nickkats1/Fred-Forecasting/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Forecast economic time series from the [Federal Reserve Economic Data
(FRED)](https://fred.stlouisfed.org/) database using a PyTorch LSTM.

Given a FRED series ID, the pipeline fetches the series, splits it
chronologically, scales it, builds sliding-window sequences, trains an LSTM, and
reports validation metrics (R², RMSE, MAE, MAPE) alongside an actual-vs-predicted
table.

## Project layout

```
src/fred_forecasting/
├── cli.py            # `fred-forecast` command-line entry point
├── config.py         # Settings (env- and CLI-configurable)
├── logging_config.py # structured logging
├── metrics.py        # R²/RMSE/MAE/MAPE
├── pipeline.py       # end-to-end orchestration
├── training.py       # train/validate loop + seeding
├── windowing.py      # sliding-window + tensor conversion
├── data/             # FRED ingestion + scaling/splitting
└── models/lstm.py    # LSTM model
tests/                # pytest suite (FRED API mocked)
notebooks/            # exploratory analysis (ARIMA, LSTM, tree models)
```

## Installation

Requires Python 3.10+. A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)
is required to fetch data (it is free).

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install with CPU PyTorch (recommended for most users):
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"           # editable install + dev tooling
```

Then provide your API key (copy `.env.example` to `.env`, or export it):

```bash
export FRED_API_KEY=your_key_here
```

## Usage

Run the pipeline via the installed console script:

```bash
fred-forecast --series-id DEXUSEU --epochs 100 --output-csv predictions.csv
```

Or the equivalent entry point:

```bash
python main.py --series-id FEDFUNDS --window-size 20
```

All options (see `fred-forecast --help`):

| Flag | Default | Description |
| --- | --- | --- |
| `--series-id` | `DEXUSEU` | A valid FRED series ID |
| `--train-size` | `0.80` | Train fraction, in (0, 1) |
| `--window-size` | `20` | Input sequence length |
| `--hidden-size` | `128` | LSTM hidden state size |
| `--num-layers` | `2` | Stacked LSTM layers |
| `--learning-rate` | `0.001` | Adam learning rate |
| `--epochs` | `100` | Training epochs |
| `--seed` | `42` | Random seed |
| `--output-csv` | _(none)_ | Write actual-vs-predicted CSV |

Any flag can also be set via environment variables (`FRED_SERIES_ID`,
`FRED_EPOCHS`, …); CLI flags take precedence. See `.env.example`.

## Docker

```bash
docker build -t fred-forecasting .
docker run --rm -e FRED_API_KEY=$FRED_API_KEY \
  -v "$PWD/data:/data" \
  fred-forecasting --series-id DEXUSEU --epochs 50 --output-csv /data/predictions.csv
```

Or with Compose (reads `FRED_API_KEY` from your environment / `.env`):

```bash
docker compose run --rm forecast
```

## Development

```bash
pytest                 # run the test suite with coverage
ruff check .           # lint
ruff format .          # format
pre-commit install     # enable git hooks (lint/format on commit)
```

The test suite mocks the FRED API, so no network or API key is needed to run it.

## Notebooks

`notebooks/fed-funds/` contains exploratory analyses (ARIMA, LSTM, and
tree-based models) used during development. They are not part of the packaged
pipeline; install the dev/notebook extras (`pip install -r requirements-dev.txt`)
to run them.

## License

MIT — see [LICENSE](LICENSE).
