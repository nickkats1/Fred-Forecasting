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

## Introduction

This codebase consists of various time series methods for Forecasting data from FRED. Methods used for forecasting the data include: LSTM, ARIMA, and XGBoost. All of this requires a fred api key. This is very simple codebase to illustrate using data from Fred. Which has very good data that has been managed well for years.

## NoteBooks Examples
- **LSTM Fed Funds Rate**: [lstm-fed-funds](/home/nick/github-projects/Fred-Forecasting/notebooks/fed-funds/_fed-funds-lstm.ipynb)
- **Arima Fed Funds**: [arima-fed-funds](/home/nick/github-projects/Fred-Forecasting/notebooks/fed-funds/_fed-funds-arima.ipynb)

## Installation

Requires Python 3.10+. A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)
is required to fetch data (it is free).

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate


pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"           # editable install + dev tooling
```

Then export your API key:

```bash
export FRED_API_KEY=your_key_here
```


### Choosing a series ID

Any FRED series works. The series ID is the last part of a series URL —
<https://fred.stlouisfed.org/series/DEXUSEU> → `DEXUSEU`.

Browse <https://fred.stlouisfed.org/> to find one. Some popular series:

| Series ID | Series | Frequency |
| --- | --- | --- |
| `DEXUSEU` | U.S. / Euro foreign exchange rate | Daily |
| `DGS10` | 10-year Treasury constant maturity rate | Daily |
| `SP500` | S&P 500 index | Daily |
| `UNRATE` | Unemployment rate | Monthly |
| `CPIAUCSL` | CPI, all urban consumers | Monthly |
| `FEDFUNDS` | Effective federal funds rate | Monthly |
| `PAYEMS` | All employees, total nonfarm | Monthly |
| `GDP` | Gross domestic product | Quarterly |

If the series ID does not exist, `fred-forecast` prints a one-line error and
exits with code 2 (as it does when `FRED_API_KEY` is not set).



## License

MIT — see [LICENSE](LICENSE).
