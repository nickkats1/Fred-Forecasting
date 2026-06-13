# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-12

### Added
- Installable `fred_forecasting` package (src layout) with a `fred-forecast`
  console script and configurable `Settings`.
- End-to-end pipeline: FRED ingestion, chronological split, scaling, sliding
  windows, LSTM training, metrics, and an actual-vs-predicted report.
- Structured logging, reproducible seeding, and CSV export of predictions.
- Test suite (pytest) with the FRED API mocked; ~99% coverage.
- Tooling: ruff lint/format, pre-commit hooks, coverage config.
- Deployment: multi-stage Dockerfile (non-root, CPU torch), docker-compose, and
  a GitHub Actions CI pipeline (lint, test matrix, Docker build).

### Fixed
- `main.py` referenced `np`/`pd` without importing them.
- Predictions were de-scaled with a second scaler re-fit on the training data;
  the pipeline now reuses the scaler fit during preprocessing.
- Ingestion errors were swallowed by a `print`; they are now logged and raised.
