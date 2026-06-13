"""Command-line interface for the forecasting pipeline."""

import argparse

import pandas as pd

from fred_forecasting import __version__
from fred_forecasting.config import Settings
from fred_forecasting.logging_config import configure_logging, get_logger
from fred_forecasting.pipeline import run_pipeline

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="fred-forecast",
        description="Forecast a FRED economic time series with an LSTM.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--series-id", help="FRED series ID (e.g. DEXUSEU).")
    parser.add_argument("--train-size", type=float, help="Train fraction in (0, 1).")
    parser.add_argument("--window-size", type=int, help="Input sequence length.")
    parser.add_argument("--hidden-size", type=int, help="LSTM hidden state size.")
    parser.add_argument("--num-layers", type=int, help="Number of stacked LSTM layers.")
    parser.add_argument("--learning-rate", type=float, help="Adam learning rate.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--output-csv",
        help="Optional path to write the actual-vs-predicted CSV.",
    )
    parser.add_argument("--log-level", default=None, help="Logging level (default INFO).")
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Build :class:`Settings` from parsed args, overriding env-based defaults."""
    overrides = {
        key: getattr(args, key)
        for key in (
            "series_id",
            "train_size",
            "window_size",
            "hidden_size",
            "num_layers",
            "learning_rate",
            "epochs",
            "seed",
        )
        if getattr(args, key) is not None
    }
    return Settings().replace(**overrides)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    settings = settings_from_args(args)
    logger.info("Settings: %s", settings)

    result = run_pipeline(settings)

    pd.set_option("display.max_rows", 40)
    print("\nValidation metrics:")
    for name, value in result.metrics.as_dict().items():
        print(f"  {name.upper():5}: {value:.4f}")

    print("\nActual vs. Predicted (head):")
    print(result.predictions.head(10).to_string(index=False))
    print("\nActual vs. Predicted (tail):")
    print(result.predictions.tail(10).to_string(index=False))

    if args.output_csv:
        result.predictions.to_csv(args.output_csv, index=False)
        logger.info("Wrote predictions to %s", args.output_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
