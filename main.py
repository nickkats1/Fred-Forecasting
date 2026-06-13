"""Convenience entry point: ``python main.py`` runs the forecasting CLI.

The packaged ``fred-forecast`` console script (see pyproject.toml) is the
preferred entry point; this shim keeps the original ``python main.py`` workflow
working after installation (``pip install -e .``).
"""

from fred_forecasting.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
