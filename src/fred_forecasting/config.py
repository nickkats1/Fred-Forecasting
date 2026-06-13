"""Configuration for the forecasting pipeline.

Settings are plain values with sensible defaults. The FRED API key is read from
the ``FRED_API_KEY`` environment variable (the only required external secret).
Hyperparameters can be overridden by environment variables prefixed with
``FRED_`` (for example ``FRED_EPOCHS=50``) which makes the pipeline easy to
configure in containers and CI without code changes.
"""

import os
from dataclasses import dataclass, field, fields


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    """Pipeline configuration.

    Attributes:
        series_id: A valid FRED series ID to fetch and forecast.
        train_size: Fraction of the series used for training (0 < x < 1).
        window_size: Length of each input sequence for the LSTM.
        hidden_size: Number of features in the LSTM hidden state.
        num_layers: Number of stacked LSTM layers.
        learning_rate: Adam optimizer learning rate.
        epochs: Number of training epochs.
        seed: Random seed for reproducibility.
        fred_api_key: FRED API key; falls back to the ``FRED_API_KEY`` env var.
    """

    series_id: str = field(default_factory=lambda: _env("FRED_SERIES_ID", "DEXUSEU"))
    train_size: float = field(default_factory=lambda: float(_env("FRED_TRAIN_SIZE", "0.80")))
    window_size: int = field(default_factory=lambda: int(_env("FRED_WINDOW_SIZE", "20")))
    hidden_size: int = field(default_factory=lambda: int(_env("FRED_HIDDEN_SIZE", "128")))
    num_layers: int = field(default_factory=lambda: int(_env("FRED_NUM_LAYERS", "2")))
    learning_rate: float = field(default_factory=lambda: float(_env("FRED_LEARNING_RATE", "0.001")))
    epochs: int = field(default_factory=lambda: int(_env("FRED_EPOCHS", "100")))
    seed: int = field(default_factory=lambda: int(_env("FRED_SEED", "42")))
    fred_api_key: str | None = field(default_factory=lambda: os.environ.get("FRED_API_KEY"))

    def __post_init__(self) -> None:
        if not 0 < self.train_size < 1:
            raise ValueError(f"train_size must be in (0, 1), got {self.train_size}")
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        for name in ("hidden_size", "num_layers", "epochs"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

    def replace(self, **changes: object) -> "Settings":
        """Return a copy of these settings with the given fields overridden."""
        valid = {f.name for f in fields(self)}
        unknown = set(changes) - valid
        if unknown:
            raise TypeError(f"Unknown settings: {sorted(unknown)}")
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        current.update(changes)
        return Settings(**current)
