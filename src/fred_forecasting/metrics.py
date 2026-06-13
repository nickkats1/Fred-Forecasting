"""Regression metrics for evaluating forecasts."""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


@dataclass(frozen=True)
class Metrics:
    """Container for forecast evaluation metrics."""

    r2: float
    rmse: float
    mae: float
    mape: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Compute R², RMSE, MAE and MAPE between true and predicted values.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values (same shape as ``y_true``).

    Returns:
        A :class:`Metrics` instance.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return Metrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=rmse,
        mae=float(mean_absolute_error(y_true, y_pred)),
        mape=float(mean_absolute_percentage_error(y_true, y_pred)),
    )
