import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def get_results(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Evaluate predictions against actual values.

    Args:
        y_true: actual outcome.
        y_pred: predicted outcome.
    """
    return {
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }
