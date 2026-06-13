"""End-to-end forecasting pipeline: fetch, train, evaluate, report."""

from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn

from fred_forecasting.config import Settings
from fred_forecasting.data.ingestion import DATE_COLUMN, fetch_data
from fred_forecasting.data.transformation import inverse_transform, split_data, transform
from fred_forecasting.logging_config import get_logger
from fred_forecasting.metrics import Metrics, compute_metrics
from fred_forecasting.models.lstm import LSTM
from fred_forecasting.training import EpochRecord, set_seed, train_validate
from fred_forecasting.windowing import convert_array_to_tensor, sliding_window

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Outputs of a completed pipeline run."""

    model: LSTM
    metrics: Metrics
    history: list[EpochRecord]
    predictions: pd.DataFrame  # columns: Date, Actual, Predicted


def select_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_pipeline(settings: Settings, device: torch.device | None = None) -> PipelineResult:
    """Run the full forecasting pipeline for the configured FRED series.

    Args:
        settings: Pipeline configuration.
        device: Torch device to train on; auto-selected if omitted.

    Returns:
        A :class:`PipelineResult` with the trained model, metrics, training
        history and an actual-vs-predicted DataFrame over the validation set.
    """
    set_seed(settings.seed)
    device = device or select_device()
    logger.info("Running pipeline on device: %s", device)

    df = fetch_data(settings.series_id, api_key=settings.fred_api_key)

    train_values, val_values = split_data(df, settings.train_size)
    train_scaled, val_scaled, scaler = transform(train_values, val_values)

    x_train, y_train = sliding_window(train_scaled, settings.window_size)
    x_val, y_val = sliding_window(val_scaled, settings.window_size)

    x_train_t = convert_array_to_tensor(x_train).to(device)
    y_train_t = convert_array_to_tensor(y_train).to(device)
    x_val_t = convert_array_to_tensor(x_val).to(device)
    y_val_t = convert_array_to_tensor(y_val).to(device)

    model = LSTM(
        input_size=1,
        hidden_size=settings.hidden_size,
        num_layers=settings.num_layers,
        output_size=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    loss_fn = nn.MSELoss()

    history = train_validate(
        model,
        x_train_t,
        y_train_t,
        x_val_t,
        y_val_t,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=settings.epochs,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        y_pred = model(x_val_t)

    y_pred_np = y_pred.cpu().numpy()
    y_val_np = y_val_t.cpu().numpy()

    # Metrics on the scaled values (the space the model was trained in).
    metrics = compute_metrics(y_val_np, y_pred_np)
    logger.info(
        "Validation metrics | R2: %.4f | RMSE: %.4f | MAE: %.4f | MAPE: %.4f",
        metrics.r2,
        metrics.rmse,
        metrics.mae,
        metrics.mape,
    )

    # Reuse the scaler fit on training data (no re-fit) to recover real units.
    pred_rescaled = inverse_transform(scaler, y_pred_np).flatten()
    actual_rescaled = inverse_transform(scaler, y_val_np).flatten()

    train_len = len(train_values)
    val_dates = df[DATE_COLUMN].iloc[train_len + settings.window_size :].reset_index(drop=True)
    predictions = pd.DataFrame(
        {
            DATE_COLUMN: val_dates.to_numpy(),
            "Actual": actual_rescaled,
            "Predicted": pred_rescaled,
        }
    )

    return PipelineResult(
        model=model,
        metrics=metrics,
        history=history,
        predictions=predictions,
    )
