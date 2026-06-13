"""Training and validation loop for the LSTM forecaster."""

from dataclasses import dataclass

import torch
from torch import nn

from fred_forecasting.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EpochRecord:
    """RMSE on train and validation sets after a single epoch."""

    epoch: int
    train_rmse: float
    val_rmse: float


def set_seed(seed: int) -> None:
    """Seed PyTorch (and CUDA, if present) for reproducible runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_validate(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device | str = "cpu",
    log_every: int = 1,
) -> list[EpochRecord]:
    """Train ``model`` for ``epochs`` and record train/validation RMSE.

    Uses full-batch gradient descent, which is appropriate for the small series
    typical of FRED data.

    Args:
        model: The model to train (moved to ``device``).
        x_train, y_train: Training inputs and targets.
        x_val, y_val: Validation inputs and targets.
        optimizer: Optimizer bound to ``model``'s parameters.
        loss_fn: Loss function (e.g. ``nn.MSELoss``).
        epochs: Number of training epochs (>= 1).
        device: Device to train on.
        log_every: Emit a log line every N epochs (and on the final epoch).

    Returns:
        A list of :class:`EpochRecord`, one per epoch.

    Raises:
        ValueError: If ``epochs < 1``.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    model = model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    history: list[EpochRecord] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_pred = model(x_train)
        loss = loss_fn(train_pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_rmse = torch.sqrt(loss_fn(model(x_train), y_train)).item()
            val_rmse = torch.sqrt(loss_fn(model(x_val), y_val)).item()

        history.append(EpochRecord(epoch, train_rmse, val_rmse))
        if epoch % log_every == 0 or epoch == epochs:
            logger.info(
                "Epoch %d/%d | train_rmse: %.4f | val_rmse: %.4f",
                epoch,
                epochs,
                train_rmse,
                val_rmse,
            )

    return history
