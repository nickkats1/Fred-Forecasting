"""LSTM model for univariate time-series forecasting."""

import torch
from torch import nn


class LSTM(nn.Module):
    """A stacked LSTM that maps a window of values to a single next-step value.

    Args:
        input_size: Number of features per timestep (1 for univariate series).
        hidden_size: Size of the LSTM hidden state.
        num_layers: Number of stacked LSTM layers.
        output_size: Number of values to predict (1 for single-step forecasts).
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input of shape ``(batch, window_size, input_size)``.

        Returns:
            Predictions of shape ``(batch, output_size)``.
        """
        h0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])
