import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int
    ):
        """Initialize LSTM class for time-series forecasting."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h0 = X.new_zeros(self.num_layers, X.size(0), self.hidden_size)
        c0 = X.new_zeros(self.num_layers, X.size(0), self.hidden_size)

        out, _ = self.lstm(X, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out