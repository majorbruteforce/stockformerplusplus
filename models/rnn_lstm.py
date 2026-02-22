"""
RNN and LSTM models for financial time series forecasting.
"""

import torch
import torch.nn as nn
from typing import Optional


class RNNModel(nn.Module):
    """
    Vanilla RNN model for time series forecasting.

    Architecture:
        Input -> RNN (2 layers, batch_first) -> Dropout -> Linear -> Output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, horizon)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch, horizon)
        """
        out, hidden = self.rnn(x)

        last_hidden = hidden[-1]

        last_hidden = self.dropout(last_hidden)

        output = self.fc(last_hidden)

        return output.squeeze(-1) if self.horizon == 1 else output


class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting.

    Architecture:
        Input -> LSTM (2 layers, batch_first) -> Dropout -> Linear -> Output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, horizon)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch, horizon)
        """
        out, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        last_hidden = self.dropout(last_hidden)

        output = self.fc(last_hidden)

        return output.squeeze(-1) if self.horizon == 1 else output


def get_model(model_name: str, input_dim: int, horizon: int = 1, **kwargs):
    """
    Factory function to get model by name.

    Args:
        model_name: 'rnn' or 'lstm'
        input_dim: Number of input features
        horizon: Prediction horizon
        **kwargs: Additional model arguments

    Returns:
        Model instance
    """
    if model_name.lower() == "rnn":
        return RNNModel(input_dim, horizon=horizon, **kwargs)
    elif model_name.lower() == "lstm":
        return LSTMModel(input_dim, horizon=horizon, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    batch_size = 32
    seq_len = 60
    input_dim = 14

    x = torch.randn(batch_size, seq_len, input_dim)

    rnn = RNNModel(input_dim, hidden_dim=64, num_layers=2, horizon=1)
    lstm = LSTMModel(input_dim, hidden_dim=64, num_layers=2, horizon=1)

    rnn_out = rnn(x)
    lstm_out = lstm(x)

    print(f"RNN output shape: {rnn_out.shape}")
    print(f"LSTM output shape: {lstm_out.shape}")
