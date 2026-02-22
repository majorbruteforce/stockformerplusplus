"""
Stockformer: Transformer model with sinusoidal positional encoding.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Stockformer(nn.Module):
    """
    Transformer encoder with sinusoidal positional encoding.

    Architecture:
        Input projection -> Positional Encoding -> Transformer Encoder
        -> Last token -> Linear -> Output
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        horizon: int = 1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.horizon = horizon

        self.input_projection = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc_out = nn.Linear(d_model, horizon)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch, horizon)
        """
        x = self.input_projection(x)

        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        last_token = x[:, -1, :]

        output = self.fc_out(last_token)

        return output.squeeze(-1) if self.horizon == 1 else output


if __name__ == "__main__":
    batch_size = 32
    seq_len = 60
    input_dim = 14

    x = torch.randn(batch_size, seq_len, input_dim)

    model = Stockformer(
        input_dim=input_dim,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        horizon=1,
    )

    output = model(x)
    print(f"Stockformer output shape: {output.shape}")
