"""
Time2Vec-enhanced Transformer model.
Implements Time2Vec layer for learnable time representations.
"""

import torch
import torch.nn as nn
import math


class Time2Vec(nn.Module):
    """
    Time2Vec layer for encoding time representations.

    t2v(t)[i] =
        w_i * t + b_i,                    (i == 0, linear)
        sin(w_i * t + b_i),               (i > 0, periodic)

    The first dimension captures linear time flow,
    while the remaining dimensions capture periodic patterns.
    """

    def __init__(self, t2v_dim: int = 16):
        super().__init__()

        self.t2v_dim = t2v_dim

        self.w = nn.Parameter(torch.randn(t2v_dim))
        self.b = nn.Parameter(torch.randn(t2v_dim))

        nn.init.uniform_(self.w, -1.0, 1.0)
        nn.init.uniform_(self.b, -1.0, 1.0)

    def forward(self, time_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_indices: Tensor of shape (batch, seq_len, 1)
                          Normalized time indices [0, 1]

        Returns:
            Time2Vec encoding of shape (batch, seq_len, t2v_dim)
        """
        t = time_indices.squeeze(-1)

        linear = self.w[0] * t + self.b[0]
        linear = linear.unsqueeze(-1)

        periodic_inputs = t.unsqueeze(-1).expand(-1, -1, self.t2v_dim - 1)
        periodic = torch.sin(periodic_inputs * self.w[1:] + self.b[1:])

        t2v = torch.cat([linear, periodic], dim=-1)

        return t2v


class Time2VecTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        market_dim: int = 0, # <-- NEW
        temperature: float = 1.0, # <-- NEW
        t2v_dim: int = 16,
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
        self.market_dim = market_dim
        self.temperature = temperature
        self.t2v_dim = t2v_dim
        self.d_model = d_model
        self.horizon = horizon

        # --- NEW: Market Gating Projection ---
        if market_dim > 0:
            self.gate_proj = nn.Linear(market_dim, input_dim)
        # -------------------------------------

        self.time2vec = Time2Vec(t2v_dim)
        self.input_projection = nn.Linear(input_dim + t2v_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncodingT2V(d_model, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, horizon)
        self._init_weights()

    # ... keep _init_weights the same ...
    def _init_weights(self):
        # Initialize the new gating projection layer if it exists
        if hasattr(self, 'gate_proj'):
            nn.init.xavier_uniform_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)
            
        # Initialize original layers
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor, market_status: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        features = x[:, :, :-1]
        time_indices = x[:, :, -1:]

        # --- NEW: Apply Market-Guided Gating ---
        if self.market_dim > 0 and market_status is not None:
            gate_weights = self.gate_proj(market_status) 
            alpha = torch.softmax(gate_weights / self.temperature, dim=-1)
            alpha = alpha * self.input_dim 
            features = features * alpha.unsqueeze(1)
        # ---------------------------------------

        time_embeddings = self.time2vec(time_indices)
        combined = torch.cat([features, time_embeddings], dim=-1)

        combined = self.input_projection(combined)
        combined = self.layer_norm(combined)
        combined = self.pos_encoder(combined)
        combined = self.transformer_encoder(combined)

        last_token = combined[:, -1, :]
        output = self.fc_out(last_token)

        return output.squeeze(-1) if self.horizon == 1 else output


class PositionalEncodingT2V(nn.Module):
    """Positional encoding for Time2Vec Transformer."""

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
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    batch_size = 32
    seq_len = 60
    input_dim = 14

    x = torch.randn(batch_size, seq_len, input_dim + 1)

    model = Time2VecTransformer(
        input_dim=input_dim,
        t2v_dim=16,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        horizon=1,
    )

    output = model(x)
    print(f"Time2Vec Transformer output shape: {output.shape}")
