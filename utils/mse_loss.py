"""
Standard MSE loss for prediction training.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Standard MSE loss for regression."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(y_pred, y_true)
