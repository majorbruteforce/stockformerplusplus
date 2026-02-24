"""
Reusable training and evaluation functions for all models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm
from config import TRAIN_CONFIG, DEVICE


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> float:
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(X)

        if outputs.dim() > 1 and outputs.shape[1] > 1:
            y = y.unsqueeze(1).expand(-1, outputs.shape[1])

        loss = criterion(outputs, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on validation/test set.

    Args:
        model: PyTorch model
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average loss, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)

            if outputs.dim() > 1 and outputs.shape[1] > 1:
                y = y.unsqueeze(1).expand(-1, outputs.shape[1])

            loss = criterion(outputs, y)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return total_loss / n_batches, all_preds, all_targets


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = TRAIN_CONFIG["epochs"],
    lr: float = TRAIN_CONFIG["learning_rate"],
    weight_decay: float = TRAIN_CONFIG["weight_decay"],
    patience: int = TRAIN_CONFIG["patience"],
    min_delta: float = TRAIN_CONFIG["min_delta"],
    device: torch.device = DEVICE,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full training loop with early stopping and learning rate scheduling.

    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        patience: Early stopping patience
        min_delta: Minimum change to qualify as improvement
        device: Device to train on
        verbose: Print progress

    Returns:
        Dictionary with training history
    """
    model = model.to(device)

    criterion = nn.HuberLoss(delta=1.0)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_model_state = None

    history = {"train_loss": [], "val_loss": [], "lr": []}

    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        scheduler.step(val_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            if verbose:
                print(
                    f"  Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, "
                    f"Val Loss = {val_loss:.6f} (best)"
                )
        else:
            patience_counter += 1
            if verbose:
                print(
                    f"  Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, "
                    f"Val Loss = {val_loss:.6f} (patience: {patience_counter})"
                )

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history["best_val_loss"] = best_val_loss

    return history


def get_predictions(
    model: nn.Module, dataloader: DataLoader, device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get model predictions and targets.

    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device

    Returns:
        Tuple of (predictions, targets)
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)

            outputs = model(X)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


if __name__ == "__main__":
    print("Training utilities loaded successfully")
