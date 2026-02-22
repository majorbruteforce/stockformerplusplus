"""
Configuration file for Stockformer+ financial forecasting framework.
Contains all hyperparameters, paths, and constants.
"""

import os
import random
import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

API_KEY = "A4G119PMOFNFAAVC"
SYMBOL = "AAPL"

DATA_CONFIG = {
    "api_key": API_KEY,
    "symbol": SYMBOL,
    "output_size": "full",
    "years_of_data": 10,
}

FEATURE_CONFIG = {
    "seq_len": 60,
    "horizons": [1],
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "rolling_mean_short": 5,
    "rolling_mean_long": 20,
    "rsi_period": 14,
    "transaction_cost": 0.001,
}

MODEL_CONFIG = {
    "rnn": {
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
    },
    "lstm": {
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
    },
    "stockformer": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
    },
    "time2vec_transformer": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "t2v_dim": 16,
    },
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 10,
    "min_delta": 1e-4,
}

RESULT_DIR = "results"
DATA_DIR = "data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
