#!/usr/bin/env python
"""Generate capital bleed trajectory figure for all sequence architectures."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from config import FEATURE_CONFIG

matplotlib.use("Agg")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"]  = 11
plt.rcParams["legend.fontsize"] = 10

RESULT_DIR = "results"

model_info = {
    "model2": ("Time2Vec", "#e41a1c", "-"),
    "model3": ("Gating", "#377eb8", "-"),
    "model4": ("RNN", "#4daf4a", "-"),
    "model5": ("LSTM", "#984ea3", "-"),
    "model6": ("Transformer+Sharpe", "#ff7f00", "-"),
}

transaction_cost = FEATURE_CONFIG["transaction_cost"]

fig, ax = plt.subplots(figsize=(10, 6))

for model_key, (model_name, color, linestyle) in model_info.items():
    try:
        preds = np.load(f"{RESULT_DIR}/{model_key}_preds.npy")
        targs = np.load(f"{RESULT_DIR}/{model_key}_targs.npy")
        
        preds = preds.flatten()
        targs = targs.flatten()
        
        position = np.where(preds > 0, 1, -1)
        
        raw_returns = position * targs
        
        position_changes = np.abs(np.diff(np.concatenate([[0], position])))
        net_costs = position_changes * transaction_cost
        strategy_returns = raw_returns - net_costs
        
        strategy_returns = np.clip(strategy_returns, -0.5, 0.5)
        
        initial_capital = 100
        cumulative_wealth = initial_capital * np.cumprod(1 + strategy_returns)
        
        cumulative_wealth = np.clip(cumulative_wealth, 0.01, 1e6)
        
        steps = np.arange(len(cumulative_wealth))
        ax.plot(steps, cumulative_wealth, label=model_name, color=color, 
                linestyle=linestyle, linewidth=1.8, alpha=0.9)
        
    except Exception as e:
        print(f"Error loading {model_key}: {e}")

ax.axhline(y=100, color="black", linestyle="--", linewidth=1.5, alpha=0.6, label="Initial Capital")

ax.set_xlabel("Trading Step", fontsize=12)
ax.set_ylabel("Portfolio Value ($)", fontsize=12)
ax.set_title("Cumulative Return Trajectories Demonstrating Capital Bleed\nAcross Sequence Architectures", 
             fontsize=13, fontweight="bold", pad=10)

ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
ax.set_xlim(0, None)
ax.set_ylim(bottom=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("capital_bleed_trajectories.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print("Saved: capital_bleed_trajectories.png")