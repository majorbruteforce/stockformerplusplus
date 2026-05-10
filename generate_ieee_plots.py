#!/usr/bin/env python
"""
IEEE-Quality Publication Figures for Stockformer++ Ablation Study
All figures use exact experimental data from walk-forward validation (20 folds).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from matplotlib.ticker import MaxNLocator

# IEEE Style Settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

RESULT_DIR = "results"
FINAL_DIR = "final_results"

# ==============================================================================
# LOAD DATA
# ==============================================================================

# Load all model results
results = {
    "Transformer (MSE)": {
        "MAE": 0.1718,
        "Sharpe": 0.309,
        "Cum_Return": 24.3,
        "MDD": -77.2,
        "Dir_Acc": 50.8,
        "Profit_Factor": 1.06,
        "RMSE": 0.2518,
        "R2": -60.29,
    },
    "Time2Vec (MSE)": {
        "MAE": 0.1795,
        "Sharpe": 0.242,
        "Cum_Return": -5.4,
        "MDD": -74.7,
        "Dir_Acc": 50.4,
        "Profit_Factor": 1.04,
        "RMSE": 0.2429,
        "R2": -56.02,
    },
    "Gating (MSE)": {
        "MAE": 0.1983,
        "Sharpe": -0.299,
        "Cum_Return": -89.6,
        "MDD": -94.0,
        "Dir_Acc": 50.0,
        "Profit_Factor": 0.95,
        "RMSE": 0.2479,
        "R2": -58.41,
    },
    "RNN (MSE)": {
        "MAE": 0.2762,
        "Sharpe": -0.196,
        "Cum_Return": -84.1,
        "MDD": -91.7,
        "Dir_Acc": 49.6,
        "Profit_Factor": 0.97,
        "RMSE": 0.3535,
        "R2": -119.82,
    },
    "LSTM (MSE)": {
        "MAE": 0.0348,
        "Sharpe": -0.408,
        "Cum_Return": -93.2,
        "MDD": -94.2,
        "Dir_Acc": 48.5,
        "Profit_Factor": 0.93,
        "RMSE": 0.0465,
        "R2": -1.09,
    },
    "Transformer (Sharpe)": {
        "MAE": 2.5009,
        "Sharpe": 0.041,
        "Cum_Return": -58.3,
        "MDD": -76.7,
        "Dir_Acc": 49.8,
        "Profit_Factor": 1.01,
        "RMSE": 2.9637,
        "R2": -8489.06,
    },
}

# Model short names for plots
short_names = {
    "Transformer (MSE)": "Transformer",
    "Time2Vec (MSE)": "Time2Vec",
    "Gating (MSE)": "Gating",
    "RNN (MSE)": "RNN",
    "LSTM (MSE)": "LSTM",
    "Transformer (Sharpe)": "Transformer (Sharpe)",
}

# Colors (grayscale-safe)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# Load fold-wise data for model 1 (Transformer)
with open(f"{RESULT_DIR}/model1_transformer.json", "r") as f:
    transformer_folds = json.load(f)
    fold_results = transformer_folds["fold_results"]

# ==============================================================================
# FIGURE 1: Prediction Quality vs Trading Quality (MAE vs Sharpe)
# ==============================================================================
print("Generating Figure 1: MAE vs Sharpe...")

fig, ax = plt.subplots(figsize=(5, 3.5))

for idx, (name, metrics) in enumerate(results.items()):
    ax.scatter(
        metrics["MAE"],
        metrics["Sharpe"],
        s=80,
        c=colors[idx],
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )

    # Direct labeling with offset to avoid overlap
    offset_x = 0.02
    offset_y = 0.08
    if name == "LSTM (MSE)":
        offset_y = -0.15
    elif name == "Transformer (Sharpe)":
        offset_x = -0.3
        offset_y = 0.12
    elif name == "Gating (MSE)":
        offset_y = -0.12

    ax.annotate(
        short_names[name],
        (metrics["MAE"], metrics["Sharpe"]),
        xytext=(offset_x, offset_y),
        textcoords="offset points",
        fontsize=8,
        ha="center",
    )

# Zero Sharpe line
ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

ax.set_xlabel("Mean Absolute Error (MAE)")
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Prediction Error vs Trading Performance", fontweight="bold", pad=8)

# Add caption
ax.text(
    0.5,
    -0.18,
    "Lower prediction error does not necessarily imply superior trading profitability.",
    transform=ax.transAxes,
    fontsize=8,
    ha="center",
    style="italic",
)

ax.set_xlim(-0.1, 3.0)
ax.set_ylim(-0.6, 0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig(
    f"{FINAL_DIR}/figures/prediction_vs_trading_quality_ieee.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"{FINAL_DIR}/figures/prediction_vs_trading_quality_ieee.pdf", bbox_inches="tight"
)
plt.close()

# ==============================================================================
# FIGURE 2: Equity Curves Comparison
# ==============================================================================
print("Generating Figure 2: Equity Curves...")

fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), gridspec_kw={"width_ratios": [2, 1]})

# Left: Equity curves (schematic - actual data not fully available)
ax = axes[0]

# Since we don't have continuous equity data, show cumulative returns as bar chart
model_names_short = list(short_names.values())
cum_returns = [results[n]["Cum_Return"] for n in results.keys()]
bar_colors = ["#2ca02c" if r > 0 else "#d62728" for r in cum_returns]

bars = ax.barh(
    model_names_short, cum_returns, color=bar_colors, edgecolor="black", linewidth=0.5
)
ax.axvline(x=0, color="black", linewidth=1)
ax.set_xlabel("Cumulative Return (%)")
ax.set_title("Portfolio Performance", fontweight="bold")
ax.set_xlim(-100, 30)

# Add value labels
for bar, val in zip(bars, cum_returns):
    x_pos = val + 3 if val > 0 else val - 3
    ax.text(
        x_pos,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center",
        ha="left" if val > 0 else "right",
        fontsize=8,
    )

ax.grid(True, alpha=0.3, axis="x")

# Right: Summary table
ax2 = axes[1]
ax2.axis("off")

# Create mini table
table_data = []
for name in results.keys():
    m = results[name]
    table_data.append(
        [short_names[name][:12], f"{m['Sharpe']:.2f}", f"{abs(m['MDD']):.1f}%"]
    )

table = ax2.table(
    cellText=table_data,
    colLabels=["Model", "Sharpe", "MDD"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(0.9, 1.3)

# Style
for j in range(3):
    table[(0, j)].set_facecolor("#404040")
    table[(0, j)].set_text_props(color="white", fontweight="bold")

ax2.set_title("Summary", fontweight="bold", pad=10)

plt.tight_layout()
plt.savefig(f"{FINAL_DIR}/figures/equity_curves_ieee.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{FINAL_DIR}/figures/equity_curves_ieee.pdf", bbox_inches="tight")
plt.close()

# ==============================================================================
# FIGURE 3: Ablation Study Comparison (Grouped Bar Charts)
# ==============================================================================
print("Generating Figure 3: Ablation Comparison...")

fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))
axes = axes.flatten()

metrics_to_plot = [
    ("MAE", "MAE (lower=better)"),
    ("Dir_Acc", "Direction Accuracy (%)"),
    ("Sharpe", "Sharpe Ratio"),
    ("Cum_Return", "Cumulative Return (%)"),
    ("MDD", "Max Drawdown (%)"),
    ("Profit_Factor", "Profit Factor"),
]

for ax, (metric, title) in zip(axes, metrics_to_plot):
    values = [results[n][metric] for n in results.keys()]

    # Handle MDD (show as positive)
    if metric == "MDD":
        values = [abs(v) for v in values]

    bars = ax.bar(
        range(len(values)), values, color=colors, edgecolor="black", linewidth=0.5
    )

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(
        [short_names[n][:8] for n in results.keys()],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_title(title, fontweight="bold", pad=5)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on top
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if metric == "Sharpe":
            label = f"{val:.2f}"
        elif metric in ["MAE", "Profit_Factor"]:
            label = f"{val:.2f}"
        else:
            label = f"{val:.1f}"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

plt.tight_layout()
plt.savefig(
    f"{FINAL_DIR}/figures/ablation_study_ieee.png", dpi=300, bbox_inches="tight"
)
plt.savefig(f"{FINAL_DIR}/figures/ablation_study_ieee.pdf", bbox_inches="tight")
plt.close()

# ==============================================================================
# FIGURE 4: Fold-wise Analysis
# ==============================================================================
print("Generating Figure 4: Fold-wise Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))

folds = range(1, len(fold_results) + 1)

# 1. Direction Accuracy
ax = axes[0]
dir_accs = [f["Dir_Acc"] for f in fold_results]
ax.bar(folds, dir_accs, color="steelblue", edgecolor="black", linewidth=0.5, alpha=0.8)
ax.axhline(y=50, color="red", linestyle="--", linewidth=1.5, label="Random (50%)")
ax.set_xlabel("Fold")
ax.set_ylabel("Direction Accuracy (%)")
ax.set_title("Direction Accuracy", fontweight="bold")
ax.set_ylim(35, 65)
ax.legend(fontsize=7, loc="upper right")
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax.grid(True, alpha=0.3, axis="y")

# 2. Sharpe Ratio
ax = axes[1]
sharpes = [f["Sharpe"] for f in fold_results]
bar_colors = ["#2ca02c" if s > 0 else "#d62728" for s in sharpes]
ax.bar(folds, sharpes, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.8)
ax.axhline(y=0, color="black", linewidth=1)
ax.set_xlabel("Fold")
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Sharpe Ratio", fontweight="bold")
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax.grid(True, alpha=0.3, axis="y")

# 3. Cumulative Returns with stats box
ax = axes[2]
returns = [f["Cum_Return"] for f in fold_results]
bar_colors = ["#2ca02c" if r > 0 else "#d62728" for r in returns]
ax.bar(folds, returns, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.8)
ax.axhline(y=0, color="black", linewidth=1)
ax.set_xlabel("Fold")
ax.set_ylabel("Return (%)")
ax.set_title("Cumulative Return", fontweight="bold")
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax.grid(True, alpha=0.3, axis="y")

# Statistics box
stats_text = f"Mean: {np.mean(returns):.1f}%\nStd: {np.std(returns):.1f}%\nPos: {sum(1 for r in returns if r > 0)}/{len(returns)}"
ax.text(
    0.95,
    0.95,
    stats_text,
    transform=ax.transAxes,
    fontsize=7,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

plt.tight_layout()
plt.savefig(f"{FINAL_DIR}/figures/fold_analysis_ieee.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{FINAL_DIR}/figures/fold_analysis_ieee.pdf", bbox_inches="tight")
plt.close()

# ==============================================================================
# FIGURE 5: Walk-Forward Validation Diagram
# ==============================================================================
print("Generating Figure 5: Walk-Forward Diagram...")

fig, ax = plt.subplots(figsize=(6, 1.5))
ax.axis("off")

# Draw timeline
n_folds = 5  # Show 5 folds for illustration
fold_width = 0.8
gap = 0.3

for i in range(n_folds):
    x_start = i * (fold_width + gap)

    # Train bar (gray)
    rect_train = plt.Rectangle(
        (x_start, 0.3),
        fold_width * 0.6,
        0.2,
        facecolor="#cccccc",
        edgecolor="black",
        linewidth=1,
    )
    ax.add_patch(rect_train)
    ax.text(
        x_start + fold_width * 0.3, 0.4, "Train", ha="center", va="center", fontsize=8
    )

    # Test bar (white with border)
    rect_test = plt.Rectangle(
        (x_start + fold_width * 0.65, 0.3),
        fold_width * 0.35,
        0.2,
        facecolor="white",
        edgecolor="black",
        linewidth=1,
    )
    ax.add_patch(rect_test)
    ax.text(
        x_start + fold_width * 0.82, 0.4, "Test", ha="center", va="center", fontsize=8
    )

# Arrow showing time progression
ax.annotate(
    "",
    xy=(n_folds * (fold_width + gap) - 0.3, 0.4),
    xytext=(0, 0.4),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
)

# Labels
ax.text(0, 0.15, "Time", fontsize=10, ha="left")
ax.text(n_folds * (fold_width + gap) - 0.3, 0.15, "→", fontsize=12)

# Fold labels
for i in range(n_folds):
    x_center = i * (fold_width + gap) + fold_width * 0.5
    ax.text(x_center, 0.05, f"Fold {i + 1}", ha="center", va="center", fontsize=7)

ax.set_xlim(-0.2, n_folds * (fold_width + gap) + 0.2)
ax.set_ylim(-0.1, 0.6)

# Title
ax.set_title(
    "Walk-Forward Validation: Rolling Window Strategy", fontweight="bold", pad=10
)

plt.tight_layout()
plt.savefig(
    f"{FINAL_DIR}/figures/walkforward_pipeline_ieee.png", dpi=300, bbox_inches="tight"
)
plt.savefig(f"{FINAL_DIR}/figures/walkforward_pipeline_ieee.pdf", bbox_inches="tight")
plt.close()

print("\n✓ All 5 figures generated successfully!")
print(f"  - Figure 1: prediction_vs_trading_quality_ieee.png/pdf")
print(f"  - Figure 2: equity_curves_ieee.png/pdf")
print(f"  - Figure 3: ablation_study_ieee.png/pdf")
print(f"  - Figure 4: fold_analysis_ieee.png/pdf")
print(f"  - Figure 5: walkforward_pipeline_ieee.png/pdf")
print(f"\nSaved to: {FINAL_DIR}/figures/")
