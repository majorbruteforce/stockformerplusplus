#!/usr/bin/env python
"""Generate all publication-ready visualizations for ablation results."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

RESULT_DIR = "results"
FINAL_DIR = "final_results"

# Load results
with open(f"{RESULT_DIR}/complete_ablation_results.json", "r") as f:
    results = json.load(f)

# Load individual model predictions
preds_dict = {}
targs_dict = {}
for i in range(1, 7):
    try:
        preds_dict[f"model{i}"] = np.load(f"{RESULT_DIR}/model{i}_preds.npy")
        targs_dict[f"model{i}"] = np.load(f"{RESULT_DIR}/model{i}_targs.npy")
    except:
        pass

model_names = {
    "Transformer (MSE)": "Transformer (MSE)",
    "Time2Vec (MSE)": "Time2Vec (MSE)",
    "Gating (MSE)": "Gating (MSE)",
    "RNN (MSE)": "RNN (MSE)",
    "LSTM (MSE)": "LSTM (MSE)",
    "Transformer (Sharpe)": "Transformer (Sharpe)",
}

# 1. Bar chart comparison - all metrics
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

metrics = ["MAE", "Dir_Acc", "Sharpe", "Cum_Return", "MDD", "Profit_Factor"]
titles = [
    "MAE (lower=better)",
    "Direction Accuracy (%)",
    "Sharpe Ratio (higher=better)",
    "Cumulative Return (%)",
    "Max Drawdown (%, lower=better)",
    "Profit Factor (higher=better)",
]

colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

for ax, metric, title in zip(axes, metrics, titles):
    names = list(results.keys())
    values = [results[n][metric] for n in names]

    # Handle negative values for coloring
    if metric == "MDD":
        values = [abs(v) for v in values]

    bars = ax.bar(
        range(len(names)), values, color=colors, edgecolor="black", linewidth=0.5
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [n.replace(" (MSE)", "").replace(" (Sharpe)", "\n(Sharpe)") for n in names],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

plt.tight_layout()
plt.savefig(
    f"{FINAL_DIR}/figures/ablation_comparison.png", dpi=150, bbox_inches="tight"
)
plt.close()
print("Saved: ablation_comparison.png")

# 2. Equity curves
fig, ax = plt.subplots(figsize=(12, 6))

for idx, (name, metrics) in enumerate(results.items()):
    # Recreate equity curve from stored results - we need the actual predictions
    pass

# Since we don't have continuous predictions stored, create a summary visualization
ax.axis("off")
ax.set_title("Equity Curves - See Individual Model Files", fontsize=14)

# Create a text summary instead
summary_text = "EQUITY CURVES SUMMARY\n\n"
for name, m in results.items():
    summary_text += f"{name}:\n"
    summary_text += f"  Return: {m['Cum_Return']:+.1f}%\n"
    summary_text += f"  Sharpe: {m['Sharpe']:.2f}\n"
    summary_text += f"  MDD: {m['MDD']:.1f}%\n\n"

ax.text(
    0.1,
    0.9,
    summary_text,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.savefig(f"{FINAL_DIR}/figures/equity_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: equity_summary.png")

# 3. Scatter plot: MAE vs Sharpe (prediction quality vs trading quality)
fig, ax = plt.subplots(figsize=(10, 6))

for name, m in results.items():
    ax.scatter(
        m["MAE"], m["Sharpe"], s=200, label=name, edgecolor="black", linewidth=1.5
    )
    ax.annotate(
        name.replace(" (MSE)", "").replace(" (Sharpe)", "\n(Sharpe)"),
        (m["MAE"], m["Sharpe"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.set_xlabel("MAE (Prediction Error, lower=better)", fontsize=12)
ax.set_ylabel("Sharpe Ratio (Trading Performance, higher=better)", fontsize=12)
ax.set_title("Prediction Quality vs Trading Quality", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FINAL_DIR}/figures/mae_vs_sharpe.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: mae_vs_sharpe.png")

# 4. Table as image
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

# Create table
cols = ["Model", "MAE", "RMSE", "R²", "Dir%", "Sharpe", "Return%", "MDD%", "PF"]
cell_text = []
for name, m in results.items():
    row = [
        name,
        f"{m['MAE']:.4f}",
        f"{m['RMSE']:.4f}",
        f"{m['R2']:.1f}",
        f"{m['Dir_Acc']:.1f}",
        f"{m['Sharpe']:.2f}",
        f"{m['Cum_Return']:+.1f}",
        f"{m['MDD']:.1f}",
        f"{m['Profit_Factor']:.2f}",
    ]
    cell_text.append(row)

table = ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Style header
for j in range(len(cols)):
    table[(0, j)].set_facecolor("#2E86AB")
    table[(0, j)].set_text_props(color="white", fontweight="bold")

# Style rows
for i in range(1, len(results) + 1):
    for j in range(len(cols)):
        if i == 1:
            table[(i, j)].set_facecolor("#E8F4F8")
        elif i % 2 == 0:
            table[(i, j)].set_facecolor("#F5F5F5")
        else:
            table[(i, j)].set_facecolor("#FFFFFF")

# Highlight best in green
best_sharpe_idx = (
    max(range(len(results)), key=lambda i: list(results.values())[i]["Sharpe"]) + 1
)
for j in range(len(cols)):
    table[(best_sharpe_idx, j)].set_facecolor("#90EE90")

ax.set_title(
    "Ablation Study Results - Full Comparison", fontsize=14, fontweight="bold", pad=20
)

plt.tight_layout()
plt.savefig(f"{FINAL_DIR}/figures/results_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results_table.png")

# 5. Fold-wise performance visualization (using model1 as example)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Load fold results
with open(f"{RESULT_DIR}/model1_transformer.json", "r") as f:
    fold_data = json.load(f)

fold_results = fold_data["fold_results"]
n_folds = len(fold_results)
folds = range(1, n_folds + 1)

# Direction accuracy
ax = axes[0, 0]
dir_accs = [f["Dir_Acc"] for f in fold_results]
ax.bar(folds, dir_accs, color="steelblue", edgecolor="black", alpha=0.7)
ax.axhline(y=50, color="red", linestyle="--", linewidth=2, label="Random (50%)")
ax.set_xlabel("Fold")
ax.set_ylabel("Direction Accuracy (%)")
ax.set_title("Fold-wise Direction Accuracy", fontweight="bold")
ax.set_ylim(30, 70)
ax.legend()
ax.grid(True, alpha=0.3)

# Sharpe ratio
ax = axes[0, 1]
sharpes = [f["Sharpe"] for f in fold_results]
colors = ["green" if s > 0 else "red" for s in sharpes]
ax.bar(folds, sharpes, color=colors, edgecolor="black", alpha=0.7)
ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
ax.set_xlabel("Fold")
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Fold-wise Sharpe Ratio", fontweight="bold")
ax.grid(True, alpha=0.3)

# Cumulative returns
ax = axes[1, 0]
returns = [f["Cum_Return"] for f in fold_results]
colors = ["green" if r > 0 else "red" for r in returns]
ax.bar(folds, returns, color=colors, edgecolor="black", alpha=0.7)
ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
ax.set_xlabel("Fold")
ax.set_ylabel("Cumulative Return (%)")
ax.set_title("Fold-wise Returns", fontweight="bold")
ax.grid(True, alpha=0.3)

# Statistics summary
ax = axes[1, 1]
ax.axis("off")
stats_text = f"""Transformer (MSE) Fold Statistics:

Direction Accuracy:
  Mean: {np.mean(dir_accs):.1f}%
  Std:  {np.std(dir_accs):.1f}%
  Min:  {np.min(dir_accs):.1f}%
  Max:  {np.max(dir_accs):.1f}%

Sharpe Ratio:
  Mean: {np.mean(sharpes):.2f}
  Std:  {np.std(sharpes):.2f}
  Pos:  {sum(1 for s in sharpes if s > 0)}/{n_folds} folds

Returns:
  Mean: {np.mean(returns):.1f}%
  Std:  {np.std(returns):.1f}%
  Pos:  {sum(1 for r in returns if r > 0)}/{n_folds} folds
"""
ax.text(
    0.1,
    0.9,
    stats_text,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.suptitle(
    "Fold-wise Performance Analysis (Transformer)", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(f"{FINAL_DIR}/figures/fold_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fold_analysis.png")

print("\n✓ All visualizations saved to final_results/figures/")
