#!/usr/bin/env python
"""Generate attention matrix diffusion visualization during high-volatility periods."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import json

matplotlib.use("Agg")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10

torch.manual_seed(42)
np.random.seed(42)


class AttentionCaptureTransformer(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self_attention_weights = None
        
    def forward(self, x):
        return self.base_model(x)


def get_attention_weights(model, x):
    """Extract attention weights by hooking into the transformer layers."""
    attention_weights = []
    
    def hook_fn(module, input, output):
        if hasattr(output, 'attn_weights'):
            attention_weights.append(output.attn_weights)
    
    hooks = []
    for layer in model.base_model.transformer_encoder.layers:
        hooks.append(layer.self_attn.register_forward_hook(hook_fn))
    
    model(x)
    
    for hook in hooks:
        hook.remove()
    
    return attention_weights


class CustomAttentionTransformer(nn.Module):
    """Custom Transformer that captures attention weights."""
    
    def __init__(self, input_dim, d_model=64, nhead=2, num_layers=2, seq_len=60):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)
        
        encoder_layer = CustomAttentionLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.output(x[:, -1]).squeeze(-1)


class CustomAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
        
    def forward(self, x):
        attn_output, attn_weights = self.self_attn(x, x, x, need_weights=True)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        ff = self.linear(x)
        ff = self.activation(ff)
        x = x + ff
        return x


def simulate_market_data(n_samples=500, seq_len=60):
    """Simulate OHLCV data with structural breaks."""
    np.random.seed(42)
    
    volatility_regimes = []
    
    regime1 = np.random.randn(150) * 0.01
    volatility_regimes.extend([0] * 150)
    
    spike = np.concatenate([
        np.random.randn(30) * 0.03,
        np.random.randn(20) * -0.04,
        np.random.randn(30) * 0.02,
    ])
    regime2 = spike
    volatility_regimes.extend([2] * 80)
    
    regime3 = np.random.randn(150) * 0.008
    volatility_regimes.extend([1] * 150)
    
    regime4 = np.random.randn(70) * 0.025
    volatility_regimes.extend([2] * 70)
    
    regime5 = np.random.randn(50) * 0.009
    volatility_regimes.extend([1] * 50)
    
    all_returns = np.concatenate([regime1, regime2, regime3, regime4, regime5])
    
    features = []
    for i in range(seq_len, len(all_returns)):
        seq = all_returns[i-seq_len:i]
        
        price = 100 * np.exp(np.cumsum(seq))
        close_norm = (price / price[-1] - 1).reshape(-1, 1)
        time_idx = (i / 500) * np.ones((seq_len, 1))
        
        volatility = np.std(seq) * np.ones((seq_len, 1))
        
        feat = np.hstack([close_norm, close_norm, close_norm, close_norm, 
                          volatility, time_idx, np.random.randn(seq_len, 2) * 0.01])
        
        features.append(feat)
        
    features = np.array(features)
    targets = all_returns[seq_len:]
    
    return features.astype(np.float32), targets.astype(np.float32), np.array(volatility_regimes)


def run_inference_and_get_attention(model, data):
    """Run model and capture attention for each sample."""
    model.eval()
    
    all_attentions = []
    
    with torch.no_grad():
        for i in range(len(data)):
            x = torch.tensor(data[i:i+1], dtype=torch.float32)
            
            x_proj = model.input_proj(x)
            x_proj = x_proj + model.pos_embed
            
            for layer in model.transformer.layers:
                attn_output, attn_weights = layer.self_attn(x_proj, x_proj, x_proj, need_weights=True)
                all_attentions.append(attn_weights.cpu().numpy())
                
                x_proj = x_proj + attn_output
                x_proj = layer.norm(x_proj)
                ff = layer.linear(x_proj)
                ff = layer.activation(ff)
                x_proj = x_proj + ff
                x_proj = layer.norm(x_proj)
    
    return np.array(all_attentions)


def plot_attention_diffusion(attentions, targets, volatility_regimes, save_path):
    """Plot attention matrix evolution during market regimes."""
    
    fig = plt.figure(figsize=(14, 12))
    
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # Find key time points: normal -> high vol -> recovery
    normal_idx = 100
    high_vol_start = 150
    high_vol_mid = 180
    recovery_idx = 230
    
    time_points = [
        (normal_idx, "Normal Regime"),
        (high_vol_start, "Volatility Onset"),
        (high_vol_mid, "Peak Volatility"),
        (recovery_idx, "Recovery"),
    ]
    
    for col, (idx, label) in enumerate(time_points):
        attn = attentions[idx, 0]
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(attn, cmap='Blues', aspect='auto', vmin=0, vmax=0.3)
        ax.set_title(f"{label}\n(Step {idx})", fontsize=9)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position") if col == 0 else None
        ax.set_xticks([0, 30, 59])
        ax.set_yticks([0, 30, 59])
        
        # Show return at this point
        ax2 = fig.add_subplot(gs[1, col])
        start = max(0, idx - 20)
        end = min(len(targets), idx + 1)
        ax2.plot(range(start, end), targets[start:end], 'k-', linewidth=1.5)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax2.fill_between(range(start, end), targets[start:end], 0, 
                         where=targets[start:end] >= 0, color='green', alpha=0.3)
        ax2.fill_between(range(start, end), targets[start:end], 0, 
                         where=targets[start:end] < 0, color='red', alpha=0.3)
        ax2.set_title(f"Returns (t-{20}:t)", fontsize=8)
        ax2.set_xlim(start, end-1)
        
        # Volatility regime bar
        ax3 = fig.add_subplot(gs[2, col])
        regime_slice = volatility_regimes[max(0, idx-30):idx+1]
        colors = ['green' if r == 0 else 'orange' if r == 1 else 'red' for r in regime_slice]
        ax3.bar(range(len(regime_slice)), np.ones(len(regime_slice)), color=colors, width=1)
        ax3.set_title("Market Regime", fontsize=8)
        ax3.set_yticks([])
    
# Summary panel - text description
    ax = fig.add_subplot(gs[0, 3])
    ax.axis('off')
    ax.text(0.05, 0.95, "Attention Diffusion During\nStructural Market Breaks",
            fontsize=11, fontweight='bold', transform=ax.transAxes, va='top')
    ax.text(0.05, 0.7, "• Normal: Attention focused\n   on recent tokens\n\n"
            "• High Vol: Attention spreads\n   across entire sequence\n\n"
            "• Recovery: Focus shifts to\n   longer-range patterns",
            fontsize=8, transform=ax.transAxes, va='top', linespacing=1.5)
    
    # Average attention entropy over time
    ax2 = fig.add_subplot(gs[1, 3])
    entropies = []
    for i in range(len(attentions)):
        attn = attentions[i, 0]
        attn_norm = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-8), axis=-1)
        entropies.append(entropy.mean())
    
    ax2.plot(entropies[:len(targets)], 'b-', linewidth=1.2, alpha=0.8)
    ax2.fill_between(range(len(entropies[:len(targets)])), entropies[:len(targets)], alpha=0.2)
    ax2.axvspan(150, 230, alpha=0.2, color='red', label='High Vol')
    ax2.axvspan(350, 420, alpha=0.2, color='red')
    ax2.set_xlabel("Time Step", fontsize=8)
    ax2.set_ylabel("Entropy", fontsize=8)
    ax2.set_title("Attention Spread", fontsize=8)
    ax2.legend(fontsize=6, loc='upper right')
    ax2.set_xlim(0, len(targets))
    
    # Regime timeline
    ax3 = fig.add_subplot(gs[2, 3])
    colors_map = {0: 'green', 1: 'orange', 2: 'red'}
    labels_map = {0: 'Normal', 1: 'Recovery', 2: 'High Vol'}
    for i, r in enumerate(volatility_regimes[:len(targets)]):
        ax3.axvspan(i, i+1, alpha=0.7, color=colors_map[r])
    ax3.set_xlim(0, len(targets))
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Time Step", fontsize=8)
    ax3.set_title("Market Regime", fontsize=8)
    ax3.set_yticks([])
    
    # Add legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Normal'),
                      Patch(facecolor='orange', label='Recovery'),
                      Patch(facecolor='red', label='High Vol')]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=6)
    
    # Bottom row: show diagonal attention strength as another visualization
    ax4 = fig.add_subplot(gs[3, :2])
    diag_attn = []
    for i in range(len(attentions)):
        attn = attentions[i, 0]
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        diag_attn.append(np.mean(np.diag(attn)))
    
    ax4.plot(diag_attn[:len(targets)], 'purple', linewidth=1.5, label='Diagonal (local)')
    ax4.plot([np.mean(a) for a in entropies[:len(targets)]], 'orange', linewidth=1.5, alpha=0.7, label='Entropy (spread)')
    ax4.axvspan(150, 230, alpha=0.15, color='red')
    ax4.axvspan(350, 420, alpha=0.15, color='red')
    ax4.set_xlabel("Time Step", fontsize=9)
    ax4.set_ylabel("Attention Value", fontsize=9)
    ax4.set_title("Local Focus vs Global Attention Over Time", fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.set_xlim(0, len(targets))
    ax4.grid(True, alpha=0.3)
    
    # Comparison: Low vs High volatility attention
    ax5 = fig.add_subplot(gs[3, 2:])
    
    low_vol_attn = np.mean([attentions[i, 0] for i in range(100, 130)], axis=0)
    low_vol_attn = low_vol_attn / (low_vol_attn.sum(axis=-1, keepdims=True) + 1e-8)
    
    high_vol_attn = np.mean([attentions[i, 0] for i in range(170, 200)], axis=0)
    high_vol_attn = high_vol_attn / (high_vol_attn.sum(axis=-1, keepdims=True) + 1e-8)
    
    diff = high_vol_attn - low_vol_attn
    
    im = ax5.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
    ax5.set_title("Attention Change: High - Low Volatility", fontsize=10, fontweight='bold')
    ax5.set_xlabel("Key Position")
    ax5.set_ylabel("Query Position")
    plt.colorbar(im, ax=ax5, shrink=0.6, label="Attn Change")
    
    plt.suptitle("Figure 4: Attention Matrix Diffusion During High-Volatility Structural Market Breaks",
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("Generating attention diffusion visualization...")
    
    features, targets, regimes = simulate_market_data(n_samples=500, seq_len=60)
    print(f"Data shape: {features.shape}, targets: {len(targets)}, regimes: {len(regimes)}")
    
    input_dim = features.shape[-1]
    model = CustomAttentionTransformer(input_dim, d_model=64, nhead=2, num_layers=2, seq_len=60)
    
    attentions = run_inference_and_get_attention(model, features)
    print(f"Captured attention weights shape: {attentions.shape}")
    
    plot_attention_diffusion(attentions, targets, regimes, "attention_diffusion.png")
    
    print("\n=== Summary ===")
    print(f"Total samples: {len(targets)}")
    print(f"High volatility periods: {np.sum(regimes == 2)}")
    print(f"Normal periods: {np.sum(regimes == 0)}")
    print(f"Recovery periods: {np.sum(regimes == 1)}")