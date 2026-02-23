# Stockformer+

> **Note:** This project is currently in progress. Experimentation, validation, and documentation are ongoing. Results presented herein are preliminary and subject to change.

A PyTorch-based framework for financial time series prediction, benchmarking transformer architectures against traditional recurrent models on stock market data.

## Overview

Stockformer++ compares four neural network architectures for stock price prediction:

| Model | Type | Description |
|-------|------|-------------|
| **RNN** | Recurrent | Basic recurrent neural network |
| **LSTM** | Recurrent | Long Short-Term Memory network with gating mechanisms |
| **Stockformer** | Transformer | Transformer encoder with sinusoidal positional encoding |
| **Time2Vec Transformer** | Hybrid | Transformer augmented with Time2Vec temporal representations |

## Architecture

### Stockformer

```
Input → Linear Projection → Positional Encoding → Transformer Encoder → FC Output
```

- **Positional Encoding**: Sinusoidal encoding (Vaswani et al., 2017)
- **Encoder Layer**: Multi-head self-attention with GELU activation
- **Output**: Last timestep representation projected to prediction horizon

### Time2Vec Transformer

Extends Stockformer with learnable periodic temporal embeddings (Kazemnejad et al., 2020), capturing both linear and periodic time patterns.

## Project Structure

```
stockformer++/
├── config.py                      # Configuration (hyperparameters, paths)
├── main.py                        # Main benchmark execution
├── requirements.txt               # Dependencies
├── data/
│   └── fetcher.py                # Data loading utilities
├── features/
│   └── engineer.py               # Feature engineering
├── models/
│   ├── stockformer.py            # Stockformer implementation
│   ├── time2vec_transformer.py   # Time2Vec Transformer
│   └── rnn_lstm.py               # RNN and LSTM baselines
├── utils/
│   ├── dataset.py                # PyTorch dataset
│   ├── training.py               # Training loop
│   ├── metrics.py                # Evaluation metrics
│   └── plotting.py               # Visualization
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- NumPy, Pandas, scikit-learn
- Matplotlib, tqdm

## Usage

Run the full benchmark:

```bash
python main.py
```

Configure parameters in `config.py`:
- `FEATURE_CONFIG`: Sequence length, prediction horizons
- `MODEL_CONFIG`: Model-specific hyperparameters
- `TRAIN_CONFIG`: Learning rate, epochs, batch size

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |
| Dir Acc | Direction Accuracy (% of correct trend predictions) |
| Sharpe | Sharpe Ratio of returns |
| MDD | Maximum Drawdown |
| Cum Return | Cumulative Return |

## Status

### Completed
- [x] Model implementations (RNN, LSTM, Stockformer, Time2Vec Transformer)
- [x] Data pipeline and feature engineering
- [x] Training infrastructure with early stopping
- [x] Evaluation framework with comprehensive metrics

### In Progress
- [ ] Hyperparameter optimization
- [ ] Extended validation on multiple datasets
- [ ] Ablation studies
- [ ] Performance optimization

### Planned
- [ ] Published results and final model selection
- [ ] Extended documentation
- [ ] Reproducibility package

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Kazemnejad, M., et al. (2020). "Time2Vec: Learning a Vector Representation of Time." *ICML*.
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*.

*This project is developed for research and educational purposes. Stock market predictions are inherently uncertain; past performance does not guarantee future results.*
