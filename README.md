# The Effect of Label Noise on Generalisation

A Theoretical and Empirical Analysis for Classifiers

## Overview

This project studies how **symmetric and asymmetric label noise** affects the generalisation performance of three classifiers — Logistic Regression, Decision Tree, and a 2-layer MLP — and demonstrates that noise-aware correction methods can recover much of the lost accuracy.

## Project Structure

```
label_noise_project/
├── config.py                 # Global configuration
├── requirements.txt          # Dependencies
├── theory.md                 # Theoretical background
├── run_experiments.py        # Main experiment orchestrator
├── run_correction.py         # Noise correction experiments
├── src/
│   ├── datasets.py           # MNIST, CIFAR-10, UCI Adult loaders
│   ├── noise.py              # Symmetric & asymmetric noise injection
│   ├── models.py             # LogReg, DecisionTree, PyTorch MLP
│   ├── losses.py             # Backward correction, GCE losses
│   ├── transition_matrix.py  # Anchor-point T estimation
│   ├── train.py              # Training loops
│   ├── evaluate.py           # Accuracy, F1, ECE, confusion matrix
│   └── plotting.py           # Visualisation utilities
├── data/                     # Auto-downloaded datasets
└── results/                  # Plots, tables, metrics
    ├── plots/
    └── tables/
```

## Installation

```bash
cd label_noise_project
pip install -r requirements.txt
```

**Dependencies**: numpy, scikit-learn, torch, torchvision, matplotlib, seaborn, pandas, tqdm

## Usage

### 1. Run Full Experiments

```bash
# Full grid: all datasets × noise types × noise rates × models
python run_experiments.py

# Quick test: MNIST only, two models
python run_experiments.py --datasets mnist --models logistic mlp

# Custom noise rates
python run_experiments.py --datasets mnist adult --noise-rates 0.0 0.1 0.3
```

### 2. Run Noise Correction Experiments

```bash
# Default: MNIST, symmetric noise, η=0.3
python run_correction.py

# Custom settings
python run_correction.py --dataset cifar10 --noise-rate 0.5 --noise-type asymmetric
```

### 3. Outputs

All results are saved to `results/`:
- `results/plots/` — Accuracy vs noise, confusion matrices, transition matrix heatmaps
- `results/tables/` — CSV and LaTeX summary tables
- `results/all_results.json` — Full metrics in JSON format

## Key Findings (Expected)

1. **Decision Trees** degrade most under noise (high variance, memorise noisy labels).
2. **Logistic Regression** is naturally more robust (limited capacity acts as regulariser).
3. **MLP** performance depends on regularisation and loss function.
4. **Backward correction** with estimated T̂ recovers significant accuracy.
5. **GCE** provides noise robustness without requiring T estimation.

## Theory

See [theory.md](theory.md) for formal derivations of:
- Risk decomposition under label noise
- Conditions for noise-tolerant losses
- Backward correction proof
- Anchor-point estimation method

## Configuration

Edit `config.py` to modify:
- Random seed
- MLP hyperparameters (hidden dim, dropout, epochs, learning rate)
- Dataset subsampling (e.g., reduce CIFAR-10 for faster CPU runs)
- Noise rates and model selection
