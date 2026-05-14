"""
Global configuration for label-noise robustness experiments.

All hyperparameters, paths, and experimental settings are centralised here
so that every module imports a single source of truth.
"""

import os
import torch
import numpy as np
import random

# ──────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Ensure output directories exist
for _dir in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, TABLES_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ──────────────────────────────────────────────
#  Experimental grid
# ──────────────────────────────────────────────
DATASETS = ["mnist", "cifar10", "adult"]
MODELS = ["logistic", "tree", "mlp"]
NOISE_TYPES = ["symmetric", "asymmetric"]
NOISE_RATES = [0.0, 0.1, 0.3, 0.5]

# ──────────────────────────────────────────────
#  Model hyperparameters
# ──────────────────────────────────────────────
# Logistic Regression
LR_MAX_ITER = 1000
LR_SOLVER = "lbfgs"

# Decision Tree
TREE_MAX_DEPTH = 20
TREE_MIN_SAMPLES_LEAF = 5

# MLP (PyTorch)
MLP_HIDDEN_DIM = 256
MLP_DROPOUT = 0.3
MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BATCH_SIZE = 256
MLP_WEIGHT_DECAY = 1e-4

# ──────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────
ECE_NUM_BINS = 15

# ──────────────────────────────────────────────
#  Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
#  Dataset subsampling (for speed on CPU)
# ──────────────────────────────────────────────
# Set to None for full dataset, or an integer to subsample training data
CIFAR10_SUBSAMPLE = 10_000  # CIFAR-10 can be slow on CPU
MNIST_SUBSAMPLE = None      # MNIST is fast enough
ADULT_SUBSAMPLE = None      # Adult is small
