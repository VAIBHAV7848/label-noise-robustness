"""
Model definitions for label-noise experiments.

Provides:
  - Logistic Regression (sklearn)
  - Decision Tree (sklearn)
  - 2-layer MLP (PyTorch)
"""

import numpy as np
import os
import sys

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ──────────────────────────────────────────────
#  Scikit-learn models
# ──────────────────────────────────────────────

def get_sklearn_model(name: str):
    """
    Get a configured scikit-learn classifier.

    Args:
        name: 'logistic' or 'tree'.

    Returns:
        sklearn estimator.
    """
    if name == "logistic":
        return LogisticRegression(
            max_iter=config.LR_MAX_ITER,
            solver=config.LR_SOLVER,
            random_state=config.SEED,
            n_jobs=-1,
        )
    elif name == "tree":
        return DecisionTreeClassifier(
            max_depth=config.TREE_MAX_DEPTH,
            min_samples_leaf=config.TREE_MIN_SAMPLES_LEAF,
            random_state=config.SEED,
        )
    else:
        raise ValueError(f"Unknown sklearn model '{name}'. Use 'logistic' or 'tree'.")


# ──────────────────────────────────────────────
#  PyTorch MLP
# ──────────────────────────────────────────────

class MLP(nn.Module):
    """
    2-hidden-layer MLP for classification.

    Architecture:
        Input → Linear(input_dim, hidden) → ReLU → Dropout
              → Linear(hidden, hidden) → ReLU → Dropout
              → Linear(hidden, num_classes)

    Outputs raw logits (no softmax).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = None,
        dropout: float = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or config.MLP_HIDDEN_DIM
        dropout = dropout or config.MLP_DROPOUT

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes),
        )

        # Initialize weights with Kaiming initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


def get_mlp(input_dim: int, num_classes: int) -> MLP:
    """Create and return a fresh MLP model."""
    model = MLP(input_dim=input_dim, num_classes=num_classes)
    return model.to(config.DEVICE)
