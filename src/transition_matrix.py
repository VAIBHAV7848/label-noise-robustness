"""
Transition matrix estimation from noisy data.

Implements the anchor-point method (Patrini et al., 2017):
  1. Train a model on the noisy data.
  2. Forward-pass all training examples and collect predicted probabilities.
  3. For each class k, find the sample with the highest predicted P(y=k|x).
  4. The k-th row of estimated T is the noisy-label distribution among top anchors.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def estimate_transition_matrix(
    model, X_train, y_noisy, num_classes, top_k=50, batch_size=512
):
    """
    Estimate noise transition matrix T using anchor-point method.

    For each class k, select top_k samples with highest P(y=k|x),
    then compute empirical distribution of their noisy labels → T[k, :].
    """
    model.eval()
    device = config.DEVICE

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    T_est = np.zeros((num_classes, num_classes))
    for k in range(num_classes):
        p_k = all_probs[:, k]
        n_anchors = min(top_k, len(p_k))
        anchor_indices = np.argsort(p_k)[-n_anchors:]
        anchor_labels = y_noisy[anchor_indices]
        for j in range(num_classes):
            T_est[k, j] = np.mean(anchor_labels == j)

    # Ensure valid stochastic matrix
    T_est = np.clip(T_est, 0.0, 1.0)
    row_sums = T_est.sum(axis=1, keepdims=True)
    T_est = T_est / np.maximum(row_sums, 1e-10)
    return T_est


def evaluate_T_estimation(T_true, T_est):
    """Compare estimated T with ground truth."""
    diff = np.abs(T_true - T_est)
    return {
        "l1_error": float(np.mean(diff)),
        "linf_error": float(np.max(diff)),
        "frobenius_error": float(np.linalg.norm(T_true - T_est, 'fro')),
    }
