"""
Evaluation metrics for classification under label noise.

Computes: accuracy, macro F1, confusion matrix, ECE, robustness drop.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_ece(probs, labels, num_bins=None):
    """
    Expected Calibration Error (ECE).

    Bins predictions by confidence, computes |accuracy - confidence| per bin.
    """
    num_bins = num_bins or config.ECE_NUM_BINS
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    ece = 0.0
    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_sklearn(model, X_test, y_test):
    """Evaluate a scikit-learn model. Returns metrics dict."""
    y_pred = model.predict(X_test)

    # Get probabilities if available (for ECE)
    ece = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)
            ece = compute_ece(probs, y_test)
        except Exception:
            pass

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "confusion_matrix": cm,
        "ece": ece,
    }


def evaluate_pytorch(model, X_test, y_test, batch_size=512, device=None):
    """Evaluate a PyTorch model. Returns metrics dict."""
    device = device or config.DEVICE
    model.eval()
    model.to(device)

    X_t = torch.tensor(X_test, dtype=torch.float32)
    y_t = torch.tensor(y_test, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(batch_y.numpy())

    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    ece = compute_ece(y_prob, y_true)

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "confusion_matrix": cm,
        "ece": float(ece),
    }


def compute_robustness_drop(clean_acc, noisy_acc):
    """Compute percentage accuracy drop from clean baseline."""
    if clean_acc == 0:
        return 0.0
    return float((clean_acc - noisy_acc) / clean_acc * 100.0)
