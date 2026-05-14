#!/usr/bin/env python3
"""
Noise correction experiments.

Demonstrates that noise-aware training improves robustness:
  1. Train MLP on noisy data with standard CrossEntropyLoss → baseline.
  2. Estimate transition matrix T̂ from the trained model (anchor-point method).
  3. Retrain MLP with BackwardCorrectionLoss using T̂.
  4. Retrain MLP with GeneralizedCrossEntropy.
  5. Compare all three.

Usage:
    python run_correction.py
    python run_correction.py --dataset mnist --noise-rate 0.3 --noise-type symmetric
"""

import argparse
import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from config import set_seed
from src.datasets import load_dataset
from src.noise import inject_noise
from src.models import get_mlp
from src.losses import BackwardCorrectionLoss, GeneralizedCrossEntropy
from src.train import train_pytorch, create_dataloader
from src.evaluate import evaluate_pytorch
from src.transition_matrix import estimate_transition_matrix, evaluate_T_estimation
from src.plotting import plot_transition_matrices, plot_correction_comparison

import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description="Noise Correction Experiments")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=config.DATASETS)
    parser.add_argument("--noise-rate", type=float, default=0.3)
    parser.add_argument("--noise-type", type=str, default="symmetric",
                        choices=config.NOISE_TYPES)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_name = args.dataset
    eta = args.noise_rate
    noise_type = args.noise_type

    print("=" * 70)
    print("NOISE CORRECTION EXPERIMENTS")
    print("=" * 70)
    print(f"Dataset:    {dataset_name}")
    print(f"Noise type: {noise_type}")
    print(f"Noise rate: {eta}")
    print(f"Device:     {config.DEVICE}")
    print("=" * 70)

    # 1. Load data
    X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    # 2. Inject noise
    y_noisy, T_true, actual_rate = inject_noise(
        y_train, noise_type=noise_type, eta=eta, num_classes=num_classes
    )
    print(f"\nTrue T diagonal: {np.diag(T_true)}")

    correction_results = {}

    # ── Step 1: Standard CE baseline ──────────────
    print(f"\n{'─' * 50}")
    print("  STEP 1: Standard CrossEntropy (baseline)")
    print(f"{'─' * 50}")

    model_ce = get_mlp(input_dim, num_classes)
    train_loader = create_dataloader(X_train, y_noisy)
    model_ce, _ = train_pytorch(
        model_ce, train_loader,
        criterion=nn.CrossEntropyLoss(),
        verbose=True,
    )
    metrics_ce = evaluate_pytorch(model_ce, X_test, y_test)
    correction_results["Standard CE"] = metrics_ce
    print(f"  Accuracy: {metrics_ce['accuracy']:.4f} | F1: {metrics_ce['macro_f1']:.4f}")

    # ── Step 2: Estimate transition matrix ────────
    print(f"\n{'─' * 50}")
    print("  STEP 2: Estimate Transition Matrix (anchor-point)")
    print(f"{'─' * 50}")

    T_est = estimate_transition_matrix(
        model_ce, X_train, y_noisy, num_classes, top_k=100
    )

    t_metrics = evaluate_T_estimation(T_true, T_est)
    print(f"  T̂ estimation errors:")
    print(f"    L1 error:        {t_metrics['l1_error']:.4f}")
    print(f"    L∞ error:        {t_metrics['linf_error']:.4f}")
    print(f"    Frobenius error: {t_metrics['frobenius_error']:.4f}")
    print(f"  Estimated T diagonal: {np.diag(T_est).round(3)}")

    # Plot transition matrix comparison
    title = f"{dataset_name}_{noise_type}_eta{eta}"
    plot_transition_matrices(T_true, T_est, title=title)

    # ── Step 3: Backward Correction ───────────────
    print(f"\n{'─' * 50}")
    print("  STEP 3: Backward Loss Correction (using T̂)")
    print(f"{'─' * 50}")

    model_bc = get_mlp(input_dim, num_classes)
    bc_loss = BackwardCorrectionLoss(T_est).to(config.DEVICE)
    model_bc, _ = train_pytorch(
        model_bc, train_loader,
        criterion=bc_loss,
        verbose=True,
    )
    metrics_bc = evaluate_pytorch(model_bc, X_test, y_test)
    correction_results["Backward Correction"] = metrics_bc
    print(f"  Accuracy: {metrics_bc['accuracy']:.4f} | F1: {metrics_bc['macro_f1']:.4f}")

    # ── Step 4: Generalized Cross-Entropy ─────────
    print(f"\n{'─' * 50}")
    print("  STEP 4: Generalized Cross-Entropy (GCE, q=0.7)")
    print(f"{'─' * 50}")

    model_gce = get_mlp(input_dim, num_classes)
    gce_loss = GeneralizedCrossEntropy(q=0.7)
    model_gce, _ = train_pytorch(
        model_gce, train_loader,
        criterion=gce_loss,
        verbose=True,
    )
    metrics_gce = evaluate_pytorch(model_gce, X_test, y_test)
    correction_results["GCE (q=0.7)"] = metrics_gce
    print(f"  Accuracy: {metrics_gce['accuracy']:.4f} | F1: {metrics_gce['macro_f1']:.4f}")

    # ── Step 5: Also train on clean for reference ─
    print(f"\n{'─' * 50}")
    print("  STEP 5: Clean Baseline (no noise)")
    print(f"{'─' * 50}")

    model_clean = get_mlp(input_dim, num_classes)
    clean_loader = create_dataloader(X_train, y_train)
    model_clean, _ = train_pytorch(
        model_clean, clean_loader,
        criterion=nn.CrossEntropyLoss(),
        verbose=True,
    )
    metrics_clean = evaluate_pytorch(model_clean, X_test, y_test)
    correction_results["Clean (oracle)"] = metrics_clean
    print(f"  Accuracy: {metrics_clean['accuracy']:.4f} | F1: {metrics_clean['macro_f1']:.4f}")

    # ── Summary ───────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CORRECTION RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Method':<25} {'Accuracy':>10} {'Macro F1':>10} {'ECE':>10}")
    print("─" * 55)
    for name, m in correction_results.items():
        ece_str = f"{m['ece']:.4f}" if m.get('ece') is not None else "N/A"
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {ece_str:>10}")
    print("=" * 70)

    # Plot comparison
    plot_correction_comparison(correction_results)

    print(f"\n✓ Correction experiments complete. Plots saved to {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
