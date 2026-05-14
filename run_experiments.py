#!/usr/bin/env python3
"""
Main experiment orchestrator for label-noise robustness study.

Runs the full experimental grid:
  datasets × noise_types × noise_rates × models

Usage:
    python run_experiments.py
    python run_experiments.py --datasets mnist adult --models logistic mlp
    python run_experiments.py --datasets mnist --noise-rates 0.0 0.1 0.3
"""

import argparse
import json
import os
import sys
import time
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from config import set_seed
from src.datasets import load_dataset
from src.noise import inject_noise
from src.models import get_sklearn_model, get_mlp
from src.train import train_sklearn, train_pytorch, create_dataloader
from src.evaluate import evaluate_sklearn, evaluate_pytorch, compute_robustness_drop
from src.plotting import (
    plot_accuracy_vs_noise,
    plot_confusion_matrix,
    plot_robustness_comparison,
    generate_results_table,
)


def run_single_experiment(
    dataset_name, noise_type, eta, model_name, X_train, y_train, X_test, y_test
):
    """
    Run a single experiment: inject noise → train → evaluate.

    Returns:
        metrics: dict with accuracy, macro_f1, confusion_matrix, ece.
    """
    num_classes = len(np.unique(y_train))

    # 1. Inject noise
    y_noisy, T_true, actual_rate = inject_noise(
        y_train, noise_type=noise_type, eta=eta, num_classes=num_classes
    )

    # 2. Train
    if model_name in ("logistic", "tree"):
        model = get_sklearn_model(model_name)
        model = train_sklearn(model, X_train, y_noisy)
        metrics = evaluate_sklearn(model, X_test, y_test)
    elif model_name == "mlp":
        input_dim = X_train.shape[1]
        model = get_mlp(input_dim, num_classes)
        train_loader = create_dataloader(X_train, y_noisy)
        model, history = train_pytorch(model, train_loader, verbose=False)
        metrics = evaluate_pytorch(model, X_test, y_test)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    metrics["actual_noise_rate"] = actual_rate
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Label Noise Robustness Experiments")
    parser.add_argument(
        "--datasets", nargs="+", default=config.DATASETS,
        choices=config.DATASETS, help="Datasets to evaluate."
    )
    parser.add_argument(
        "--models", nargs="+", default=config.MODELS,
        choices=config.MODELS, help="Models to evaluate."
    )
    parser.add_argument(
        "--noise-types", nargs="+", default=config.NOISE_TYPES,
        choices=config.NOISE_TYPES, help="Noise types."
    )
    parser.add_argument(
        "--noise-rates", nargs="+", type=float, default=config.NOISE_RATES,
        help="Noise rates (e.g., 0.0 0.1 0.3 0.5)."
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("LABEL NOISE ROBUSTNESS EXPERIMENTS")
    print("=" * 70)
    print(f"Datasets:    {args.datasets}")
    print(f"Models:      {args.models}")
    print(f"Noise types: {args.noise_types}")
    print(f"Noise rates: {args.noise_rates}")
    print(f"Device:      {config.DEVICE}")
    print("=" * 70)

    all_results = {}
    start_time = time.time()

    for dataset_name in args.datasets:
        print(f"\n{'─' * 50}")
        print(f"  DATASET: {dataset_name.upper()}")
        print(f"{'─' * 50}")

        try:
            X_train, y_train, X_test, y_test = load_dataset(dataset_name)
        except Exception as e:
            print(f"  [ERROR] Failed to load {dataset_name}: {e}")
            continue

        for noise_type in args.noise_types:
            for eta in args.noise_rates:
                for model_name in args.models:
                    key = (dataset_name, noise_type, eta, model_name)
                    print(f"\n  ▸ {dataset_name} | {noise_type} | η={eta:.2f} | {model_name}")

                    try:
                        metrics = run_single_experiment(
                            dataset_name, noise_type, eta, model_name,
                            X_train, y_train, X_test, y_test,
                        )
                        all_results[key] = metrics
                        print(f"    Accuracy: {metrics['accuracy']:.4f} | "
                              f"F1: {metrics['macro_f1']:.4f} | "
                              f"ECE: {metrics.get('ece', 'N/A')}")
                    except Exception as e:
                        print(f"    [ERROR] {e}")
                        import traceback
                        traceback.print_exc()

    # ── Generate plots ──────────────────────────────
    print(f"\n{'=' * 50}")
    print("GENERATING PLOTS & TABLES")
    print(f"{'=' * 50}")

    for dataset_name in args.datasets:
        for noise_type in args.noise_types:
            try:
                plot_accuracy_vs_noise(all_results, dataset_name, noise_type)
                plot_robustness_comparison(all_results, dataset_name, noise_type)
            except Exception as e:
                print(f"  [Plot Error] {dataset_name}/{noise_type}: {e}")

    # Save selected confusion matrices (clean and 50% noise for each model)
    for dataset_name in args.datasets:
        for model_name in args.models:
            for eta in [0.0, 0.5]:
                for noise_type in args.noise_types:
                    key = (dataset_name, noise_type, eta, model_name)
                    if key in all_results and "confusion_matrix" in all_results[key]:
                        cm = all_results[key]["confusion_matrix"]
                        title = f"{dataset_name}_{model_name}_{noise_type}_eta{eta}"
                        try:
                            plot_confusion_matrix(cm, title)
                        except Exception:
                            pass

    # Generate summary table
    # Convert confusion_matrix to list for JSON serialisation
    serialisable = {}
    for key, metrics in all_results.items():
        m = dict(metrics)
        if "confusion_matrix" in m:
            m["confusion_matrix"] = m["confusion_matrix"].tolist()
        serialisable[str(key)] = m

    json_path = os.path.join(config.RESULTS_DIR, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  [JSON] All results saved to: {json_path}")

    generate_results_table(all_results)

    elapsed = time.time() - start_time
    print(f"\n✓ All experiments completed in {elapsed:.1f}s")
    print(f"  Results in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
