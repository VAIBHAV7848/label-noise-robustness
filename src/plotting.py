"""
Visualisation utilities for label-noise experiments.

Generates:
  - Accuracy vs noise rate line plots
  - Confusion matrix heatmaps
  - Transition matrix comparison heatmaps
  - Robustness drop bar charts
  - Summary results tables (CSV + LaTeX)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Style ──────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 8,
})
sns.set_style("whitegrid")

MODEL_COLORS = {"logistic": "#2196F3", "tree": "#FF9800", "mlp": "#4CAF50"}
MODEL_MARKERS = {"logistic": "o", "tree": "s", "mlp": "D"}
MODEL_LABELS = {"logistic": "Logistic Regression", "tree": "Decision Tree", "mlp": "MLP"}


def plot_accuracy_vs_noise(results, dataset, noise_type, save_dir=None):
    """
    Line plot of accuracy vs noise rate for each model.

    Args:
        results: dict keyed by (dataset, noise_type, eta, model_name) → metrics dict.
        dataset: dataset name.
        noise_type: 'symmetric' or 'asymmetric'.
        save_dir: directory to save plot.
    """
    save_dir = save_dir or config.PLOTS_DIR
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric_key, metric_label in [
        (axes[0], "accuracy", "Test Accuracy"),
        (axes[1], "macro_f1", "Macro F1 Score"),
    ]:
        for model_name in config.MODELS:
            rates, values = [], []
            for eta in config.NOISE_RATES:
                key = (dataset, noise_type, eta, model_name)
                if key in results:
                    rates.append(eta)
                    values.append(results[key][metric_key])

            if rates:
                ax.plot(
                    rates, values,
                    color=MODEL_COLORS.get(model_name, "gray"),
                    marker=MODEL_MARKERS.get(model_name, "x"),
                    label=MODEL_LABELS.get(model_name, model_name),
                    linewidth=2.5,
                    markersize=9,
                )

        ax.set_xlabel("Noise Rate (η)")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{dataset.upper()} — {noise_type.title()} Noise")
        ax.set_xticks(config.NOISE_RATES)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"accuracy_vs_noise_{dataset}_{noise_type}.png"
    plt.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved: {fname}")


def plot_confusion_matrix(cm, title, save_dir=None, filename=None):
    """Heatmap of a confusion matrix."""
    save_dir = save_dir or config.PLOTS_DIR
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fname = filename or f"cm_{title.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
    plt.close()


def plot_transition_matrices(T_true, T_est, title="", save_dir=None):
    """Side-by-side heatmaps of true vs estimated transition matrix."""
    save_dir = save_dir or config.PLOTS_DIR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(T_true, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax1,
                vmin=0, vmax=1, cbar=True)
    ax1.set_title("True T")
    ax1.set_xlabel("Noisy Label")
    ax1.set_ylabel("True Label")

    sns.heatmap(T_est, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax2,
                vmin=0, vmax=1, cbar=True)
    ax2.set_title("Estimated T̂")
    ax2.set_xlabel("Noisy Label")
    ax2.set_ylabel("True Label")

    fig.suptitle(f"Transition Matrix Comparison {title}", fontsize=14)
    plt.tight_layout()
    fname = f"transition_matrix_{title.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved: {fname}")


def plot_robustness_comparison(results, dataset, noise_type, save_dir=None):
    """Bar chart of accuracy drop from clean baseline for each model × noise rate."""
    save_dir = save_dir or config.PLOTS_DIR

    # Get clean accuracies
    clean_accs = {}
    for model_name in config.MODELS:
        key = (dataset, noise_type, 0.0, model_name)
        if key in results:
            clean_accs[model_name] = results[key]["accuracy"]

    if not clean_accs:
        return

    noisy_rates = [r for r in config.NOISE_RATES if r > 0]
    x = np.arange(len(noisy_rates))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(config.MODELS):
        if model_name not in clean_accs:
            continue
        drops = []
        for eta in noisy_rates:
            key = (dataset, noise_type, eta, model_name)
            if key in results:
                drop = (clean_accs[model_name] - results[key]["accuracy"]) / clean_accs[model_name] * 100
                drops.append(drop)
            else:
                drops.append(0)

        ax.bar(
            x + i * width, drops, width,
            label=MODEL_LABELS.get(model_name, model_name),
            color=MODEL_COLORS.get(model_name, "gray"),
            alpha=0.85,
        )

    ax.set_xlabel("Noise Rate (η)")
    ax.set_ylabel("Accuracy Drop (%)")
    ax.set_title(f"Robustness Drop — {dataset.upper()} ({noise_type.title()} Noise)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{r:.0%}" for r in noisy_rates])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = f"robustness_drop_{dataset}_{noise_type}.png"
    plt.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved: {fname}")


def generate_results_table(results, save_dir=None):
    """Generate a summary CSV and LaTeX table of all results."""
    save_dir = save_dir or config.TABLES_DIR

    rows = []
    for key, metrics in sorted(results.items()):
        dataset, noise_type, eta, model_name = key
        rows.append({
            "Dataset": dataset.upper(),
            "Noise Type": noise_type.title(),
            "Noise Rate (η)": eta,
            "Model": MODEL_LABELS.get(model_name, model_name),
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Macro F1": f"{metrics['macro_f1']:.4f}",
            "ECE": f"{metrics.get('ece', 'N/A')}",
        })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(save_dir, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [Table] Saved CSV: {csv_path}")

    # Save LaTeX (optional — requires jinja2 >= 3.1.5)
    try:
        latex_path = os.path.join(save_dir, "results_summary.tex")
        df.to_latex(latex_path, index=False, float_format="%.4f")
        print(f"  [Table] Saved LaTeX: {latex_path}")
    except (ImportError, Exception) as e:
        print(f"  [Table] LaTeX export skipped ({e})")

    # Print to console
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    return df


def plot_correction_comparison(results_dict, save_dir=None):
    """
    Bar chart comparing standard CE, backward correction, and GCE accuracy.

    Args:
        results_dict: dict keyed by loss_name → metrics dict.
    """
    save_dir = save_dir or config.PLOTS_DIR

    names = list(results_dict.keys())
    accs = [results_dict[n]["accuracy"] for n in names]
    f1s = [results_dict[n]["macro_f1"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#F44336", "#4CAF50", "#2196F3", "#FF9800"]

    ax1.bar(names, accs, color=colors[:len(names)], alpha=0.85)
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Accuracy: Standard vs Noise-Corrected")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(names, f1s, color=colors[:len(names)], alpha=0.85)
    ax2.set_ylabel("Macro F1")
    ax2.set_title("F1: Standard vs Noise-Corrected")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = "correction_comparison.png"
    plt.savefig(os.path.join(save_dir, fname), bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved: {fname}")
