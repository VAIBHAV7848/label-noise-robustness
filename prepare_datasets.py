#!/usr/bin/env python3
"""
==========================================================
  Dataset Preparation Script
  --------------------------
  Downloads MNIST, CIFAR-10, and UCI Adult datasets
  and saves them as .npz files for the notebook to load.

  Usage:
      python prepare_datasets.py
==========================================================
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
os.makedirs(DATA_DIR, exist_ok=True)


def prepare_mnist():
    """Download and save MNIST as a .npz file."""
    print("=" * 50)
    print("  Preparing MNIST...")
    print("=" * 50)
    from torchvision import datasets

    train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True)

    X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy().astype(np.int64)
    X_test = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_ds.targets.numpy().astype(np.int64)

    path = os.path.join(DATA_DIR, "mnist.npz")
    np.savez_compressed(path, X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
    print(f"  ✓ Saved: {path}")
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"    Classes: {len(np.unique(y_train))}")
    return path


def prepare_cifar10():
    """Download and save CIFAR-10 as a .npz file."""
    print("\n" + "=" * 50)
    print("  Preparing CIFAR-10...")
    print("=" * 50)
    from torchvision import datasets

    train_ds = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    test_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

    X_train = np.array(train_ds.data).reshape(-1, 3072).astype(np.float32) / 255.0
    y_train = np.array(train_ds.targets).astype(np.int64)
    X_test = np.array(test_ds.data).reshape(-1, 3072).astype(np.float32) / 255.0
    y_test = np.array(test_ds.targets).astype(np.int64)

    # Subsample CIFAR-10 for speed on CPU
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X_train), 10000, replace=False)
    X_train, y_train = X_train[idx], y_train[idx]

    path = os.path.join(DATA_DIR, "cifar10.npz")
    np.savez_compressed(path, X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
    print(f"  ✓ Saved: {path}")
    print(f"    Train: {X_train.shape} (subsampled to 10k), Test: {X_test.shape}")
    print(f"    Classes: {len(np.unique(y_train))}")
    return path


def prepare_adult():
    """Download and save UCI Adult as a .npz file."""
    print("\n" + "=" * 50)
    print("  Preparing UCI Adult...")
    print("=" * 50)
    from sklearn.datasets import fetch_openml
    import pandas as pd

    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame

    le = LabelEncoder()
    y = le.fit_transform(df.iloc[:, -1])

    X = df.iloc[:, :-1]
    X = pd.get_dummies(X, drop_first=True)
    X = X.values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    path = os.path.join(DATA_DIR, "adult.npz")
    np.savez_compressed(path, X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
    print(f"  ✓ Saved: {path}")
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"    Classes: {len(np.unique(y_train))}")
    return path


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║   LABEL NOISE PROJECT — DATASET PREPARATION     ║")
    print("╚══════════════════════════════════════════════════╝\n")

    paths = []
    paths.append(prepare_mnist())
    paths.append(prepare_cifar10())
    paths.append(prepare_adult())

    print("\n" + "=" * 50)
    print("  ALL DATASETS READY!")
    print("=" * 50)
    for p in paths:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  • {os.path.basename(p):15s} — {size_mb:.1f} MB")
    print("\n  Now open the Jupyter notebook and run it!")
