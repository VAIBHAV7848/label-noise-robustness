"""
Dataset loading utilities for label-noise experiments.

Supports:
  - MNIST (via torchvision)
  - CIFAR-10 (via torchvision)
  - UCI Adult (via sklearn / OpenML)

Every loader returns (X_train, y_train, X_test, y_test) as numpy arrays.
Features are flattened and normalised to [0, 1] for image datasets.
"""

import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_mnist(subsample: int = None) -> tuple:
    """
    Load MNIST dataset.

    Returns:
        (X_train, y_train, X_test, y_test) — flattened 784-dim vectors, float32.
    """
    from torchvision import datasets, transforms

    train_ds = datasets.MNIST(root=config.DATA_DIR, train=True, download=True)
    test_ds = datasets.MNIST(root=config.DATA_DIR, train=False, download=True)

    X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy().astype(np.int64)
    X_test = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_ds.targets.numpy().astype(np.int64)

    subsample = subsample or config.MNIST_SUBSAMPLE
    if subsample is not None and subsample < len(X_train):
        rng = np.random.RandomState(config.SEED)
        idx = rng.choice(len(X_train), subsample, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    print(f"[MNIST] Train: {X_train.shape}, Test: {X_test.shape}, "
          f"Classes: {len(np.unique(y_train))}")
    return X_train, y_train, X_test, y_test


def load_cifar10(subsample: int = None) -> tuple:
    """
    Load CIFAR-10 dataset.

    Returns:
        (X_train, y_train, X_test, y_test) — flattened 3072-dim vectors, float32.
    """
    from torchvision import datasets

    train_ds = datasets.CIFAR10(root=config.DATA_DIR, train=True, download=True)
    test_ds = datasets.CIFAR10(root=config.DATA_DIR, train=False, download=True)

    X_train = np.array(train_ds.data).reshape(-1, 3072).astype(np.float32) / 255.0
    y_train = np.array(train_ds.targets).astype(np.int64)
    X_test = np.array(test_ds.data).reshape(-1, 3072).astype(np.float32) / 255.0
    y_test = np.array(test_ds.targets).astype(np.int64)

    subsample = subsample or config.CIFAR10_SUBSAMPLE
    if subsample is not None and subsample < len(X_train):
        rng = np.random.RandomState(config.SEED)
        idx = rng.choice(len(X_train), subsample, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    print(f"[CIFAR-10] Train: {X_train.shape}, Test: {X_test.shape}, "
          f"Classes: {len(np.unique(y_train))}")
    return X_train, y_train, X_test, y_test


def load_adult(subsample: int = None) -> tuple:
    """
    Load UCI Adult (Census Income) dataset via OpenML.

    Returns:
        (X_train, y_train, X_test, y_test) — feature-engineered, float32.
        Binary classification: income <=50K (0) vs >50K (1).
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import pandas as pd

    # Fetch Adult dataset from OpenML (dataset id 1590)
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame

    # Target encoding
    le = LabelEncoder()
    y = le.fit_transform(df.iloc[:, -1])  # <=50K → 0, >50K → 1

    # Feature encoding: one-hot for categoricals, standardise numericals
    X = df.iloc[:, :-1]
    X = pd.get_dummies(X, drop_first=True)  # one-hot encode categoricals
    X = X.values.astype(np.float32)

    # Standardise features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=y
    )

    subsample = subsample or config.ADULT_SUBSAMPLE
    if subsample is not None and subsample < len(X_train):
        rng = np.random.RandomState(config.SEED)
        idx = rng.choice(len(X_train), subsample, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    print(f"[Adult] Train: {X_train.shape}, Test: {X_test.shape}, "
          f"Classes: {len(np.unique(y_train))}")
    return X_train, y_train, X_test, y_test


def load_dataset(name: str) -> tuple:
    """
    Dispatcher: load a dataset by name.

    Args:
        name: one of 'mnist', 'cifar10', 'adult'.

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    loaders = {
        "mnist": load_mnist,
        "cifar10": load_cifar10,
        "adult": load_adult,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(loaders.keys())}")
    return loaders[name]()
