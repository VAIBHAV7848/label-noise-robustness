"""
Synthetic label-noise generators.

Implements:
  - Symmetric noise: flip each label uniformly to any other class with prob η.
  - Asymmetric noise: flip each label according to a class-conditional transition
    matrix T, where T[i, j] = P(ỹ = j | y = i).

The transition matrix T is always row-stochastic (rows sum to 1).
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_symmetric_T(num_classes: int, eta: float) -> np.ndarray:
    """
    Build a symmetric (uniform) noise transition matrix.

    T[i, j] = η / (K-1)   for i ≠ j
    T[i, i] = 1 - η

    Args:
        num_classes: number of classes K.
        eta: noise rate in [0, 1).

    Returns:
        T: (K, K) row-stochastic transition matrix.
    """
    assert 0.0 <= eta < 1.0, f"Noise rate η must be in [0, 1), got {eta}"
    K = num_classes
    T = np.full((K, K), eta / (K - 1))
    np.fill_diagonal(T, 1.0 - eta)
    return T


def build_asymmetric_T(num_classes: int, eta: float) -> np.ndarray:
    """
    Build an asymmetric (class-conditional) noise transition matrix.

    Models realistic label confusion: class i flips to class (i+1) mod K
    with probability η. All other off-diagonal entries are 0.

    For MNIST/CIFAR-10 this means:
      0→1, 1→2, ..., 8→9, 9→0  with probability η.

    For binary (Adult): 0→1 and 1→0 with probability η (same as symmetric).

    Args:
        num_classes: number of classes K.
        eta: noise rate in [0, 1).

    Returns:
        T: (K, K) row-stochastic transition matrix.
    """
    assert 0.0 <= eta < 1.0, f"Noise rate η must be in [0, 1), got {eta}"
    K = num_classes
    T = np.eye(K) * (1.0 - eta)
    for i in range(K):
        j = (i + 1) % K
        T[i, j] = eta
    return T


def apply_noise(labels: np.ndarray, T: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Apply label noise according to transition matrix T.

    For each sample with true label y = i, draw noisy label ỹ ~ Categorical(T[i, :]).

    Args:
        labels: (N,) array of integer labels.
        T: (K, K) row-stochastic transition matrix.
        seed: random seed for reproducibility.

    Returns:
        noisy_labels: (N,) array of noisy integer labels.
    """
    rng = np.random.RandomState(seed)
    K = T.shape[0]
    noisy_labels = labels.copy()

    for i in range(K):
        mask = labels == i
        n_i = mask.sum()
        if n_i > 0:
            noisy_labels[mask] = rng.choice(K, size=n_i, p=T[i])

    return noisy_labels


def inject_noise(
    labels: np.ndarray,
    noise_type: str,
    eta: float,
    num_classes: int = None,
    seed: int = None,
) -> tuple:
    """
    High-level noise injection dispatcher.

    Args:
        labels: (N,) integer labels.
        noise_type: 'symmetric' or 'asymmetric'.
        eta: noise rate in [0, 1). If 0, returns clean labels.
        num_classes: inferred from labels if not provided.
        seed: random seed.

    Returns:
        noisy_labels: (N,) noisy integer labels.
        T: (K, K) transition matrix used.
        actual_noise_rate: empirical fraction of labels that were flipped.
    """
    if seed is None:
        seed = config.SEED

    if num_classes is None:
        num_classes = len(np.unique(labels))

    if eta == 0.0:
        T = np.eye(num_classes)
        return labels.copy(), T, 0.0

    if noise_type == "symmetric":
        T = build_symmetric_T(num_classes, eta)
    elif noise_type == "asymmetric":
        T = build_asymmetric_T(num_classes, eta)
    else:
        raise ValueError(f"Unknown noise type '{noise_type}'. Use 'symmetric' or 'asymmetric'.")

    noisy_labels = apply_noise(labels, T, seed=seed)
    actual_noise_rate = np.mean(noisy_labels != labels)

    print(f"  [Noise] type={noise_type}, η={eta:.2f}, "
          f"actual flip rate={actual_noise_rate:.4f}, "
          f"samples={len(labels)}")

    return noisy_labels, T, actual_noise_rate
