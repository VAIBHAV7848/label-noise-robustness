"""
Custom loss functions for noise-robust training.

Implements:
  - BackwardCorrectionLoss: uses the inverse transition matrix T⁻¹ to
    correct the standard cross-entropy loss, yielding an unbiased estimator
    of the clean risk (Patrini et al., 2017).

  - GeneralizedCrossEntropy (GCE): a noise-tolerant loss based on the
    Box–Cox transformation. Interpolates between CE (q→0) and MAE (q=1).
    More robust to label noise than standard CE (Zhang & Sabuncu, 2018).

Both losses operate on logits and integer labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BackwardCorrectionLoss(nn.Module):
    """
    Backward loss correction (Patrini et al., 2017).

    Given the noise transition matrix T where T[i,j] = P(ỹ=j | y=i),
    the corrected loss is:

        ℓ_backward(f(x), ỹ) = Σ_i (T⁻¹)[ỹ, i] · ℓ(f(x), i)

    This gives an unbiased estimate of the clean loss, provided T is
    known (or well-estimated).

    Args:
        T: (K, K) numpy array — noise transition matrix.
    """

    def __init__(self, T: np.ndarray):
        super().__init__()
        # Compute T⁻¹ and register as a buffer (non-trainable)
        T_inv = np.linalg.inv(T)
        self.register_buffer(
            "T_inv", torch.tensor(T_inv, dtype=torch.float32)
        )
        self.num_classes = T.shape[0]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute backward-corrected loss.

        Args:
            logits: (B, K) raw model outputs.
            targets: (B,) noisy integer labels.

        Returns:
            Scalar loss value.
        """
        # Compute per-class CE losses: ℓ(f(x), i) for all i
        log_probs = F.log_softmax(logits, dim=1)  # (B, K)

        # For each sample, weight the losses by T⁻¹[ỹ, :]
        # T_inv[targets] gives (B, K) weights
        weights = self.T_inv[targets]  # (B, K)

        # Corrected loss = Σ_i T⁻¹[ỹ, i] · (-log p_i)
        loss = -torch.sum(weights * log_probs, dim=1)  # (B,)

        # Clamp to avoid extremely negative losses from T⁻¹
        loss = torch.clamp(loss, min=0.0)

        return loss.mean()


class GeneralizedCrossEntropy(nn.Module):
    """
    Generalized Cross-Entropy loss (Zhang & Sabuncu, 2018).

    GCE(f(x), y) = (1 - f_y(x)^q) / q

    where f_y(x) is the predicted probability for the true class.
    - q → 0: recovers standard cross-entropy.
    - q = 1: recovers mean absolute error (most noise-robust).
    - q ∈ (0, 1): trade-off between learning speed and noise robustness.

    Args:
        q: exponent parameter, default 0.7 (good empirical default).
        k: truncation threshold — ignore samples where p_y < k to
           further stabilise training. Default 0.0 (disabled).
           Set to ~0.1 for extra noise robustness on very noisy data.
    """

    def __init__(self, q: float = 0.7, k: float = 0.0):
        super().__init__()
        assert 0 < q <= 1, f"q must be in (0, 1], got {q}"
        self.q = q
        self.k = k

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute GCE loss.

        Args:
            logits: (B, K) raw model outputs.
            targets: (B,) integer labels.

        Returns:
            Scalar loss value.
        """
        probs = F.softmax(logits, dim=1)  # (B, K)

        # Gather predicted probability for the target class
        p_y = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)

        # Clamp for numerical stability
        p_y = torch.clamp(p_y, min=1e-7, max=1.0)

        # GCE loss: (1 - p_y^q) / q
        loss = (1.0 - p_y ** self.q) / self.q

        # Optional truncation: ignore samples with low confidence
        # (the model is very unsure → likely noisy label)
        if self.k > 0:
            mask = p_y.detach() >= self.k
            if mask.sum() > 0:
                loss = loss[mask]
            # If all are masked, fall back to full batch
            # (early training when model is uncertain)

        return loss.mean()
