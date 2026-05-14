"""
Training utilities for sklearn and PyTorch models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_dataloader(X, y, batch_size=None, shuffle=True):
    """Wrap numpy arrays into a PyTorch DataLoader."""
    batch_size = batch_size or config.MLP_BATCH_SIZE
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_sklearn(model, X_train, y_train):
    """Train a scikit-learn model. Returns the fitted model."""
    model.fit(X_train, y_train)
    return model


def train_pytorch(
    model,
    train_loader,
    criterion=None,
    optimizer=None,
    epochs=None,
    device=None,
    verbose=True,
):
    """
    Train a PyTorch model.

    Args:
        model: nn.Module.
        train_loader: DataLoader.
        criterion: loss function (default: CrossEntropyLoss).
        optimizer: optimizer (default: Adam).
        epochs: number of epochs (default: config.MLP_EPOCHS).
        device: torch device.
        verbose: print progress.

    Returns:
        model: trained model.
        history: list of epoch losses.
    """
    device = device or config.DEVICE
    epochs = epochs or config.MLP_EPOCHS
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.MLP_LR,
            weight_decay=config.MLP_WEIGHT_DECAY,
        )

    model.train()
    model.to(device)
    history = []

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) if verbose else train_loader
        for batch_x, batch_y in iterator:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        history.append(avg_loss)

        if verbose:
            print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

    return model, history
