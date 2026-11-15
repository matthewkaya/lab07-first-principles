"""Handwritten epoch and validation-based training functions."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass
class TrainingResult:
    """Per-epoch measurements and the best validation checkpoint details."""

    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> tuple[float, float]:
    """Train for one epoch when given an optimizer, otherwise evaluate.

    Returns sample-weighted mean loss and mean absolute error in rental counts.
    """
    if criterion is None:
        criterion = nn.MSELoss()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            predictions = model(features)
            if predictions.shape != targets.shape:
                raise ValueError(
                    "Model predictions and targets must have the same shape; "
                    f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
                )
            loss = criterion(predictions, targets)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_absolute_error += (
            (predictions.detach() - targets).abs().sum().item()
        )
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot run an epoch with an empty DataLoader.")

    mean_loss = total_loss / total_samples
    mean_absolute_error = total_absolute_error / total_samples
    return mean_loss, mean_absolute_error


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: Optimizer,
    epochs: int,
    criterion: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> TrainingResult:
    """Train for multiple epochs and restore the lowest-validation-loss weights."""
    if epochs < 1:
        raise ValueError("epochs must be at least one.")
    if criterion is None:
        criterion = nn.MSELoss()

    model.to(device)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        training_loss, _ = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation_loss, validation_mae = run_epoch(
            model, validation_loader, criterion, optimizer=None, device=device
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "validation_mae": validation_mae,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training completed without a validation checkpoint.")
    model.load_state_dict(best_state)
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )
