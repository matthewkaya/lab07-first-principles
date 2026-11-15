"""Handwritten epoch and validation-based training functions."""

from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from bikes_data import make_rentals
from model import BikeRentalRegressor
from prepare import SEED, make_dataloaders, prepare_data, save_hour_rentals_plot

plt.switch_backend("Agg")


@dataclass
class TrainingResult:
    """Per-epoch measurements and the best validation checkpoint details."""

    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters for one reproducible neural-network run."""

    name: str
    hidden_sizes: tuple[int, ...] = (64, 32)
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    seed: int = SEED


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
        total_absolute_error += (predictions.detach() - targets).abs().sum().item()
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
        training_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device)
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


def evaluate_baselines(data) -> dict[str, float]:
    """Fit both baselines on training rows and report validation MAE only."""
    training_mean = float(np.mean(data.y_train))
    mean_predictions = np.full_like(data.y_validation, training_mean)
    linear_model = LinearRegression()
    linear_model.fit(data.X_train, data.y_train.ravel())
    linear_predictions = linear_model.predict(data.X_validation).reshape(-1, 1)

    return {
        "training_mean_validation_mae": float(
            mean_absolute_error(data.y_validation, mean_predictions)
        ),
        "linear_regression_validation_mae": float(
            mean_absolute_error(data.y_validation, linear_predictions)
        ),
    }


def save_learning_curve(
    history: list[dict[str, float | int]],
    best_epoch: int,
    path: str | Path,
    title: str,
) -> None:
    """Save training and validation losses with the selected epoch marked."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    training_loss = [float(row["training_loss"]) for row in history]
    validation_loss = [float(row["validation_loss"]) for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, training_loss, label="Training loss")
    ax.plot(epochs, validation_loss, label="Validation loss")
    selected_loss = validation_loss[best_epoch - 1]
    ax.scatter([best_epoch], [selected_loss], color="crimson", zorder=3)
    ax.axvline(best_epoch, color="crimson", linestyle="--", alpha=0.6)
    ax.annotate(f"Best epoch: {best_epoch}", (best_epoch, selected_loss))
    ax.set(title=title, xlabel="Epoch", ylabel="Mean squared error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_experiments(
    output_directory: str | Path = "reports",
    seed: int = SEED,
) -> dict[str, Any]:
    """Run baselines and three network configurations using validation only."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = make_rentals(n=6000, seed=seed)
    save_hour_rentals_plot(frame, output_path / "rentals_by_hour.png")
    data = prepare_data(frame, seed=seed)
    baselines = evaluate_baselines(data)

    configurations = [
        ExperimentConfig(name="main"),
        ExperimentConfig(name="low_learning_rate", learning_rate=0.0001),
        ExperimentConfig(name="wider_hidden_layers", hidden_sizes=(128, 64)),
    ]
    experiment_rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float | int]]] = {}

    for config in configurations:
        seed_everything(config.seed)
        train_loader, validation_loader, _ = make_dataloaders(
            data, batch_size=config.batch_size, seed=config.seed
        )
        model = BikeRentalRegressor(
            input_size=data.X_train.shape[1], hidden_sizes=config.hidden_sizes
        )
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
        result = train_model(
            model,
            train_loader,
            validation_loader,
            optimizer,
            epochs=config.epochs,
        )
        histories[config.name] = result.history
        best_row = result.history[result.best_epoch - 1]
        last_row = result.history[-1]
        experiment_rows.append(
            {
                "name": config.name,
                "hyperparameters": json.dumps(
                    {
                        "hidden_sizes": config.hidden_sizes,
                        "learning_rate": config.learning_rate,
                        "batch_size": config.batch_size,
                        "epochs": config.epochs,
                        "seed": config.seed,
                    },
                    sort_keys=True,
                ),
                "best_epoch": result.best_epoch,
                "best_validation_mae": float(best_row["validation_mae"]),
                "best_validation_loss": result.best_validation_loss,
                "final_training_loss": float(last_row["training_loss"]),
                "final_validation_loss": float(last_row["validation_loss"]),
            }
        )

        curve_name = (
            "learning_curves.png"
            if config.name == "main"
            else f"learning_curves_{config.name}.png"
        )
        save_learning_curve(
            result.history,
            result.best_epoch,
            output_path / curve_name,
            title=f"{config.name.replace('_', ' ').title()} run",
        )
        if config.name == "main":
            with (output_path / "history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as history_file:
                writer = csv.DictWriter(
                    history_file,
                    fieldnames=[
                        "epoch",
                        "training_loss",
                        "validation_loss",
                        "validation_mae",
                    ],
                )
                writer.writeheader()
                writer.writerows(result.history)

    with (output_path / "experiments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as experiment_file:
        writer = csv.DictWriter(experiment_file, fieldnames=experiment_rows[0].keys())
        writer.writeheader()
        writer.writerows(experiment_rows)

    selected_experiment = min(
        experiment_rows, key=lambda row: row["best_validation_mae"]
    )
    report = {
        "baselines": baselines,
        "experiments": experiment_rows,
        "selected_neural_network": selected_experiment["name"],
        "selection_metric": "best_validation_mae",
    }
    with (output_path / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as metrics_file:
        json.dump(report, metrics_file, indent=2)
        metrics_file.write("\n")

    return report


def main() -> None:
    """Run Phase 4 validation experiments and print their validation metrics."""
    torch.set_num_threads(1)
    report = run_experiments()
    print("Validation MAE (rentals):")
    for baseline, score in report["baselines"].items():
        print(f"  {baseline}: {score:.3f}")
    for row in report["experiments"]:
        print(
            f"  {row['name']}: {row['best_validation_mae']:.3f} "
            f"(best epoch {row['best_epoch']})"
        )
    print(f"Selected neural network: {report['selected_neural_network']}")


if __name__ == "__main__":
    main()
