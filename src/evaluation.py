from __future__ import annotations

import copy
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def train_supervised_classifier(
    model: nn.Module,
    train_dataset: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    num_workers: int = 2,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """
    Train one freshly initialised supervised classifier.

    The caller must provide only the currently labelled subset.
    """

    if len(train_dataset) == 0:
        raise ValueError(
            "train_dataset must contain labelled samples."
        )

    if epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    effective_batch_size = min(
        batch_size,
        len(train_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )
    )

    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()
            optimizer.step()

            batch_size_actual = (
                targets.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size_actual
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += batch_size_actual

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_accuracy": (
                epoch_accuracy
            ),
            "learning_rate": (
                optimizer.param_groups[0]["lr"]
            ),
            "epoch_seconds": (
                time.time() - epoch_start
            ),
        }

        history.append(epoch_record)

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"loss: {epoch_loss:.4f} "
                f"accuracy: "
                f"{100 * epoch_accuracy:.2f}%"
            )

    return model, history


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    test_dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 2,
) -> dict[str, Any]:
    """
    Evaluate a trained classifier on the CIFAR-10 test set.
    """

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model.eval()

    criterion = nn.CrossEntropyLoss(
        reduction="sum"
    )

    total_loss = 0.0
    correct = 0
    total = 0

    class_correct = torch.zeros(
        10,
        dtype=torch.long,
    )

    class_total = torch.zeros(
        10,
        dtype=torch.long,
    )

    for images, targets in test_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        total_loss += criterion(
            logits,
            targets,
        ).item()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct_mask = (
            predictions == targets
        )

        correct += correct_mask.sum().item()
        total += targets.size(0)

        targets_cpu = targets.cpu()
        correct_cpu = correct_mask.cpu()

        for class_index in range(10):
            class_mask = (
                targets_cpu == class_index
            )

            class_total[class_index] += (
                class_mask.sum()
            )

            class_correct[class_index] += (
                correct_cpu[class_mask].sum()
            )

    per_class_accuracy = {}

    for class_index in range(10):
        class_count = int(
            class_total[class_index]
        )

        if class_count == 0:
            accuracy = None
        else:
            accuracy = float(
                class_correct[class_index]
                / class_total[class_index]
            )

        per_class_accuracy[
            str(class_index)
        ] = accuracy

    return {
        "test_loss": total_loss / total,
        "test_accuracy": correct / total,
        "correct_predictions": correct,
        "test_samples": total,
        "per_class_accuracy": (
            per_class_accuracy
        ),
    }
