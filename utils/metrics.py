"""
Evaluation metrics for continual learning on Seq-CIFAR-10.

Two evaluation scenarios:
  - Class-IL: model must classify among all 10 classes simultaneously.
  - Task-IL: model knows which task (2-class subset) the sample belongs to.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import TASK_CLASSES


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def evaluate_class_il(
    model: nn.Module,
    test_loaders: List[DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    """
    Class-Incremental evaluation: model sees all 10 classes, no task hint.

    Args:
        model:        Trained backbone with 10-class classifier head.
        test_loaders: One DataLoader per task (tasks seen so far).
        device:       Torch device.

    Returns:
        {"acc_per_task": [...], "avg_acc": float}
    """
    # TODO: implement Class-IL evaluation loop
    raise NotImplementedError


def evaluate_task_il(
    model: nn.Module,
    test_loaders: List[DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    """
    Task-Incremental evaluation: model knows task id, classifies within 2 classes.

    Args:
        model:        Trained backbone with 10-class classifier head.
        test_loaders: One DataLoader per task (tasks seen so far).
        device:       Torch device.

    Returns:
        {"acc_per_task": [...], "avg_acc": float}
    """
    # TODO: implement Task-IL evaluation loop (mask logits to task classes)
    raise NotImplementedError


def compute_forgetting(
    acc_matrix: List[List[float]],
) -> List[float]:
    """
    Compute per-task forgetting from an accuracy matrix.

    acc_matrix[i][j] = accuracy on task j after training on task i.
    Forgetting for task j = acc_matrix[j][j] - acc_matrix[-1][j].

    Returns:
        List of forgetting values (one per task except the last).
    """
    # TODO: implement forgetting metric
    raise NotImplementedError
