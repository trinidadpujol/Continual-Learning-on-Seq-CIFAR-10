"""
Abstract base class for all continual learning methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.backbone import Backbone


class BaseMethod(ABC):
    """
    Base class for continual learning methods.

    Each method must implement train_task() and may override begin_task() /
    end_task() hooks for per-task setup and teardown (e.g. computing Fisher
    information, saving the previous model, updating the replay buffer).
    """

    def __init__(self, backbone: Backbone, device: torch.device):
        self.backbone = backbone.to(device)
        self.device = device

    def begin_task(self, task_id: int) -> None:
        """Called before training on task_id. Override as needed."""
        pass

    @abstractmethod
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_epochs: int,
    ) -> Dict[str, list]:
        """
        Train on a single task.

        Returns:
            Dictionary of logged metrics, e.g. {"loss": [...], "acc": [...]}.
        """
        ...

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Called after training on task_id. Override as needed."""
        pass
