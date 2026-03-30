"""
Naive Fine-Tuning baseline.

Simply re-trains the full model on each new task with cross-entropy loss,
with no mechanism to mitigate catastrophic forgetting.
This serves as the lower-bound reference for all other methods.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from methods.base import BaseMethod
from models.backbone import Backbone


class NaiveFineTuning(BaseMethod):
    """Fine-tune the full model on each task without any forgetting prevention."""

    def __init__(
        self,
        backbone: Backbone,
        device: torch.device,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
    ):
        super().__init__(backbone, device)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_epochs: int = 50,
    ) -> Dict[str, list]:
        # TODO: implement naive fine-tuning training loop
        raise NotImplementedError
