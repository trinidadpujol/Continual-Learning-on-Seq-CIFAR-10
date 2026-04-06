"""
Naive Fine-Tuning baseline.

Simply re-trains the full model on each new task with cross-entropy loss,
with no mechanism to mitigate catastrophic forgetting.
This serves as the lower-bound reference for all other methods.

Data protocol:  no replay — train_loader contains ONLY task-t samples.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from methods.base import BaseMethod
from models.backbone import Backbone


class NaiveFineTuning(BaseMethod):
    """Fine-tune the full model on each task without any forgetting prevention.

    Training loop contract
    ----------------------
    - Iterates only over the DataLoader passed in (task-t data exclusively).
    - _validate_batch_labels() is called every batch — any stray label from
      another task raises RuntimeError immediately.
    - No buffer is consulted.
    """

    uses_replay: bool = False

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
        """Train on task_id using only the provided DataLoader.

        Args:
            task_id:      Index of the current task (0-based).
            train_loader: DataLoader whose samples belong to task_id only.
            n_epochs:     Number of passes over the task data.

        Returns:
            {"loss": [float, ...], "acc": [float, ...]}  — one entry per epoch.
        """
        # begin_task sets _allowed_classes; call it here too in case the
        # caller skipped the ContinualTrainer and invoked train_task directly.
        if self._current_task_id != task_id:
            self.begin_task(task_id)

        optimizer = self._make_optimizer(self.lr, self.momentum, self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        log: Dict[str, list] = {"loss": [], "acc": []}

        for epoch in range(n_epochs):
            self.backbone.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for imgs, labels in train_loader:
                # ── Protocol enforcement ──────────────────────────────────
                # Every label in this batch must come from task_id's classes.
                # Source: task DataLoader only — no buffer involved here.
                self._validate_batch_labels(labels, task_id)
                # ─────────────────────────────────────────────────────────

                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.backbone(imgs)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                correct      += (logits.argmax(1) == labels).sum().item()
                total        += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc  = correct / total
            log["loss"].append(epoch_loss)
            log["acc"].append(epoch_acc)
            self._epoch_log(task_id, epoch, n_epochs, epoch_loss, epoch_acc)

        return log
