"""
Naive Fine-Tuning (baseline / cota inferior).

Reentrena el modelo completo con cross-entropy en cada tarea nueva, sin ningún
mecanismo para mitigar el olvido catastrófico. Es lo que esperaríamos que pase
si simplemente ignoramos el problema: el modelo olvida todo lo anterior.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from methods.base import BaseMethod
from models.backbone import Backbone


class NaiveFineTuning(BaseMethod):
    """Fine-tuning secuencial sin ninguna mitigación del olvido.

    Sirve como baseline (cota inferior) para comparar contra los otros métodos.
    Solo usa datos de la tarea actual, sin buffer de replay.
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
                self._validate_batch_labels(labels, task_id)

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
