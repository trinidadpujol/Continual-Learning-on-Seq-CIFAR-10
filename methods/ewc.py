"""
Elastic Weight Consolidation (EWC).
Kirkpatrick et al., PNAS 2017. https://arxiv.org/abs/1612.00796

Idea central: después de aprender la tarea t, estimamos la diagonal de la
Fisher Information Matrix F_t y guardamos los parámetros óptimos θ*_t.
Al entrenar t+1, penalizamos los cambios en parámetros que eran importantes:

    L = L_CE  +  (λ/2) · Σ_prev_t  Σ_i  F_t_i · (θ_i - θ*_t_i)²

Usamos la Fisher empírica (promedio de gradientes² sobre el dataset de la tarea),
que es una aproximación de primer orden pero funciona bien en la práctica.
El término de penalización se acumula para todas las tareas pasadas.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from methods.base import BaseMethod
from models.backbone import Backbone

EWC_LAMBDA = 400.0  # λ=400 funcionó bien en nuestros experimentos con CIFAR-10


class EWC(BaseMethod):
    """EWC: regularización con Fisher diagonal acumulada por tarea.

    ewc_lambda controla el trade-off estabilidad/plasticidad. Valores más altos
    preservan mejor las tareas pasadas pero dificultan aprender las nuevas.
    """

    uses_replay: bool = False

    def __init__(
        self,
        backbone: Backbone,
        device: torch.device,
        ewc_lambda: float = EWC_LAMBDA,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
    ):
        super().__init__(backbone, device)
        self.ewc_lambda   = ewc_lambda
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay

        # Una entrada por tarea completada
        self._fisher: List[Dict[str, torch.Tensor]] = []
        self._optima: List[Dict[str, torch.Tensor]] = []

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

        log: Dict[str, list] = {"loss": [], "acc": [], "penalty": []}

        for epoch in range(n_epochs):
            self.backbone.train()
            running_loss    = 0.0
            running_penalty = 0.0
            correct = total = 0

            for imgs, labels in train_loader:
                self._validate_batch_labels(labels, task_id)

                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits  = self.backbone(imgs)
                ce      = criterion(logits, labels)
                penalty = self._ewc_penalty()
                loss    = ce + (self.ewc_lambda / 2.0) * penalty

                loss.backward()
                optimizer.step()

                running_loss    += ce.item() * labels.size(0)
                running_penalty += penalty.item() * labels.size(0)
                correct         += (logits.argmax(1) == labels).sum().item()
                total           += labels.size(0)

            epoch_loss    = running_loss    / total
            epoch_penalty = running_penalty / total
            epoch_acc     = correct / total

            log["loss"].append(epoch_loss)
            log["acc"].append(epoch_acc)
            log["penalty"].append(epoch_penalty)
            self._epoch_log(task_id, epoch, n_epochs, epoch_loss, epoch_acc)

        return log

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Estima la Fisher diagonal y guarda snapshot de parámetros al terminar la tarea."""
        super().end_task(task_id, train_loader)

        self.backbone.eval()
        criterion = nn.CrossEntropyLoss()

        # Acumulamos gradientes² sobre todos los batches de la tarea
        fisher_diag: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in self.backbone.named_parameters()
            if param.requires_grad
        }

        n_batches = 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)

            self.backbone.zero_grad()
            logits = self.backbone(imgs)
            loss   = criterion(logits, labels)
            loss.backward()

            for name, param in self.backbone.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_diag[name] += param.grad.detach() ** 2

            n_batches += 1

        n_batches = max(n_batches, 1)
        for name in fisher_diag:
            fisher_diag[name] /= n_batches

        # Snapshot de parámetros óptimos (en CPU para ahorrar memoria)
        optima: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in self.backbone.named_parameters()
            if param.requires_grad
        }

        self._fisher.append(fisher_diag)
        self._optima.append(optima)

        self._log(
            f"end_task  task={task_id}  "
            f"Fisher entries={len(fisher_diag)}  "
            f"tasks accumulated={len(self._fisher)}"
        )

    def _ewc_penalty(self) -> torch.Tensor:
        """Calcula Σ_prev_tasks Σ_params F_i · (θ_i - θ*_i)². Devuelve 0 si no hay tareas pasadas."""
        if not self._fisher:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for fisher, optima in zip(self._fisher, self._optima):
            for name, param in self.backbone.named_parameters():
                if name not in fisher:
                    continue
                f  = fisher[name].to(self.device)
                o  = optima[name].to(self.device)
                penalty = penalty + (f * (param - o) ** 2).sum()

        return penalty
