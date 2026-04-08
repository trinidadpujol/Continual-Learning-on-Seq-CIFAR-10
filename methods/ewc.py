"""
Elastic Weight Consolidation (EWC).

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks", PNAS 2017. https://arxiv.org/abs/1612.00796

Algorithm
---------
After completing task t, EWC:
  1. Snapshots the current optimal parameters  θ*_t.
  2. Estimates the diagonal of the Fisher Information Matrix F_t by
     accumulating squared gradients of the log-likelihood (cross-entropy)
     over the task-t training data.

When training on a subsequent task t+1, the loss becomes:
    L = L_CE  +  (λ/2) * Σ_prev_t  Σ_i  F_t_i * (θ_i - θ*_t_i)^2

The sum over previous tasks (online/accumulate mode) means each new task
adds one more penalty term.  This is the "multi-head" variant: separate
Fisher/optima per past task, summed together.

Fisher estimation
-----------------
We use the empirical Fisher (squared gradients of the CE loss w.r.t. the
current parameters, averaged over all batches):

    F_i ≈ (1/N) Σ_n (∂ log p(y_n|x_n, θ) / ∂ θ_i)^2

Computed with a single forward+backward pass per batch (no second-order
computation required).
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from methods.base import BaseMethod
from models.backbone import Backbone

EWC_LAMBDA = 400.0


class EWC(BaseMethod):
    """Elastic Weight Consolidation continual learning method.

    Args:
        backbone:    Backbone instance (full model, encoder + classifier).
        device:      Torch device.
        ewc_lambda:  Regularisation strength λ.  Higher values preserve past
                     tasks more strongly at the cost of plasticity on new tasks.
                     400 is a reasonable default for CIFAR-10 with ResNet-18.
        lr:          SGD learning rate.
        momentum:    SGD momentum.
        weight_decay: L2 regularisation (separate from EWC penalty).
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

        # One entry per completed past task.
        # _fisher[t][name] = diagonal Fisher tensor  (same shape as param)
        # _optima[t][name] = optimal parameter snapshot after task t
        self._fisher: List[Dict[str, torch.Tensor]] = []
        self._optima: List[Dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_epochs: int = 50,
    ) -> Dict[str, list]:
        """Train with CE loss + EWC penalty from all previously seen tasks.

        Args:
            task_id:      0-based task index.
            train_loader: DataLoader containing only task_id's samples.
            n_epochs:     Number of epochs.

        Returns:
            {"loss": [...], "acc": [...], "penalty": [...]}
            One entry per epoch; "penalty" is the EWC term (0 on task 0).
        """
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
                logits = self.backbone(imgs)
                ce     = criterion(logits, labels)
                penalty = self._ewc_penalty()
                loss   = ce + (self.ewc_lambda / 2.0) * penalty

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

    # ------------------------------------------------------------------
    # Post-task: Fisher estimation and parameter snapshot
    # ------------------------------------------------------------------

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Estimate Fisher diagonal and snapshot parameters after task_id.

        Uses the empirical Fisher: squared gradients of the CE loss averaged
        over all batches of the completed task.

        Args:
            task_id:      Index of the just-completed task.
            train_loader: DataLoader for the task-t training data (used to
                          compute the Fisher estimate on task-t distribution).
        """
        super().end_task(task_id, train_loader)

        self.backbone.eval()
        criterion = nn.CrossEntropyLoss()

        # Accumulate squared gradients
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

        # Average over batches
        n_batches = max(n_batches, 1)
        for name in fisher_diag:
            fisher_diag[name] /= n_batches

        # Snapshot current optimal parameters (detached, on CPU to save memory)
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

    # ------------------------------------------------------------------
    # EWC penalty
    # ------------------------------------------------------------------

    def _ewc_penalty(self) -> torch.Tensor:
        """Compute Σ_prev_tasks Σ_params F_i * (θ_i - θ*_i)^2.

        Returns a scalar tensor (0 if no past tasks).
        """
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
