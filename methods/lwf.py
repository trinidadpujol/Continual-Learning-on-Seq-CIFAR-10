"""
Learning without Forgetting (LwF).

Reference: Li & Hoiem, "Learning without Forgetting", TPAMI 2018.
https://arxiv.org/abs/1606.09282

Algorithm
---------
Before starting task t (t > 0):
  - Snapshot the current model as a frozen "teacher" f_θ_prev.

While training on task t:
  - For each batch x from task-t loader:
      1. Compute soft teacher targets:  p_prev = softmax(f_θ_prev(x) / T)
      2. Compute student soft outputs:  p_curr = log_softmax(f_θ(x) / T)
      3. Distillation loss:  L_dist = T² · KL(p_prev ‖ p_curr)
                                     = T² · Σ p_prev · (log p_prev − p_curr)
      4. CE loss on new task: L_ce = CrossEntropy(f_θ(x), y)
      5. Combined: L = L_ce + α · L_dist

  The T² factor (temperature squared) re-scales the soft-target gradient
  magnitude to match the hard-target gradient scale (Hinton et al. 2015).

Task 0 has no previous model, so L_dist = 0 (pure CE).

Distillation covers ALL 10 logit dimensions, not just the old task classes.
This forces the new model to preserve the full output distribution of the
previous model on new-task inputs — a strong and simple regulariser.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.base import BaseMethod
from models.backbone import Backbone

LWF_ALPHA       = 1.0    # distillation loss weight
LWF_TEMPERATURE = 2.0    # softmax temperature for soft targets


class LwF(BaseMethod):
    """Learning without Forgetting continual learning method.

    Args:
        backbone:    Backbone instance (full model).
        device:      Torch device.
        alpha:       Weight of the distillation loss relative to CE.
                     α=1.0 balances both terms equally.
        temperature: Softmax temperature T for producing soft targets.
                     Higher T → softer probability distributions.
        lr:          SGD learning rate.
        momentum:    SGD momentum.
        weight_decay: L2 regularisation.
    """

    uses_replay: bool = False

    def __init__(
        self,
        backbone: Backbone,
        device: torch.device,
        alpha: float = LWF_ALPHA,
        temperature: float = LWF_TEMPERATURE,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
    ):
        super().__init__(backbone, device)
        self.alpha        = alpha
        self.temperature  = temperature
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self._prev_model: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_task(self, task_id: int) -> None:
        """Snapshot the current model as a frozen teacher before task t.

        On task 0 no teacher exists, so distillation is skipped entirely.
        On task t > 0 a deep copy is frozen (requires_grad=False, eval mode).
        The teacher is moved to the training device but never updated.
        """
        super().begin_task(task_id)
        if task_id > 0:
            self._prev_model = copy.deepcopy(self.backbone).to(self.device)
            self._prev_model.eval()
            for p in self._prev_model.parameters():
                p.requires_grad = False
            self._log(f"teacher snapshot taken for task={task_id}")
        else:
            self._prev_model = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_epochs: int = 50,
    ) -> Dict[str, list]:
        """Train with CE loss + knowledge distillation from the previous model.

        Args:
            task_id:      0-based task index.
            train_loader: DataLoader containing only task_id's samples.
            n_epochs:     Number of epochs.

        Returns:
            {"loss": [...], "acc": [...], "ce": [...], "distill": [...]}
            One entry per epoch.  "distill" is 0.0 on task 0.
        """
        if self._current_task_id != task_id:
            self.begin_task(task_id)

        optimizer = self._make_optimizer(self.lr, self.momentum, self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        log: Dict[str, list] = {"loss": [], "acc": [], "ce": [], "distill": []}

        for epoch in range(n_epochs):
            self.backbone.train()
            running_loss    = 0.0
            running_ce      = 0.0
            running_distill = 0.0
            correct = total = 0

            for imgs, labels in train_loader:
                self._validate_batch_labels(labels, task_id)

                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                logits = self.backbone(imgs)          # (B, num_classes)
                ce     = criterion(logits, labels)

                distill = self._distillation_loss(imgs, logits)
                loss    = ce + self.alpha * distill

                loss.backward()
                optimizer.step()

                B = labels.size(0)
                running_loss    += loss.item()    * B
                running_ce      += ce.item()      * B
                running_distill += distill.item() * B
                correct         += (logits.argmax(1) == labels).sum().item()
                total           += B

            n = max(total, 1)
            log["loss"].append(running_loss    / n)
            log["ce"].append(running_ce        / n)
            log["distill"].append(running_distill / n)
            log["acc"].append(correct / n)
            self._epoch_log(task_id, epoch, n_epochs, running_loss / n, correct / n)

        return log

    # ------------------------------------------------------------------
    # Distillation loss
    # ------------------------------------------------------------------

    def _distillation_loss(
        self,
        imgs: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL-divergence distillation from the frozen teacher.

        L_dist = T² · KL( softmax(z_teacher/T) ‖ log_softmax(z_student/T) )
               = T² · Σ_c  p_teacher_c · (log p_teacher_c − log p_student_c)

        Args:
            imgs:           (B, C, H, W) batch already on device.
            student_logits: (B, num_classes) student logits (already computed).

        Returns:
            Scalar distillation loss (0.0 if no teacher).
        """
        if self._prev_model is None:
            return torch.tensor(0.0, device=self.device)

        T = self.temperature
        with torch.no_grad():
            teacher_logits = self._prev_model(imgs)           # (B, num_classes)

        # Soft teacher targets
        p_teacher = F.softmax(teacher_logits / T, dim=1)     # (B, num_classes)
        # Log-softmax of student at same temperature
        log_p_student = F.log_softmax(student_logits / T, dim=1)

        # KL divergence: mean over batch, T² rescaling
        distill = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T ** 2)
        return distill
