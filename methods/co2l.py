"""
Contrastive Continual Learning (Co²L).

Reference: Cha et al., "Co²L: Contrastive Continual Learning", ICCV 2021.
https://arxiv.org/abs/2106.14413

Co²L combines:
  1. Supervised Contrastive Loss (SupCon) on current-task data + replay buffer.
  2. Asymmetric Distillation Loss (A-Distill) to preserve the previous model's
     representation structure.

Combined objective: L = L_SupCon + lambda * L_distill
Expected performance on Seq-CIFAR-10: Class-IL ~47-52%, Task-IL ~88-92%.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.buffer import ReplayBuffer
from losses.supcon import SupConLoss
from losses.distillation import AsymmetricDistillationLoss
from methods.base import BaseMethod
from models.backbone import Backbone

CO2L_LAMBDA = 1.0


class Co2L(BaseMethod):
    """Contrastive Continual Learning method."""

    def __init__(
        self,
        backbone: Backbone,
        device: torch.device,
        replay_buffer: ReplayBuffer,
        co2l_lambda: float = CO2L_LAMBDA,
        temperature: float = 0.07,
        lr: float = 0.5,
    ):
        super().__init__(backbone, device)
        self.buffer = replay_buffer
        self.co2l_lambda = co2l_lambda
        self.supcon_loss = SupConLoss(temperature=temperature)
        self.distill_loss = AsymmetricDistillationLoss(temperature=temperature)
        self.lr = lr
        self._prev_model: Optional[nn.Module] = None

    def begin_task(self, task_id: int) -> None:
        """Freeze a copy of the backbone as the teacher for distillation."""
        if task_id > 0:
            self._prev_model = copy.deepcopy(self.backbone).eval()
            for p in self._prev_model.parameters():
                p.requires_grad = False

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Update replay buffer with samples from the finished task."""
        self.buffer.update_from_loader(train_loader, task_id)
        self._log(f"buffer updated — {self.buffer}")

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_epochs: int = 500,
    ) -> Dict[str, list]:
        # TODO: implement Co²L training loop
        raise NotImplementedError
