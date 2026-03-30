"""
Learning without Forgetting (LwF).

Reference: Li & Hoiem, "Learning without Forgetting", TPAMI 2018.
https://arxiv.org/abs/1606.09282

When training on task t, LwF uses the previous model's predictions on the
new-task data as soft targets, distilling knowledge via KL divergence.
No access to previous task data is required.
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

LWF_ALPHA = 1.0
LWF_TEMPERATURE = 2.0


class LwF(BaseMethod):
    """Learning without Forgetting continual learning method."""

    def __init__(
        self,
        backbone: Backbone,
        device: torch.device,
        alpha: float = LWF_ALPHA,
        temperature: float = LWF_TEMPERATURE,
        lr: float = 0.01,
    ):
        super().__init__(backbone, device)
        self.alpha = alpha
        self.temperature = temperature
        self.lr = lr
        self._prev_model: Optional[nn.Module] = None

    def begin_task(self, task_id: int) -> None:
        """Save a frozen copy of the current model as the teacher."""
        if task_id > 0:
            self._prev_model = copy.deepcopy(self.backbone).eval()
            for p in self._prev_model.parameters():
                p.requires_grad = False

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_epochs: int = 50,
    ) -> Dict[str, list]:
        # TODO: implement LwF training loop with knowledge distillation
        raise NotImplementedError
