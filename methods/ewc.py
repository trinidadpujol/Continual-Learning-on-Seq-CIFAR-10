"""
Elastic Weight Consolidation (EWC).

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks", PNAS 2017. https://arxiv.org/abs/1612.00796

After each task, EWC estimates the importance of each parameter using the
diagonal of the Fisher Information Matrix and adds a quadratic penalty:
    L = L_CE + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
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
    """Elastic Weight Consolidation continual learning method."""

    def __init__(
        self,
        backbone: Backbone,
        device: torch.device,
        ewc_lambda: float = EWC_LAMBDA,
        lr: float = 0.01,
    ):
        super().__init__(backbone, device)
        self.ewc_lambda = ewc_lambda
        self.lr = lr

        # Accumulated Fisher diagonals and parameter snapshots from past tasks
        self._fisher: List[Dict[str, torch.Tensor]] = []
        self._optima: List[Dict[str, torch.Tensor]] = []

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Compute and store Fisher diagonal after finishing task_id."""
        # TODO: compute Fisher Information Matrix diagonal
        raise NotImplementedError

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_epochs: int = 50,
    ) -> Dict[str, list]:
        # TODO: implement EWC training loop with penalty term
        raise NotImplementedError
