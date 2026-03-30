"""
Asymmetric Distillation Loss for Co²L.

Reference: Cha et al., "Co²L: Contrastive Continual Learning", ICCV 2021.
https://arxiv.org/abs/2106.14413

The loss aligns the current model's representations with the previous model's
representations on the current-task data, using an asymmetric formulation that
only updates the student (current model), not the teacher (frozen previous model).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricDistillationLoss(nn.Module):
    """
    Asymmetric Distillation Loss (A-Distill) from Co²L.

    Args:
        temperature: Temperature for the distillation contrastive objective.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        current_features: torch.Tensor,
        previous_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current_features:  (B, feat_dim) — current model embeddings (student).
            previous_features: (B, feat_dim) — frozen previous model embeddings (teacher).

        Returns:
            Scalar distillation loss.
        """
        # TODO: implement asymmetric distillation loss
        raise NotImplementedError
