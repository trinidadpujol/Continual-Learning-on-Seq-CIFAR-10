"""
Supervised Contrastive Loss (SupCon).

Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
https://arxiv.org/abs/2004.11362
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Expects features of shape (B, n_views, feat_dim) and labels of shape (B,).
    For self-supervised use, pass labels=None to fall back to SimCLR loss.

    Args:
        temperature: Logit scale temperature tau.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, n_views, feat_dim) — L2-normalised embeddings.
            labels:   (B,) integer class labels. If None, uses SimCLR objective.

        Returns:
            Scalar loss.
        """
        # TODO: implement SupCon loss
        raise NotImplementedError
