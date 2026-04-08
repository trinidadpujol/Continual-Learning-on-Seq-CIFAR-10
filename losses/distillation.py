"""
Asymmetric Distillation Loss (A-Distill) for Co²L.

Reference: Cha et al., "Co²L: Contrastive Continual Learning", ICCV 2021.
https://arxiv.org/abs/2106.14413

Formulation
-----------
Given a batch of B current-task samples:

  z_curr[i]  = L2-normalised encoder embedding from the *current* model
  z_prev[i]  = L2-normalised encoder embedding from the *frozen* previous model

The asymmetric loss treats each z_curr[i] as an anchor and its matching
z_prev[i] as the single positive, while all other z_prev[j≠i] are negatives:

    ℓ_i = -log  exp(z_curr_i · z_prev_i / τ)
                ─────────────────────────────────────
                Σ_{j=1..B}  exp(z_curr_i · z_prev_j / τ)

    L_distill = (1/B) Σ_i ℓ_i

This is "asymmetric" because:
  • Only z_curr acts as the query — gradients flow only into the *current* model.
  • z_prev is treated as read-only keys (no gradient; teacher is frozen).
  • There is exactly one positive per anchor (the same sample in the old space).

Contrast with SupCon, where every same-class pair is a positive.  Here the
signal is purely: "keep each sample close to where the old model placed it",
independent of class labels.

Numerical stability
-------------------
Standard log-sum-exp trick: subtract row max before exponentiation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricDistillationLoss(nn.Module):
    """Asymmetric Distillation Loss as described in Co²L (Cha et al. 2021).

    Args:
        temperature: Contrastive temperature τ.  Should match the SupCon
                     temperature used during pre-training (default 0.07).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        current_features: torch.Tensor,
        previous_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the asymmetric distillation loss.

        Args:
            current_features:  (B, feat_dim) L2-normalised embeddings from
                               the *current* (student) model.
            previous_features: (B, feat_dim) L2-normalised embeddings from
                               the *frozen* (teacher) model.
                               Must correspond to the same B samples in the
                               same order as current_features.

        Returns:
            Scalar loss — mean over all B anchors.

        Raises:
            ValueError: if the two feature tensors have different shapes or
                        the batch size is less than 2.
        """
        if current_features.shape != previous_features.shape:
            raise ValueError(
                f"current_features {current_features.shape} and "
                f"previous_features {previous_features.shape} must match."
            )
        B = current_features.size(0)
        if B < 2:
            raise ValueError("AsymmetricDistillationLoss requires B >= 2.")

        # ── Similarity matrix: z_curr (queries) × z_prev (keys) ──────────
        # Both are already L2-normalised → dot product = cosine similarity.
        # sim[i, j] = z_curr_i · z_prev_j / τ              shape: (B, B)
        sim = torch.matmul(current_features, previous_features.T) / self.temperature

        # ── Numerically stable log-sum-exp ────────────────────────────────
        max_sim = sim.max(dim=1, keepdim=True).values.detach()   # (B, 1)
        exp_sim = torch.exp(sim - max_sim)                        # (B, B)

        # ── Log-probability of the diagonal (positive pair) ───────────────
        # log p_i = sim[i,i] - max_i - log Σ_j exp(sim[i,j] - max_i)
        log_denom = torch.log(exp_sim.sum(dim=1) + 1e-9)         # (B,)
        diag_sim  = sim[torch.arange(B), torch.arange(B)]        # (B,)
        log_prob  = (diag_sim - max_sim.squeeze(1)) - log_denom  # (B,)

        return -log_prob.mean()
