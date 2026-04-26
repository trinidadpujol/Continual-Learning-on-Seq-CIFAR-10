"""
Asymmetric Distillation Loss para Co²L (Cha et al., ICCV 2021).
https://arxiv.org/abs/2106.14413

Para cada sample i, tratamos z_curr[i] como anchor y z_prev[i] como el único
positivo, con todos los z_prev[j≠i] como negativos:

    ℓ_i = -log  exp(z_curr_i · z_prev_i / τ)
                ─────────────────────────────
                Σ_j exp(z_curr_i · z_prev_j / τ)

Es "asimétrica" porque solo z_curr actúa como query (gradientes fluyen solo al
modelo actual), mientras z_prev es tratado como read-only (teacher frozen).
La idea es preservar la estructura de representación del modelo anterior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricDistillationLoss(nn.Module):
    """Asymmetric Distillation Loss de Co²L."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        current_features: torch.Tensor,
        previous_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        current_features:  (B, feat_dim) L2-norm, modelo actual (recibe gradientes).
        previous_features: (B, feat_dim) L2-norm, teacher frozen (sin gradientes).
        Devuelve la pérdida escalar promediada sobre el batch.
        """
        if current_features.shape != previous_features.shape:
            raise ValueError(
                f"current_features {current_features.shape} y "
                f"previous_features {previous_features.shape} deben tener el mismo shape."
            )
        B = current_features.size(0)
        if B < 2:
            raise ValueError("AsymmetricDistillationLoss requiere B >= 2.")

        # sim[i,j] = z_curr_i · z_prev_j / τ  (B, B)
        sim = torch.matmul(current_features, previous_features.T) / self.temperature

        # Log-sum-exp estable
        max_sim = sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim - max_sim)

        # Log-prob de la diagonal (par positivo: mismo sample en el espacio del teacher)
        log_denom = torch.log(exp_sim.sum(dim=1) + 1e-9)
        diag_sim  = sim[torch.arange(B), torch.arange(B)]
        log_prob  = (diag_sim - max_sim.squeeze(1)) - log_denom

        return -log_prob.mean()
