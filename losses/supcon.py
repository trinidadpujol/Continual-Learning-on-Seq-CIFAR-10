"""
Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
https://arxiv.org/abs/2004.11362

Para cada anchor i en la matriz de contraste (2B x feat_dim):
  - positivos: todas las vistas de imágenes de la misma clase (supervised)
               o solo la otra vista del mismo sample (SimCLR, sin labels)
  - negativos: todo el resto

Loss por anchor i:
    ℓ_i = -1/|P(i)| · Σ_{p∈P(i)} log [ exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) ]

Estabilidad numérica: restamos el max de cada fila antes de exp (log-sum-exp trick).

En Co²L reutilizamos esta misma clase con el batch conjunto
(tarea actual + replay buffer), los labels se encargan del masking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    features: (B, n_views, feat_dim) — ya L2-normalizadas, típicamente n_views=2.
    labels:   (B,) enteros. Si None, cae al objetivo SimCLR (self-supervised).
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature      = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        features:    (B, n_views, feat_dim) — L2-normalizadas.
        labels:      (B,) enteros opcionales. None → SimCLR.
        anchor_mask: (B,) bool opcional. True = el sample actúa como anchor
                     (contribuye al numerador). False = solo aparece en el
                     denominador como negativo. None = todos son anchors.
        """
        if features.ndim != 3:
            raise ValueError(
                f"features debe ser (B, n_views, feat_dim), recibido {features.shape}"
            )

        B, n_views, feat_dim = features.shape
        device = features.device

        if B < 2:
            raise ValueError("SupConLoss necesita al menos 2 samples por batch.")

        # Aplanar a (2B, feat_dim): cada vista es una fila
        contrast = features.view(B * n_views, feat_dim)

        # Similitudes coseno (features ya L2-norm → dot product = cosine)
        sim = torch.matmul(contrast, contrast.T) / self.temperature  # (2B, 2B)

        # Máscara de positivos
        if labels is not None:
            # Supervised: positivos = todas las vistas de la misma clase
            labels_rep = labels.repeat_interleave(n_views)
            label_mask = torch.eq(
                labels_rep.unsqueeze(0), labels_rep.unsqueeze(1)
            ).float().to(device)
        else:
            # SimCLR: solo la otra vista del mismo sample
            label_mask = torch.zeros(B * n_views, B * n_views, device=device)
            for i in range(B):
                for vi in range(n_views):
                    for vj in range(n_views):
                        if vi != vj:
                            label_mask[i * n_views + vi, i * n_views + vj] = 1.0

        # Excluir diagonal (self-similarity) de positivos y denominador
        eye = torch.eye(B * n_views, device=device)
        pos_mask   = label_mask * (1 - eye)
        denom_mask = 1 - eye

        # Log-sum-exp estable: restar max por fila antes de exp
        sim_masked = sim - eye * 1e9
        max_sim    = sim_masked.max(dim=1, keepdim=True).values.detach()
        exp_sim    = torch.exp(sim - max_sim)

        # log p(par positivo)
        denom    = (exp_sim * denom_mask).sum(dim=1, keepdim=True)
        log_prob = (sim - max_sim) - torch.log(denom + 1e-9)

        # Promedio sobre positivos por anchor (evitando división por cero)
        n_pos      = pos_mask.sum(dim=1)
        valid      = n_pos > 0
        n_pos_safe = n_pos.clamp(min=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / n_pos_safe

        # Rescaleo por temperatura como en el código oficial
        loss = -(self.temperature / self.base_temperature) * mean_log_prob

        # Si se provee anchor_mask, solo promediamos sobre los anchors válidos
        if anchor_mask is not None:
            anchor_mask_rep = anchor_mask.to(device).repeat_interleave(n_views).bool()
            valid = valid & anchor_mask_rep

        loss = loss[valid].mean()

        return loss
