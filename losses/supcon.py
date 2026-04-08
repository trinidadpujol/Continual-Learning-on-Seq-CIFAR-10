"""
Supervised Contrastive Loss (SupCon).

Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
https://arxiv.org/abs/2004.11362

Algorithm (per anchor i)
------------------------
Given N samples each with n_views augmented views, we work on the
2·N × feat_dim "contrast" matrix — every view of every sample is an anchor
in turn.

For anchor i:
  - positives: all other views of the same image AND all views of images
               with the same class label (Supervised) — or only the other
               augmented view of the same image (SimCLR / label-free).
  - negatives: all remaining views.

Loss for anchor i:
    ℓ_i = -1/|P(i)| · Σ_{p∈P(i)} log [
        exp(z_i · z_p / τ) /
        Σ_{a≠i} exp(z_i · z_a / τ)
    ]

Numerical stability: we subtract max(logits) before exp — standard
log-sum-exp trick.  Diagonal self-similarity is masked out before the
denominator sum.

Reuse in Co²L
-------------
Co²L reuses SupConLoss directly with the combined
[current-task batch + replay-buffer samples] feature matrix.  The
`labels` argument handles the positive-pair mask automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised (or self-supervised) Contrastive Loss.

    Args:
        temperature: Logit-scale temperature τ (default 0.07, as in the paper).
        base_temperature: Reference temperature used to re-scale the loss
            magnitude (default 0.07). Set equal to temperature to disable
            re-scaling.
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature      = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the SupCon loss.

        Args:
            features: (B, n_views, feat_dim) — **L2-normalised** projection
                      vectors, e.g. from Backbone.forward_supcon().
                      Typically n_views=2 (two augmented views).
            labels:   (B,) integer class labels.  If None, falls back to the
                      SimCLR self-supervised objective (only same-image pairs
                      are positive).

        Returns:
            Scalar loss (mean over all valid anchors).

        Raises:
            ValueError: if features is not 3-dimensional or B < 2.
        """
        if features.ndim != 3:
            raise ValueError(
                f"features must be (B, n_views, feat_dim), got shape {features.shape}"
            )

        B, n_views, feat_dim = features.shape
        device = features.device

        if B < 2:
            raise ValueError("SupConLoss requires at least 2 samples per batch.")

        # ── 1. Build the (2B, feat_dim) contrast matrix ─────────────────────
        # Flatten: each of the B×n_views views becomes one row.
        contrast = features.view(B * n_views, feat_dim)   # (2B, feat_dim)

        # ── 2. Pairwise cosine-similarity logits ─────────────────────────────
        # Features are already L2-normalised → dot product = cosine similarity.
        sim = torch.matmul(contrast, contrast.T) / self.temperature  # (2B, 2B)

        # ── 3. Mask: which pairs are positives? ──────────────────────────────
        if labels is not None:
            # Supervised: positives are all views sharing the same class label.
            # Repeat each label n_views times to align with the contrast matrix.
            labels_rep = labels.repeat_interleave(n_views)  # (2B,)
            label_mask = torch.eq(
                labels_rep.unsqueeze(0), labels_rep.unsqueeze(1)
            ).float().to(device)                             # (2B, 2B)
        else:
            # SimCLR: only the other augmented view of the same image is positive.
            # Indices 2i and 2i+1 are the two views of sample i.
            label_mask = torch.zeros(B * n_views, B * n_views, device=device)
            for i in range(B):
                for vi in range(n_views):
                    for vj in range(n_views):
                        if vi != vj:
                            label_mask[i * n_views + vi, i * n_views + vj] = 1.0

        # Exclude self-similarity from both positives and negatives.
        eye = torch.eye(B * n_views, device=device)
        pos_mask = label_mask * (1 - eye)   # same-label pairs, no diagonal
        neg_mask = 1 - label_mask           # different-label pairs (includes off-diag)
        # All valid denominator entries: everyone except self.
        denom_mask = 1 - eye               # (2B, 2B)

        # ── 4. Numerically stable log-sum-exp ────────────────────────────────
        # Subtract row max before exp to prevent overflow.
        # Mask diagonal to -inf so it never contributes to the denominator.
        sim_masked = sim - eye * 1e9            # diagonal → −∞ for denominator
        max_sim    = sim_masked.max(dim=1, keepdim=True).values.detach()
        exp_sim    = torch.exp(sim - max_sim)   # (2B, 2B)

        # ── 5. Log-probability of each pair being a positive ─────────────────
        # log_prob[i, j] = sim[i,j] - log( Σ_{k≠i} exp(sim[i,k]) )
        denom      = (exp_sim * denom_mask).sum(dim=1, keepdim=True)   # (2B, 1)
        log_prob   = (sim - max_sim) - torch.log(denom + 1e-9)         # (2B, 2B)

        # ── 6. Mean log-probability over positive pairs for each anchor ───────
        n_pos = pos_mask.sum(dim=1)                           # (2B,) positive count
        valid = n_pos > 0                                     # anchors with ≥1 positive

        # Avoid division by zero for anchors that have no positives.
        n_pos_safe = n_pos.clamp(min=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / n_pos_safe  # (2B,)

        # ── 7. Final loss ────────────────────────────────────────────────────
        # Optionally re-scale by temperature ratio (as in the official code).
        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        loss = loss[valid].mean()

        return loss
