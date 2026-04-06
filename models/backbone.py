"""
Backbone architecture for continual learning on Seq-CIFAR-10.

Structure
---------
                        ┌─────────────┐
    image (B,3,32,32) ─►│   encoder   │─► features h  (B, feat_dim)
                        └─────────────┘         │
                                         ┌──────┴──────┐
                                         ▼             ▼
                                   projector      classifier
                                  (2-layer MLP)  (linear layer)
                                         │             │
                                         ▼             ▼
                              z (B, proj_dim)    logits (B, num_classes)
                           L2-normalised         used with CrossEntropyLoss

Encoder: ResNet-18 adapted for 32×32 inputs.
  - 7×7 stride-2 conv  → 3×3 stride-1 conv   (keeps spatial resolution)
  - MaxPool             → Identity            (avoids over-downsampling)

Projector: used only during SupCon pre-training.
    Linear(feat_dim, feat_dim) → ReLU → Linear(feat_dim, proj_dim)

Classifier: linear probe attached after pre-training (encoder frozen),
  or fine-tuned jointly during continual learning stages.

Forward modes
-------------
    backbone(x)                        → logits             (classification)
    backbone.encode(x)                 → L2-normalised h    (embeddings)
    backbone.project(x)                → L2-normalised z    (contrastive)
    backbone.forward_supcon(x_2views)  → (B,2,proj_dim) z   (SupCon training)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Backbone(nn.Module):
    """
    ResNet-18 encoder with a projection head and a linear classifier.

    Args:
        num_classes: Number of output classes for the classifier (10 for CIFAR-10).
        proj_dim:    Output dimension of the projection head.
        feat_dim:    Feature dimension produced by the encoder (512 for ResNet-18).
    """

    def __init__(
        self,
        num_classes: int = 10,
        proj_dim: int = 128,
        feat_dim: int = 512,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.proj_dim    = proj_dim
        self.feat_dim    = feat_dim

        # ------------------------------------------------------------------
        # Encoder: ResNet-18 adapted for 32×32 CIFAR images
        # ------------------------------------------------------------------
        base = resnet18(weights=None)
        # Replace 7×7 stride-2 conv with 3×3 stride-1 so 32×32 inputs keep
        # spatial resolution after the first layer.
        base.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the max-pool that would halve 32→16; use identity instead.
        base.maxpool = nn.Identity()
        # Drop the final FC layer — we attach our own heads.
        self.encoder = nn.Sequential(*list(base.children())[:-1])

        # ------------------------------------------------------------------
        # Projection head  (SupCon pre-training only)
        # 2-layer MLP as in Khosla et al. 2020.
        # ------------------------------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

        # ------------------------------------------------------------------
        # Linear classification head
        # Used both as a linear probe (encoder frozen) and for fine-tuning.
        # ------------------------------------------------------------------
        self.classifier = nn.Linear(feat_dim, num_classes)

    # ------------------------------------------------------------------
    # Core forward passes
    # ------------------------------------------------------------------

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder and return raw, un-normalised feature vectors.

        Args:
            x: (B, C, H, W) image tensor.

        Returns:
            h: (B, feat_dim) raw feature vectors.
        """
        return self.encoder(x).flatten(1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised feature embeddings.

        Use this for distance-based operations (e.g. distillation loss,
        nearest-neighbour retrieval, t-SNE visualisation).

        Returns:
            (B, feat_dim) unit-norm vectors.
        """
        return F.normalize(self._extract_features(x), dim=1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised projection head outputs.

        Use this when computing the Supervised Contrastive loss.

        Returns:
            (B, proj_dim) unit-norm vectors.
        """
        h = self._extract_features(x)
        return F.normalize(self.projector(h), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return classification logits.

        Use this with CrossEntropyLoss during fine-tuning or the linear probe.

        Returns:
            (B, num_classes) unnormalised logits.
        """
        return self.classifier(self._extract_features(x))

    def forward_supcon(self, x: torch.Tensor) -> torch.Tensor:
        """Project a two-view batch for Supervised Contrastive training.

        Accepts the stacked two-view tensors produced by TwoViewTransform
        (shape B×2×C×H×W) and returns the L2-normalised projections in the
        (B, n_views, proj_dim) shape expected by SupConLoss.

        Args:
            x: (B, 2, C, H, W) two-view batch.

        Returns:
            (B, 2, proj_dim) L2-normalised projection vectors.
        """
        B = x.size(0)
        # Merge the two views into a single batch, forward together,
        # then split back.  (B, 2, C, H, W) → (2B, C, H, W) → (B, 2, proj_dim)
        x_flat = x.view(B * 2, *x.shape[2:])
        z_flat = self.project(x_flat)       # (2B, proj_dim)
        return z_flat.view(B, 2, self.proj_dim)

    # ------------------------------------------------------------------
    # Encoder freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (used before training the linear probe)."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters for end-to-end fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = True

    @property
    def is_encoder_frozen(self) -> bool:
        """True if all encoder parameters have requires_grad=False."""
        return all(not p.requires_grad for p in self.encoder.parameters())

    # ------------------------------------------------------------------
    # Classifier head management
    # ------------------------------------------------------------------

    def reset_classifier(self) -> None:
        """Re-initialise the classifier head with Kaiming-uniform weights.

        Call this after SupCon pre-training, just before attaching and
        training the linear probe — ensures the probe starts from scratch
        rather than from a randomly initialised head that was never trained.
        """
        nn.init.kaiming_uniform_(self.classifier.weight, nonlinearity="linear")
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        frozen = "frozen" if self.is_encoder_frozen else "trainable"
        return (
            f"Backbone(\n"
            f"  encoder  : ResNet-18 for 32×32  [{frozen}]\n"
            f"  feat_dim : {self.feat_dim}\n"
            f"  projector: Linear({self.feat_dim}, {self.feat_dim}) → ReLU "
            f"→ Linear({self.feat_dim}, {self.proj_dim}) → L2-norm\n"
            f"  classifier: Linear({self.feat_dim}, {self.num_classes})\n"
            f")"
        )
