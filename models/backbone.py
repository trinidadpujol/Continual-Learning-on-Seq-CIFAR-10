"""
Backbone encoder for continual learning on Seq-CIFAR-10.

Architecture: ResNet-18 adapted for 32x32 inputs, with a projection head
for contrastive learning and a separate linear classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class Backbone(nn.Module):
    """
    ResNet-18 encoder with a projection head (for SupCon) and an
    optional linear classification head (frozen encoder + linear probe).

    Args:
        num_classes: Total number of output classes (10 for CIFAR-10).
        proj_dim:    Output dimension of the projection head.
        feat_dim:    Dimension of the encoder's feature vector (512 for ResNet-18).
    """

    def __init__(
        self,
        num_classes: int = 10,
        proj_dim: int = 128,
        feat_dim: int = 512,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        # ResNet-18 adapted for 32x32 (replace 7x7 conv with 3x3, remove maxpool)
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # remove fc

        # Projection head (used during SupCon pre-training)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

        # Classification head (attached after pre-training, encoder frozen)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised feature embeddings."""
        h = self.encoder(x).flatten(1)
        return nn.functional.normalize(h, dim=1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised projection head outputs (for contrastive loss)."""
        h = self.encoder(x).flatten(1)
        z = self.projector(h)
        return nn.functional.normalize(z, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits from the linear classification head."""
        h = self.encoder(x).flatten(1)
        return self.classifier(h)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
