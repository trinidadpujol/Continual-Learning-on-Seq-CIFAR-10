"""
Backbone para Seq-CIFAR-10: ResNet-18 adaptado a 32x32 con cabeza de proyección
y clasificador lineal.

Arquitectura:
    imagen (B,3,32,32) → encoder → h (B,512)
                                     ├→ projector (MLP 2 capas) → z (B,128) L2-norm  [SupCon]
                                     └→ classifier (lineal)     → logits (B,10)       [CE]

Modificaciones al ResNet-18 estándar para imágenes 32x32:
  - conv1: 7x7 stride-2 → 3x3 stride-1  (no perder resolución espacial)
  - maxpool: → Identity                   (evitar downsampling excesivo)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Backbone(nn.Module):
    """ResNet-18 con projection head y clasificador lineal.

    Se usa en dos modos:
      - Pre-entrenamiento: encoder + projector con SupConLoss (encoder frozen)
      - Fine-tuning continuo: encoder + classifier con CrossEntropyLoss
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

        # Encoder: ResNet-18 adaptado para 32x32
        base = resnet18(weights=None)
        # Reemplazamos conv1 para no reducir resolución en imágenes pequeñas
        base.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Sin maxpool: en 32x32 ya sería demasiado downsampling
        base.maxpool = nn.Identity()
        # Sacamos la FC final, usamos nuestras propias cabezas
        self.encoder = nn.Sequential(*list(base.children())[:-1])

        # Projection head (Khosla et al. 2020): MLP de 2 capas
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

        # Clasificador lineal: usado como linear probe y para fine-tuning
        self.classifier = nn.Linear(feat_dim, num_classes)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder forward, devuelve features sin normalizar."""
        return self.encoder(x).flatten(1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Features L2-normalizadas del encoder (para distilación y visualización)."""
        return F.normalize(self._extract_features(x), dim=1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Salida L2-normalizada del projection head (para SupConLoss)."""
        h = self._extract_features(x)
        return F.normalize(self.projector(h), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits de clasificación para CrossEntropyLoss."""
        return self.classifier(self._extract_features(x))

    def forward_supcon(self, x: torch.Tensor) -> torch.Tensor:
        """Forward para batch de dos vistas (B, 2, C, H, W) → (B, 2, proj_dim).

        Juntamos las dos vistas en un batch para procesarlas en paralelo y
        después las separamos de vuelta al formato que espera SupConLoss.
        """
        B = x.size(0)
        x_flat = x.view(B * 2, *x.shape[2:])
        z_flat = self.project(x_flat)       # (2B, proj_dim)
        return z_flat.view(B, 2, self.proj_dim)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    @property
    def is_encoder_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.encoder.parameters())

    def reset_classifier(self) -> None:
        """Reinicializa la cabeza de clasificación con Kaiming-uniform.

        Llamar antes de entrenar el linear probe para que arranque de cero
        y no herede ningún gradiente de la fase contrastiva.
        """
        nn.init.kaiming_uniform_(self.classifier.weight, nonlinearity="linear")
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def __repr__(self) -> str:
        frozen = "frozen" if self.is_encoder_frozen else "trainable"
        return (
            f"Backbone(\n"
            f"  encoder  : ResNet-18 para 32x32  [{frozen}]\n"
            f"  feat_dim : {self.feat_dim}\n"
            f"  projector: Linear({self.feat_dim}, {self.feat_dim}) → ReLU "
            f"→ Linear({self.feat_dim}, {self.proj_dim}) → L2-norm\n"
            f"  classifier: Linear({self.feat_dim}, {self.num_classes})\n"
            f")"
        )
