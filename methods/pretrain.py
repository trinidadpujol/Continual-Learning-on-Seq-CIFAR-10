"""
SupCon pre-training for the backbone encoder.

Trains only the encoder + projection head (no classifier) with
Supervised Contrastive Loss on a single task's two-view augmented data.
After this step the encoder weights are saved as a reusable checkpoint.
The classifier head is intentionally left uninitialised until the linear
probe phase so it can be reset cleanly with backbone.reset_classifier().

Typical use
-----------
    history = pretrain_supcon(
        backbone, seq_cifar,
        task_id=0,
        n_epochs=200,
        device=device,
        checkpoint_dir="checkpoints",
    )
    print(history["loss_history"])           # list[float], one per epoch
    print(history["checkpoint_path"])        # path to saved .pt file
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from data.dataset import SeqCIFAR10
from losses.supcon import SupConLoss
from models.backbone import Backbone

logger = logging.getLogger(__name__)


def pretrain_supcon(
    backbone: Backbone,
    seq_cifar: SeqCIFAR10,
    task_id: int = 0,
    n_epochs: int = 200,
    lr: float = 0.5,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 0,
) -> Dict[str, object]:
    """Pre-train the backbone encoder with Supervised Contrastive Loss.

    Only the encoder and projection head participate in this phase.
    The classifier parameters are not updated and can be reset afterwards
    via backbone.reset_classifier() before attaching a linear probe.

    Args:
        backbone:         Backbone instance (encoder + projector + classifier).
        seq_cifar:        SeqCIFAR10 dataset helper.
        task_id:          Which task to use for pre-training (default 0).
        n_epochs:         Number of training epochs.
        lr:               Initial SGD learning rate.
        momentum:         SGD momentum.
        weight_decay:     L2 regularisation.
        temperature:      SupCon loss temperature τ.
        device:           Training device.  Defaults to backbone's current device.
        checkpoint_dir:   Directory to save checkpoints.  Created if absent.
        checkpoint_every: Save an intermediate checkpoint every this many epochs.
                          0 (default) saves only the final checkpoint.

    Returns:
        dict with:
          ``loss_history``    – list[float], mean SupCon loss per epoch.
          ``checkpoint_path`` – str, path to the final saved checkpoint.
    """
    if device is None:
        device = next(backbone.parameters()).device

    backbone = backbone.to(device)
    backbone.train()

    # ── Data ──────────────────────────────────────────────────────────────
    loader = seq_cifar.get_task_loader(task_id, train=True, supcon=True)
    c0, c1 = seq_cifar.get_class_names(task_id)
    logger.info(
        "[pretrain_supcon] task=%d  classes=(%s, %s)  n_epochs=%d  lr=%.4f",
        task_id, c0, c1, n_epochs, lr,
    )

    # ── Criterion ─────────────────────────────────────────────────────────
    criterion = SupConLoss(temperature=temperature).to(device)

    # ── Optimise only encoder + projector (not the classifier head) ───────
    supcon_params = list(backbone.encoder.parameters()) + \
                    list(backbone.projector.parameters())
    optimizer = torch.optim.SGD(
        supcon_params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    # ── Checkpoint dir ─────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────────────
    loss_history: List[float] = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for x, labels in loader:
            # x: (B, 2, C, H, W)  labels: (B,)
            x      = x.to(device)
            labels = labels.to(device)

            features = backbone.forward_supcon(x)          # (B, 2, proj_dim)
            loss     = criterion(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        mean_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(mean_loss)

        logger.info(
            "[pretrain_supcon] epoch=%d/%d  loss=%.4f  lr=%.6f",
            epoch + 1, n_epochs, mean_loss,
            scheduler.get_last_lr()[0],
        )

        # ── Intermediate checkpoint ────────────────────────────────────────
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(
                checkpoint_dir, f"supcon_task{task_id}_epoch{epoch + 1}.pt"
            )
            _save_checkpoint(backbone, epoch + 1, loss_history, ckpt_path, temperature)
            logger.info("[pretrain_supcon] intermediate checkpoint → %s", ckpt_path)

    # ── Final checkpoint ────────────────────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, f"supcon_pretrained_task{task_id}.pt")
    _save_checkpoint(backbone, n_epochs, loss_history, final_path, temperature)
    logger.info("[pretrain_supcon] final checkpoint → %s", final_path)

    return {
        "loss_history":    loss_history,
        "checkpoint_path": final_path,
    }


def load_pretrained_backbone(
    checkpoint_path: str,
    backbone: Backbone,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Load a backbone state dict saved by pretrain_supcon().

    Args:
        checkpoint_path: Path to a .pt file saved by pretrain_supcon().
        backbone:        Backbone instance to load weights into.
        device:          Target device.  Defaults to the backbone's device.

    Returns:
        The checkpoint metadata dict (epoch, loss_history, config).
    """
    if device is None:
        device = next(backbone.parameters()).device

    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.to(device)

    logger.info(
        "[load_pretrained_backbone] loaded from %s  (epoch=%d)",
        checkpoint_path, ckpt.get("epoch", -1),
    )
    return ckpt


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(
    backbone: Backbone,
    epoch: int,
    loss_history: List[float],
    path: str,
    temperature: float,
) -> None:
    torch.save(
        {
            "backbone_state_dict": backbone.state_dict(),
            "epoch":               epoch,
            "loss_history":        loss_history,
            "config": {
                "temperature": temperature,
                "feat_dim":    backbone.feat_dim,
                "proj_dim":    backbone.proj_dim,
                "num_classes": backbone.num_classes,
            },
        },
        path,
    )
