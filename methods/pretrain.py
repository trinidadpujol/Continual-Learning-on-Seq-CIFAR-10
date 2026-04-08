"""
SupCon pre-training and linear probe for the backbone encoder.

Two-stage workflow
------------------
1. pretrain_supcon()      — train encoder + projector with SupConLoss
2. train_linear_probe()   — freeze encoder, train classifier with CE loss

Typical use
-----------
    # Stage 1 – contrastive pre-training
    pt_history = pretrain_supcon(
        backbone, seq_cifar, task_id=0, n_epochs=200, device=device,
        checkpoint_dir="checkpoints",
    )

    # Stage 2 – linear probe (encoder stays frozen)
    probe_history = train_linear_probe(
        backbone, seq_cifar, task_id=0, n_epochs=100, device=device,
        checkpoint_dir="checkpoints",
    )
    print(probe_history["test_acc"])          # float
    print(probe_history["checkpoint_path"])   # path to saved .pt file
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
    save_stages: bool = False,
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
        save_stages:      If True, additionally save snapshots at epoch 0
                          (before any training) and epoch n_epochs//2 (midpoint).
                          Used by the visualisation stage to produce init/mid/final
                          embedding projections.

    Returns:
        dict with:
          ``loss_history``    – list[float], mean SupCon loss per epoch.
          ``checkpoint_path`` – str, path to the final saved checkpoint.
          ``stage_paths``     – dict {"init": str, "mid": str, "final": str}
                                (only present when save_stages=True).
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

    # ── Stage snapshot: epoch 0 (before any training) ─────────────────────
    stage_paths: Dict[str, str] = {}
    if save_stages:
        init_path = os.path.join(checkpoint_dir, f"supcon_stage_init_task{task_id}.pt")
        _save_checkpoint(backbone, 0, [], init_path, temperature)
        stage_paths["init"] = init_path
        logger.info("[pretrain_supcon] stage checkpoint (init) → %s", init_path)

    mid_epoch = n_epochs // 2

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

        # ── Stage snapshot: midpoint ──────────────────────────────────────
        if save_stages and (epoch + 1) == mid_epoch:
            mid_path = os.path.join(checkpoint_dir, f"supcon_stage_mid_task{task_id}.pt")
            _save_checkpoint(backbone, epoch + 1, loss_history, mid_path, temperature)
            stage_paths["mid"] = mid_path
            logger.info("[pretrain_supcon] stage checkpoint (mid) → %s", mid_path)

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

    result: Dict[str, object] = {
        "loss_history":    loss_history,
        "checkpoint_path": final_path,
    }
    if save_stages:
        stage_paths["final"] = final_path
        result["stage_paths"] = stage_paths
    return result


def train_linear_probe(
    backbone: Backbone,
    seq_cifar: SeqCIFAR10,
    task_id: int = 0,
    n_epochs: int = 100,
    lr: float = 1.0,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, object]:
    """Freeze the encoder and train a linear classification head with CE loss.

    Must be called after pretrain_supcon().  The encoder is frozen before
    training begins and the freeze is verified every epoch — a RuntimeError
    is raised if any encoder gradient somehow appears.

    The classifier is reset (Kaiming-uniform init) at the start so that
    it trains purely from the frozen features and does not inherit any
    gradient signal from the contrastive phase.

    Args:
        backbone:        Backbone with a pre-trained encoder.
        seq_cifar:       SeqCIFAR10 dataset helper.
        task_id:         Task whose two classes to use for probe training/eval.
        n_epochs:        Number of training epochs for the linear head.
        lr:              Initial SGD learning rate (cosine-annealed).
        momentum:        SGD momentum.
        weight_decay:    L2 regularisation (typically 0 for a linear probe).
        device:          Training device.
        checkpoint_dir:  Directory for the saved checkpoint.

    Returns:
        dict with:
          ``loss_history``      – list[float], mean CE loss per epoch.
          ``train_acc_history`` – list[float], training accuracy per epoch.
          ``test_acc``          – float, final accuracy on the task test split.
          ``checkpoint_path``   – str, path to the saved checkpoint.

    Raises:
        RuntimeError: if any encoder parameter has requires_grad=True during
                      training (encoder-freeze sanity check).
    """
    if device is None:
        device = next(backbone.parameters()).device

    backbone = backbone.to(device)

    # ── 1. Freeze encoder; reset classifier ───────────────────────────────
    backbone.freeze_encoder()
    backbone.reset_classifier()

    if not backbone.is_encoder_frozen:
        raise RuntimeError(
            "freeze_encoder() was called but is_encoder_frozen is still False. "
            "Check Backbone.freeze_encoder() implementation."
        )

    c0_name, c1_name = seq_cifar.get_class_names(task_id)
    logger.info(
        "[train_linear_probe] task=%d  classes=(%s, %s)  n_epochs=%d  lr=%.4f",
        task_id, c0_name, c1_name, n_epochs, lr,
    )

    # ── 2. Data ────────────────────────────────────────────────────────────
    train_loader = seq_cifar.get_task_loader(task_id, train=True,  supcon=False)
    test_loader  = seq_cifar.get_task_loader(task_id, train=False)

    # ── 3. Optimise only the classifier head ──────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        backbone.classifier.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── 4. Training loop ───────────────────────────────────────────────────
    loss_history:      List[float] = []
    train_acc_history: List[float] = []

    for epoch in range(n_epochs):
        # Sanity-check: encoder must stay frozen every epoch
        if not backbone.is_encoder_frozen:
            raise RuntimeError(
                f"Encoder became unfrozen at epoch {epoch + 1}. "
                "Probe training aborted to prevent encoder contamination."
            )

        backbone.eval()   # BN / Dropout in eval mode (encoder frozen)
        backbone.classifier.train()   # only the head in train mode

        epoch_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                h = backbone._extract_features(x)   # frozen, no grad needed

            logits = backbone.classifier(h)         # only head gets gradients
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)

        scheduler.step()

        mean_loss  = epoch_loss / max(len(train_loader), 1)
        train_acc  = correct / max(total, 1)
        loss_history.append(mean_loss)
        train_acc_history.append(train_acc)

        logger.info(
            "[train_linear_probe] epoch=%d/%d  loss=%.4f  train_acc=%.3f  lr=%.6f",
            epoch + 1, n_epochs, mean_loss, train_acc,
            scheduler.get_last_lr()[0],
        )

    # ── 5. Final evaluation on the test split ─────────────────────────────
    backbone.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = backbone(x)
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)
    test_acc = correct / max(total, 1)

    logger.info(
        "[train_linear_probe] task=%d  test_acc=%.3f  (encoder frozen: %s)",
        task_id, test_acc, backbone.is_encoder_frozen,
    )

    # ── 6. Save checkpoint ─────────────────────────────────────────────────
    ckpt_path = os.path.join(checkpoint_dir, f"linear_probe_task{task_id}.pt")
    torch.save(
        {
            "backbone_state_dict": backbone.state_dict(),
            "loss_history":        loss_history,
            "train_acc_history":   train_acc_history,
            "test_acc":            test_acc,
            "task_id":             task_id,
            "config": {
                "n_epochs":    n_epochs,
                "lr":          lr,
                "feat_dim":    backbone.feat_dim,
                "num_classes": backbone.num_classes,
            },
        },
        ckpt_path,
    )
    logger.info("[train_linear_probe] checkpoint saved → %s", ckpt_path)

    return {
        "loss_history":      loss_history,
        "train_acc_history": train_acc_history,
        "test_acc":          test_acc,
        "checkpoint_path":   ckpt_path,
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
