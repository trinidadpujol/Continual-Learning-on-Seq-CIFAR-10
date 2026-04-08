"""
Contrastive Continual Learning (Co²L) — v1 (SupCon + replay, no distillation yet).

Reference: Cha et al., "Co²L: Contrastive Continual Learning", ICCV 2021.
https://arxiv.org/abs/2106.14413

Architecture
------------
Co²L trains with two simultaneous objectives:

  1. Supervised Contrastive Loss on a joint batch of
       (a) two-view augmented current-task samples, and
       (b) two views synthesised from single-view replay samples.
     This pulls together same-class representations from both the current
     task and past tasks.

  2. Cross-Entropy on the current-task batch only, using the classifier head.
     (The SupCon phase trains encoder + projector; CE ensures the classifier
     stays aligned with the new task's classes.)

  3. [v2, not yet] Asymmetric Distillation — aligns student encoder
     representations with a frozen teacher snapshot taken before each task.

Joint batch construction
------------------------
ContinualTrainer provides a two-view SupCon loader for the current task.
The replay buffer stores single-view tensors (saved from the standard
transform at end_task time). During training we synthesise a second view for
each buffer sample with a lightweight random horizontal flip.

Replay is active from task 1 onward. On task 0 the buffer is empty and the
training degenerates to pure SupCon + CE (same as the Stage-2 pre-training).

Buffer update
-------------
end_task() calls buffer.update_from_loader() with the standard (single-view)
task DataLoader. This stores normalised (C,H,W) tensors, not two-view ones.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.buffer import ReplayBuffer
from data.dataset import SeqCIFAR10
from losses.supcon import SupConLoss
from methods.base import BaseMethod
from models.backbone import Backbone

CO2L_LAMBDA     = 1.0    # distillation weight (unused in v1; reserved for v2)
SUPCON_TEMP     = 0.07


class Co2L(BaseMethod):
    """Contrastive Continual Learning (v1: SupCon + replay, no distillation).

    Args:
        backbone:      Backbone instance.
        device:        Torch device.
        seq_cifar:     SeqCIFAR10 instance — used to obtain the two-view
                       SupCon loader for each task inside train_task().
        replay_buffer: Fixed-capacity buffer (reservoir sampling).
        co2l_lambda:   Weight for the distillation term (reserved for v2).
        temperature:   SupCon temperature τ.
        lr:            SGD learning rate.
        momentum:      SGD momentum.
        weight_decay:  L2 regularisation.
        n_buf_samples: Number of replay samples mixed into each SupCon batch.
    """

    uses_replay: bool = True

    def __init__(
        self,
        backbone: Backbone,
        device: torch.device,
        seq_cifar: SeqCIFAR10,
        replay_buffer: ReplayBuffer,
        co2l_lambda: float = CO2L_LAMBDA,
        temperature: float = SUPCON_TEMP,
        lr: float = 0.5,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        n_buf_samples: int = 64,
    ):
        super().__init__(backbone, device)
        self.seq_cifar     = seq_cifar
        self.buffer        = replay_buffer
        self.co2l_lambda   = co2l_lambda
        self.temperature   = temperature
        self.lr            = lr
        self.momentum      = momentum
        self.weight_decay  = weight_decay
        self.n_buf_samples = n_buf_samples

        self.supcon_loss = SupConLoss(temperature=temperature)
        self._prev_model: Optional[nn.Module] = None   # reserved for v2

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_task(self, task_id: int) -> None:
        """Snapshot the current model as a frozen teacher (v2 distillation)."""
        super().begin_task(task_id)
        if task_id > 0:
            self._prev_model = copy.deepcopy(self.backbone).to(self.device)
            self._prev_model.eval()
            for p in self._prev_model.parameters():
                p.requires_grad = False
            self._log(f"teacher snapshot taken for task={task_id}")

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Update replay buffer with single-view samples from the finished task."""
        super().end_task(task_id, train_loader)
        self.buffer.update_from_loader(train_loader, task_id)
        self._log(f"buffer updated — {self.buffer}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_epochs: int = 500,
    ) -> Dict[str, list]:
        """Train with SupCon on the joint batch (current + replay) and CE on current.

        Args:
            task_id:      0-based task index.
            train_loader: DataLoader for task_id (standard single-view, for CE
                          and for the _validate_batch_labels check).
                          The two-view SupCon loader is built internally via
                          seq_cifar.get_task_loader(task_id, supcon=True).
            n_epochs:     Number of epochs.

        Returns:
            {"supcon": [...], "ce": [...], "loss": [...], "acc": [...]}
            One entry per epoch.
        """
        if self._current_task_id != task_id:
            self.begin_task(task_id)

        # ── Two-view loader for SupCon ────────────────────────────────────
        supcon_loader = self.seq_cifar.get_task_loader(task_id, train=True, supcon=True)

        # ── Optimiser: encoder + projector + classifier ───────────────────
        optimizer = torch.optim.SGD(
            self.backbone.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )
        ce_criterion = nn.CrossEntropyLoss()

        log: Dict[str, list] = {"supcon": [], "ce": [], "loss": [], "acc": []}

        for epoch in range(n_epochs):
            self.backbone.train()
            running_supcon = 0.0
            running_ce     = 0.0
            running_loss   = 0.0
            correct = total = 0
            n_batches = 0

            for (x2v, labels_cur) in supcon_loader:
                # x2v: (B, 2, C, H, W)  labels_cur: (B,)
                self._validate_batch_labels(labels_cur, task_id)

                x2v        = x2v.to(self.device)
                labels_cur = labels_cur.to(self.device)

                # ── Build joint batch for SupCon ──────────────────────────
                feats_cur = self.backbone.forward_supcon(x2v)   # (B, 2, proj_dim)

                if len(self.buffer) > 0:
                    buf_imgs, buf_labels = self.buffer.sample(self.n_buf_samples)
                    buf_imgs   = buf_imgs.to(self.device)       # (N, C, H, W)
                    buf_labels = buf_labels.to(self.device)

                    # Synthesise a second view for buffer samples via random flip
                    buf_v2    = _random_flip(buf_imgs)
                    buf_2v    = torch.stack([buf_imgs, buf_v2], dim=1)  # (N,2,C,H,W)
                    feats_buf = self.backbone.forward_supcon(buf_2v)    # (N,2,proj_dim)

                    joint_feats  = torch.cat([feats_cur, feats_buf], dim=0)  # (B+N,2,d)
                    joint_labels = torch.cat([labels_cur, buf_labels], dim=0) # (B+N,)
                else:
                    joint_feats  = feats_cur
                    joint_labels = labels_cur

                supcon = self.supcon_loss(joint_feats, joint_labels)

                # ── CE on current-task single-view ────────────────────────
                # Use the first view (view 0) of the two-view batch for CE.
                logits_cur = self.backbone(x2v[:, 0])           # (B, num_classes)
                ce         = ce_criterion(logits_cur, labels_cur)

                loss = supcon + ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                B = labels_cur.size(0)
                running_supcon += supcon.item() * B
                running_ce     += ce.item()     * B
                running_loss   += loss.item()   * B
                correct        += (logits_cur.argmax(1) == labels_cur).sum().item()
                total          += B
                n_batches      += 1

            scheduler.step()

            n = max(total, 1)
            log["supcon"].append(running_supcon / n)
            log["ce"].append(running_ce         / n)
            log["loss"].append(running_loss     / n)
            log["acc"].append(correct           / n)
            self._epoch_log(task_id, epoch, n_epochs,
                            running_loss / n, correct / n)

        return log


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _random_flip(imgs: torch.Tensor) -> torch.Tensor:
    """Apply random horizontal flip to each image independently.

    Args:
        imgs: (B, C, H, W) tensor.

    Returns:
        (B, C, H, W) tensor with each image independently flipped with p=0.5.
    """
    mask = torch.rand(imgs.size(0), device=imgs.device) > 0.5  # (B,)
    out  = imgs.clone()
    if mask.any():
        out[mask] = torch.flip(imgs[mask], dims=[-1])
    return out
