"""
Contrastive Continual Learning (Co²L) — full implementation.

Reference: Cha et al., "Co²L: Contrastive Continual Learning", ICCV 2021.
https://arxiv.org/abs/2106.14413

Combined objective
------------------
    L = L_SupCon  +  λ · L_distill  +  L_CE

  L_SupCon  — Supervised Contrastive loss on the JOINT batch:
                (a) two-view current-task samples
                (b) two views synthesised from single-view replay samples
              Pulls same-class representations together across both present
              and past tasks.

  L_distill — Asymmetric Distillation loss on the CURRENT-TASK batch only:
                for each sample i, treats z_curr_i as an anchor and
                z_prev_i (frozen teacher embedding) as the single positive;
                all other z_prev_j are negatives.
              Preserves the representation structure learned on past tasks.
              Active only on tasks t > 0 (teacher = snapshot before task t).

  L_CE      — CrossEntropy on view-0 of the current-task batch.
              Keeps the linear classification head aligned with the new task.

Joint batch construction
------------------------
The SupCon loader yields two-view batches (B, 2, C, H, W) for the current
task.  The replay buffer stores single-view tensors (saved with the standard
transform in end_task).  A second buffer view is synthesised at runtime via
random horizontal flip — sufficient augmentation for contrastive learning.

Replay is active from task 1 onward.  On task 0 the buffer is empty and the
training is pure SupCon + CE, matching the Stage-2 pre-training setup.

Buffer update
-------------
end_task() calls buffer.update_from_loader() with the standard (single-view)
task DataLoader.  Second views are generated lazily during training.
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
from losses.distillation import AsymmetricDistillationLoss
from losses.supcon import SupConLoss
from methods.base import BaseMethod
from models.backbone import Backbone

CO2L_LAMBDA = 1.0   # distillation weight λ
SUPCON_TEMP = 0.07


class Co2L(BaseMethod):
    """Contrastive Continual Learning — full Co²L implementation.

    Args:
        backbone:      Backbone instance (encoder + projector + classifier).
        device:        Torch device.
        seq_cifar:     SeqCIFAR10 instance — provides the two-view SupCon
                       loader for each task inside train_task().
        replay_buffer: Fixed-capacity buffer (reservoir sampling).
        co2l_lambda:   Weight λ for the distillation term.
        temperature:   SupCon / distillation temperature τ.
        lr:            SGD learning rate (0.5 as in the paper).
        momentum:      SGD momentum.
        weight_decay:  L2 regularisation.
        n_buf_samples: Replay samples mixed into each SupCon batch.
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

        self.supcon_loss  = SupConLoss(temperature=temperature)
        self.distill_loss = AsymmetricDistillationLoss(temperature=temperature)
        self._prev_model: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_task(self, task_id: int) -> None:
        """Snapshot the current model as the frozen teacher for distillation."""
        super().begin_task(task_id)
        if task_id > 0:
            self._prev_model = copy.deepcopy(self.backbone).to(self.device)
            self._prev_model.eval()
            for p in self._prev_model.parameters():
                p.requires_grad = False
            self._log(f"teacher snapshot taken for task={task_id}")
        else:
            self._prev_model = None

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
        """Train with SupCon + A-Distill + CE.

        Args:
            task_id:      0-based task index.
            train_loader: Standard single-view DataLoader for task_id.
                          Used for _validate_batch_labels and end_task;
                          the two-view SupCon loader is built internally.
            n_epochs:     Number of training epochs.

        Returns:
            {"supcon", "distill", "ce", "loss", "acc"} — one float per epoch.
            "distill" is 0.0 on task 0 (no teacher yet).
        """
        if self._current_task_id != task_id:
            self.begin_task(task_id)

        supcon_loader = self.seq_cifar.get_task_loader(task_id, train=True, supcon=True)

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

        log: Dict[str, list] = {
            "supcon": [], "distill": [], "ce": [], "loss": [], "acc": [],
        }

        for epoch in range(n_epochs):
            self.backbone.train()
            running = dict(supcon=0.0, distill=0.0, ce=0.0, loss=0.0)
            correct = total = 0

            for x2v, labels_cur in supcon_loader:
                # x2v: (B, 2, C, H, W)   labels_cur: (B,)
                self._validate_batch_labels(labels_cur, task_id)
                x2v        = x2v.to(self.device)
                labels_cur = labels_cur.to(self.device)
                B          = labels_cur.size(0)

                # ── 1. SupCon on joint batch ──────────────────────────────
                feats_cur = self.backbone.forward_supcon(x2v)   # (B, 2, proj_dim)

                if len(self.buffer) > 0:
                    buf_imgs, buf_labels = self.buffer.sample(self.n_buf_samples)
                    buf_imgs   = buf_imgs.to(self.device)
                    buf_labels = buf_labels.to(self.device)
                    buf_v2     = _random_flip(buf_imgs)
                    buf_2v     = torch.stack([buf_imgs, buf_v2], dim=1)  # (N,2,C,H,W)
                    feats_buf  = self.backbone.forward_supcon(buf_2v)    # (N,2,proj_dim)
                    joint_feats  = torch.cat([feats_cur, feats_buf], dim=0)
                    joint_labels = torch.cat([labels_cur, buf_labels],   dim=0)
                else:
                    joint_feats  = feats_cur
                    joint_labels = labels_cur

                supcon = self.supcon_loss(joint_feats, joint_labels)

                # ── 2. Asymmetric Distillation on current-task samples ────
                distill = self._compute_distillation(x2v[:, 0])   # view-0

                # ── 3. CE on current-task (view-0) ────────────────────────
                logits_cur = self.backbone(x2v[:, 0])
                ce         = ce_criterion(logits_cur, labels_cur)

                # ── 4. Combined loss ──────────────────────────────────────
                loss = supcon + self.co2l_lambda * distill + ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running["supcon"]  += supcon.item()  * B
                running["distill"] += distill.item() * B
                running["ce"]      += ce.item()      * B
                running["loss"]    += loss.item()    * B
                correct            += (logits_cur.argmax(1) == labels_cur).sum().item()
                total              += B

            scheduler.step()

            n = max(total, 1)
            for key in ("supcon", "distill", "ce", "loss"):
                log[key].append(running[key] / n)
            log["acc"].append(correct / n)
            self._epoch_log(task_id, epoch, n_epochs,
                            running["loss"] / n, correct / n)

        return log

    # ------------------------------------------------------------------
    # Distillation helper
    # ------------------------------------------------------------------

    def _compute_distillation(self, imgs: torch.Tensor) -> torch.Tensor:
        """Compute A-Distill between current and frozen-teacher encoder outputs.

        Uses backbone.encode() — L2-normalised encoder features (not the
        projector), so the distillation operates in the same space used for
        downstream classification.

        Args:
            imgs: (B, C, H, W) single-view current-task batch, on device.

        Returns:
            Scalar distillation loss; 0.0 if no teacher (task 0).
        """
        if self._prev_model is None:
            return torch.tensor(0.0, device=self.device)

        z_curr = self.backbone.encode(imgs)          # (B, feat_dim), L2-norm

        with torch.no_grad():
            z_prev = self._prev_model.encode(imgs)   # (B, feat_dim), frozen

        return self.distill_loss(z_curr, z_prev)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _random_flip(imgs: torch.Tensor) -> torch.Tensor:
    """Apply independent random horizontal flip to each image (p=0.5).

    Args:
        imgs: (B, C, H, W) tensor.

    Returns:
        (B, C, H, W) tensor.
    """
    mask = torch.rand(imgs.size(0), device=imgs.device) > 0.5
    out  = imgs.clone()
    if mask.any():
        out[mask] = torch.flip(imgs[mask], dims=[-1])
    return out
