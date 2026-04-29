"""
Contrastive Continual Learning (Co²L).
Cha et al., ICCV 2021. https://arxiv.org/abs/2106.14413

Objetivo combinado:
    L = L_SupCon  +  λ · L_distill  +  L_CE

  L_SupCon  — Contrastive supervisado sobre el batch CONJUNTO:
                (a) dos vistas de samples de la tarea actual
                (b) dos vistas sintetizadas de samples del replay buffer
              Aproxima representaciones de la misma clase independientemente
              de si son de la tarea actual o pasada.

  L_distill — Asymmetric Distillation sobre samples de la tarea actual:
                preserva la estructura de representaciones del modelo anterior.
                Solo activo para t > 0 (teacher = snapshot antes de la tarea t).

  L_CE      — Cross-entropy sobre la vista-0 de la tarea actual.
              Mantiene alineada la cabeza de clasificación.

Construcción del batch conjunto
--------------------------------
El loader SupCon produce batches (B, 2, C, H, W). El buffer guarda tensores de
una sola vista; sintetizamos la segunda vista con flip horizontal aleatorio
(suficiente como augmentación para el aprendizaje contrastivo).

El buffer se actualiza al terminar cada tarea con muestras de una sola vista.
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

CO2L_LAMBDA = 1.0   # peso λ del término de distilación
SUPCON_TEMP = 0.07


class Co2L(BaseMethod):
    """Implementación completa de Co²L (Cha et al. 2021).

    Combina SupCon sobre batch conjunto (tarea actual + replay), destilación
    asimétrica para preservar representaciones pasadas, y CE para la tarea actual.
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

    def begin_task(self, task_id: int) -> None:
        """Snapshot del modelo actual como teacher frozen para la destilación."""
        super().begin_task(task_id)
        if task_id > 0:
            self._prev_model = copy.deepcopy(self.backbone).to(self.device)
            self._prev_model.eval()
            for p in self._prev_model.parameters():
                p.requires_grad = False
            self._log(f"teacher snapshot tomado para task={task_id}")
        else:
            self._prev_model = None

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Actualiza el replay buffer con samples de vista única de la tarea terminada."""
        super().end_task(task_id, train_loader)
        self.buffer.update_from_loader(train_loader, task_id)
        self._log(f"buffer actualizado — {self.buffer}")

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_epochs: int = 500,
    ) -> Dict[str, list]:
        """Entrenamiento con SupCon + A-Distill + CE.

        train_loader es el loader estándar (una vista) para validación y end_task.
        El loader SupCon (dos vistas) se construye internamente.
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
        # Cosine annealing como en el paper original
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
                # x2v: (B, 2, C, H, W),  labels_cur: (B,)
                self._validate_batch_labels(labels_cur, task_id)
                x2v        = x2v.to(self.device)
                labels_cur = labels_cur.to(self.device)
                B          = labels_cur.size(0)

                # 1. SupCon sobre batch conjunto (tarea actual + replay buffer)
                feats_cur = self.backbone.forward_supcon(x2v)   # (B, 2, proj_dim)

                if len(self.buffer) > 0:
                    buf_imgs, buf_labels = self.buffer.sample(self.n_buf_samples)
                    buf_imgs   = buf_imgs.to(self.device)
                    buf_labels = buf_labels.to(self.device)
                    N = buf_labels.size(0)

                    # Buffer como negativos de una sola vista: se proyectan una vez
                    # y se expanden a (N, 2, proj_dim) para compatibilidad con
                    # SupConLoss, pero anchor_mask=False los excluye del numerador.
                    feats_buf    = self.backbone.project(buf_imgs)               # (N, proj_dim)
                    feats_buf_2v = feats_buf.unsqueeze(1).expand(-1, 2, -1).contiguous()  # (N, 2, proj_dim)

                    joint_feats  = torch.cat([feats_cur, feats_buf_2v], dim=0)  # (B+N, 2, proj_dim)
                    joint_labels = torch.cat([labels_cur, buf_labels],  dim=0)  # (B+N,)
                    anchor_mask  = torch.cat([
                        torch.ones(B,  dtype=torch.bool, device=self.device),
                        torch.zeros(N, dtype=torch.bool, device=self.device),
                    ])
                else:
                    joint_feats  = feats_cur
                    joint_labels = labels_cur
                    anchor_mask  = None

                supcon = self.supcon_loss(joint_feats, joint_labels, anchor_mask=anchor_mask)

                # 2. Destilación asimétrica sobre samples de la tarea actual (vista-0)
                distill = self._compute_distillation(x2v[:, 0])

                # 3. CE sobre la tarea actual (vista-0)
                logits_cur = self.backbone(x2v[:, 0])
                ce         = ce_criterion(logits_cur, labels_cur)

                # 4. Loss combinada
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

    def _compute_distillation(self, imgs: torch.Tensor) -> torch.Tensor:
        """A-Distill entre features del modelo actual y el teacher frozen.

        Usamos backbone.encode() (features del encoder, no del projector) porque
        es el espacio que se usa para clasificación downstream.
        Devuelve 0.0 si no hay teacher (tarea 0).
        """
        if self._prev_model is None:
            return torch.tensor(0.0, device=self.device)

        z_curr = self.backbone.encode(imgs)

        with torch.no_grad():
            z_prev = self._prev_model.encode(imgs)

        return self.distill_loss(z_curr, z_prev)


