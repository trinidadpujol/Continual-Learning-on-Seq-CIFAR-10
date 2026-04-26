"""
Learning without Forgetting (LwF).
Li & Hoiem, TPAMI 2018. https://arxiv.org/abs/1606.09282

Antes de entrenar la tarea t (t > 0), guardamos un snapshot frozen del modelo
como "teacher". Durante el entrenamiento usamos sus predicciones como soft targets:

    L = L_CE(nueva tarea) + α · L_distill

donde L_distill = T² · KL(softmax(teacher/T) ‖ log_softmax(student/T))

El factor T² reescala el gradiente de los soft targets para que tenga la misma
magnitud que el de los hard targets (Hinton et al., 2015).

La destilación se hace sobre todos los 10 logits, no solo los de la tarea anterior,
lo que obliga al modelo a preservar la distribución completa del teacher.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.base import BaseMethod
from models.backbone import Backbone

LWF_ALPHA       = 1.0   # peso de la pérdida de destilación
LWF_TEMPERATURE = 2.0   # temperatura para los soft targets


class LwF(BaseMethod):
    """Learning without Forgetting: destilación de conocimiento sin acceso a datos anteriores.

    alpha controla el balance entre aprender la nueva tarea y preservar las anteriores.
    temperature suaviza las distribuciones del teacher (más alto = más suave).
    """

    uses_replay: bool = False

    def __init__(
        self,
        backbone: Backbone,
        device: torch.device,
        alpha: float = LWF_ALPHA,
        temperature: float = LWF_TEMPERATURE,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
    ):
        super().__init__(backbone, device)
        self.alpha        = alpha
        self.temperature  = temperature
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self._prev_model: Optional[nn.Module] = None

    def begin_task(self, task_id: int) -> None:
        """Snapshot del modelo actual como teacher frozen antes de la tarea t.

        En la tarea 0 no hay teacher, así que L_distill = 0 (solo CE).
        """
        super().begin_task(task_id)
        if task_id > 0:
            self._prev_model = copy.deepcopy(self.backbone).to(self.device)
            self._prev_model.eval()
            for p in self._prev_model.parameters():
                p.requires_grad = False
            self._log(f"teacher snapshot tomado para task={task_id}")
        else:
            self._prev_model = None

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_epochs: int = 50,
    ) -> Dict[str, list]:
        if self._current_task_id != task_id:
            self.begin_task(task_id)

        optimizer = self._make_optimizer(self.lr, self.momentum, self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        log: Dict[str, list] = {"loss": [], "acc": [], "ce": [], "distill": []}

        for epoch in range(n_epochs):
            self.backbone.train()
            running_loss    = 0.0
            running_ce      = 0.0
            running_distill = 0.0
            correct = total = 0

            for imgs, labels in train_loader:
                self._validate_batch_labels(labels, task_id)

                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                logits = self.backbone(imgs)
                ce     = criterion(logits, labels)

                distill = self._distillation_loss(imgs, logits)
                loss    = ce + self.alpha * distill

                loss.backward()
                optimizer.step()

                B = labels.size(0)
                running_loss    += loss.item()    * B
                running_ce      += ce.item()      * B
                running_distill += distill.item() * B
                correct         += (logits.argmax(1) == labels).sum().item()
                total           += B

            n = max(total, 1)
            log["loss"].append(running_loss    / n)
            log["ce"].append(running_ce        / n)
            log["distill"].append(running_distill / n)
            log["acc"].append(correct / n)
            self._epoch_log(task_id, epoch, n_epochs, running_loss / n, correct / n)

        return log

    def _distillation_loss(
        self,
        imgs: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence entre el teacher frozen y el student a temperatura T.

        L_dist = T² · KL(softmax(z_teacher/T) ‖ log_softmax(z_student/T))

        Devuelve 0.0 si no hay teacher (tarea 0).
        """
        if self._prev_model is None:
            return torch.tensor(0.0, device=self.device)

        T = self.temperature
        with torch.no_grad():
            teacher_logits = self._prev_model(imgs)

        p_teacher     = F.softmax(teacher_logits / T, dim=1)
        log_p_student = F.log_softmax(student_logits / T, dim=1)

        # Factor T² para reescalar el gradiente (Hinton et al. 2015)
        distill = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T ** 2)
        return distill
