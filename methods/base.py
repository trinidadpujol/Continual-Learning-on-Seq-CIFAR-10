"""
Clase base para todos los métodos de aprendizaje continuo.

El contrato es simple: train_task() recibe un DataLoader que contiene SOLO
las muestras de task_id. Acceder a datos de tareas anteriores solo está
permitido a través del replay buffer. _validate_batch_labels() hace esta
validación en cada batch para detectar leakage accidental.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.constants import TASK_CLASSES
from models.backbone import Backbone

logger = logging.getLogger(__name__)


class BaseMethod(ABC):
    """Clase base para métodos de CL.

    Las subclases implementan:
      - train_task()  — loop de entrenamiento por tarea (obligatorio)
      - begin_task()  — hook antes de la tarea (opcional, ej: snapshot del modelo)
      - end_task()    — hook después de la tarea (opcional, ej: actualizar buffer / Fisher)

    uses_replay debe ser True en subclases que lean del ReplayBuffer.
    """

    uses_replay: bool = False

    def __init__(self, backbone: Backbone, device: torch.device):
        self.backbone = backbone.to(device)
        self.device = device

        self._current_task_id: int = -1
        self._allowed_classes: Set[int] = set()

    def begin_task(self, task_id: int) -> None:
        self._current_task_id = task_id
        c0, c1 = TASK_CLASSES[task_id]
        self._allowed_classes = {c0, c1}
        self._log(f"begin_task  task={task_id}  classes={self._allowed_classes}  replay={self.uses_replay}")

    @abstractmethod
    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        n_epochs: int,
    ) -> Dict[str, list]:
        """Entrenar sobre una única tarea.

        El DataLoader debe contener solo muestras de task_id.
        Llamar _validate_batch_labels() en cada batch para verificarlo.

        Devuelve un dict de métricas por época, ej: {"loss": [...], "acc": [...]}.
        """
        ...

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        self._log(f"end_task  task={task_id}")

    def _validate_batch_labels(self, labels: torch.Tensor, task_id: int) -> None:
        """Verifica que todos los labels del batch pertenecen a task_id.

        Solo se aplica a batches del DataLoader de la tarea (no a muestras del buffer,
        que por definición vienen de tareas pasadas).
        """
        seen: Set[int] = set(labels.cpu().tolist())
        unexpected = seen - self._allowed_classes
        if unexpected:
            raise RuntimeError(
                f"Data leakage en tarea {task_id}: "
                f"el batch contiene labels {unexpected} que no pertenecen a "
                f"las clases {self._allowed_classes} de esta tarea. "
                f"Verificar que se está usando get_task_loader(task_id={task_id})."
            )

    def _make_optimizer(
        self,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        parameters=None,
    ) -> torch.optim.SGD:
        params = parameters if parameters is not None else self.backbone.parameters()
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def _log(self, msg: str) -> None:
        logger.info("[%s] %s", type(self).__name__, msg)

    def _epoch_log(self, task_id: int, epoch: int, n_epochs: int, loss: float, acc: float) -> None:
        self._log(
            f"task={task_id}  epoch={epoch + 1}/{n_epochs}  "
            f"loss={loss:.4f}  acc={acc:.3f}"
        )
