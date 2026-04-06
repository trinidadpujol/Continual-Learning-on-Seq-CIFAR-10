"""
Abstract base class for all continual learning methods.

Protocol contract
-----------------
- train_task(task_id, train_loader, n_epochs) receives a DataLoader that
  already contains ONLY the data for task_id.  The method must not load
  or iterate any other task's dataset directly.
- Past examples may enter the training step exclusively through a
  ReplayBuffer (replay-enabled methods), never via a separate DataLoader.
- _validate_batch_labels() is called on every batch from train_loader and
  asserts that all labels belong to task_id's two classes.  This is a
  runtime guard against accidental data leakage.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import TASK_CLASSES
from models.backbone import Backbone

logger = logging.getLogger(__name__)


class BaseMethod(ABC):
    """
    Base class for all continual learning methods.

    Subclasses implement:
      - train_task()  — the per-task training loop (required)
      - begin_task()  — pre-task hook, e.g. snapshot the model (optional)
      - end_task()    — post-task hook, e.g. update buffer / compute Fisher (optional)

    Class attribute ``uses_replay`` must be set to True in any subclass that
    reads from a ReplayBuffer during training.  It controls validation behaviour
    and is used by ContinualTrainer for logging.
    """

    #: Override in replay-enabled subclasses.
    uses_replay: bool = False

    def __init__(self, backbone: Backbone, device: torch.device):
        self.backbone = backbone.to(device)
        self.device = device

        # Set by begin_task / train_task; used by _validate_batch_labels.
        self._current_task_id: int = -1
        self._allowed_classes: Set[int] = set()

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def begin_task(self, task_id: int) -> None:
        """Called before training on task_id.  Override as needed."""
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
        """
        Train on a single task.

        The DataLoader passed here must only contain samples from task_id.
        Use _validate_batch_labels() on every batch to enforce this at runtime.

        Returns:
            Dictionary of per-epoch logged metrics, e.g. {"loss": [...], "acc": [...]}.
        """
        ...

    def end_task(self, task_id: int, train_loader: DataLoader) -> None:
        """Called after training on task_id.  Override as needed."""
        self._log(f"end_task  task={task_id}")

    # ------------------------------------------------------------------
    # Protocol-enforcement utilities
    # ------------------------------------------------------------------

    def _validate_batch_labels(self, labels: torch.Tensor, task_id: int) -> None:
        """Assert that all labels in a batch belong to the current task.

        Only validates batches from the task DataLoader (not buffer samples,
        which by definition come from past tasks).

        Raises:
            RuntimeError: if any label is outside the task's two classes.
        """
        seen: Set[int] = set(labels.cpu().tolist())
        unexpected = seen - self._allowed_classes
        if unexpected:
            raise RuntimeError(
                f"Data leakage detected in task {task_id}!  "
                f"Batch contains labels {unexpected} which do not belong to "
                f"task {task_id} classes {self._allowed_classes}.  "
                f"Make sure you are passing get_task_loader(task_id={task_id}) "
                f"and not a loader for another task."
            )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

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
        """Emit a structured log line prefixed with the method name."""
        logger.info("[%s] %s", type(self).__name__, msg)

    def _epoch_log(self, task_id: int, epoch: int, n_epochs: int, loss: float, acc: float) -> None:
        self._log(
            f"task={task_id}  epoch={epoch + 1}/{n_epochs}  "
            f"loss={loss:.4f}  acc={acc:.3f}"
        )
