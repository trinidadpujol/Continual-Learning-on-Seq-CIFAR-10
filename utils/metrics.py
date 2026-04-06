"""
Evaluation metrics for continual learning on Seq-CIFAR-10.

Two evaluation scenarios
------------------------
Class-IL  Predict among all 10 classes; no task identity is given at test time.
           This is the harder setting and the primary benchmark.

Task-IL   The task id is known at test time, so the model only needs to
           distinguish the 2 classes of the current task.  The 10-class
           logits are masked to the task's 2 classes before argmax.

Usage
-----
    # After training on task t:
    test_loaders = [seq_cifar.get_task_loader(k, train=False) for k in range(t + 1)]

    class_il = evaluate_class_il(backbone, test_loaders, device)
    task_il  = evaluate_task_il(backbone, test_loaders, device)
    # → {"acc_per_task": [0.82, 0.61, ...], "avg_acc": 0.715}

    # Accumulate across tasks for plots and tables:
    tracker = MetricsTracker(n_tasks=5)
    tracker.update(t, class_il, task_il)
    print(tracker.summary_df())
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import TASK_CLASSES


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation functions
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_class_il(
    model: nn.Module,
    test_loaders: List[DataLoader],
    device: torch.device,
) -> Dict[str, object]:
    """Class-Incremental evaluation: no task hint, predict among all 10 classes.

    Args:
        model:        Backbone whose forward() returns (B, 10) logits.
        test_loaders: test_loaders[j] is the DataLoader for task j.
                      Only tasks 0..len(test_loaders)-1 are evaluated.
        device:       Torch device.

    Returns:
        {
          "acc_per_task": [float, ...],  # accuracy per task in test_loaders
          "avg_acc":       float,        # mean over tasks
        }
    """
    model.eval()
    acc_per_task: List[float] = []

    for task_id, loader in enumerate(test_loaders):
        correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)                     # (B, num_classes) — all 10
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        acc_per_task.append(correct / total)

    return {
        "acc_per_task": acc_per_task,
        "avg_acc":      float(np.mean(acc_per_task)),
    }


@torch.no_grad()
def evaluate_task_il(
    model: nn.Module,
    test_loaders: List[DataLoader],
    device: torch.device,
) -> Dict[str, object]:
    """Task-Incremental evaluation: task id known, classify within 2 classes.

    The 10-class logits are masked to −∞ for all classes outside the task's
    2-class subset; argmax is then taken over the full 10-dim vector (but
    effectively only the 2 unmasked entries compete).

    Args:
        model:        Backbone whose forward() returns (B, 10) logits.
        test_loaders: test_loaders[j] is the DataLoader for task j.
        device:       Torch device.

    Returns:
        {
          "acc_per_task": [float, ...],
          "avg_acc":       float,
        }
    """
    model.eval()
    acc_per_task: List[float] = []

    for task_id, loader in enumerate(test_loaders):
        c0, c1 = TASK_CLASSES[task_id]

        # Build a fixed additive mask: 0 for the 2 task classes, −∞ elsewhere.
        # Applied to every batch so we don't recreate it per batch.
        num_classes = next(iter(loader))[0].shape  # peek for dtype; use 10 directly
        mask = torch.full((10,), float("-inf"), device=device)
        mask[c0] = 0.0
        mask[c1] = 0.0

        correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits        = model(imgs) + mask      # (B,10); non-task logits → −∞
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)
        acc_per_task.append(correct / total)

    return {
        "acc_per_task": acc_per_task,
        "avg_acc":      float(np.mean(acc_per_task)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Forgetting
# ──────────────────────────────────────────────────────────────────────────────

def compute_forgetting(acc_matrix: List[List[float]]) -> List[float]:
    """Per-task forgetting from a lower-triangular accuracy matrix.

    acc_matrix[i][j]  = accuracy on task j measured after training on task i
                        (only defined for j <= i).

    Forgetting for task j:
        F_j = acc_matrix[j][j] - acc_matrix[T-1][j]

    where T = len(acc_matrix).  The last task has no forgetting by definition.

    Args:
        acc_matrix: List of lists.  Row i must have at least i+1 elements.

    Returns:
        Forgetting values for tasks 0 .. T-2  (length T-1).
    """
    T = len(acc_matrix)
    if T < 2:
        return []
    return [
        acc_matrix[j][j] - acc_matrix[T - 1][j]
        for j in range(T - 1)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# MetricsTracker
# ──────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """Accumulates Class-IL and Task-IL results across tasks for one method.

    Usage
    -----
        tracker = MetricsTracker(n_tasks=5)

        # inside eval_fn, after training on task t:
        tracker.update(t, class_il_result, task_il_result)

        # after all tasks:
        df = tracker.summary_df()      # pandas DataFrame
        print(tracker.forgetting())    # per-task forgetting
        m  = tracker.acc_matrix("class_il")  # full lower-triangular matrix

    The accuracy matrix rows correspond to the state of the model after task t,
    and columns to the task being evaluated.  Only j <= t entries are defined.
    """

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        # acc_matrix_[scenario][t][j] = acc on task j after training on task t
        self._class_il: List[List[float]] = []   # row t = acc_per_task after task t
        self._task_il:  List[List[float]] = []
        self._avg_class_il: List[float]   = []
        self._avg_task_il:  List[float]   = []

    def update(
        self,
        task_id: int,
        class_il_result: Dict[str, object],
        task_il_result:  Dict[str, object],
    ) -> None:
        """Record evaluation results after finishing task_id.

        Args:
            task_id:          Index of the task just trained (0-based).
            class_il_result:  Output of evaluate_class_il().
            task_il_result:   Output of evaluate_task_il().
        """
        self._class_il.append(list(class_il_result["acc_per_task"]))
        self._task_il.append(list(task_il_result["acc_per_task"]))
        self._avg_class_il.append(float(class_il_result["avg_acc"]))
        self._avg_task_il.append(float(task_il_result["avg_acc"]))

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def acc_matrix(self, scenario: str = "class_il") -> np.ndarray:
        """Return the full accuracy matrix as an ndarray.

        Shape: (tasks_trained, tasks_trained) — lower triangle populated,
        upper triangle filled with NaN.

        Args:
            scenario: "class_il" or "task_il".
        """
        rows = self._class_il if scenario == "class_il" else self._task_il
        T = len(rows)
        mat = np.full((T, T), np.nan)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                mat[i, j] = v
        return mat

    def forgetting(self, scenario: str = "class_il") -> List[float]:
        """Per-task forgetting: acc right after learning task j minus final acc.

        Returns an empty list if fewer than 2 tasks have been recorded.
        """
        rows = self._class_il if scenario == "class_il" else self._task_il
        return compute_forgetting(rows)

    def avg_forgetting(self, scenario: str = "class_il") -> float:
        """Mean forgetting over all tasks except the last."""
        f = self.forgetting(scenario)
        return float(np.mean(f)) if f else 0.0

    def avg_acc_curve(self, scenario: str = "class_il") -> List[float]:
        """Average accuracy after each task (for the accuracy-vs-tasks plot)."""
        return self._avg_class_il if scenario == "class_il" else self._avg_task_il

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary_df(self):
        """Return a pandas DataFrame summarising results per task.

        Columns: Task, Class-IL avg, Task-IL avg, Class-IL forgetting,
                 Task-IL forgetting.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for summary_df()") from exc

        T = len(self._avg_class_il)
        forget_c = self.forgetting("class_il") + [float("nan")]  # last task = NaN
        forget_t = self.forgetting("task_il")  + [float("nan")]

        rows = []
        for t in range(T):
            rows.append({
                "Task":             t,
                "Class-IL avg":     round(self._avg_class_il[t], 4),
                "Task-IL avg":      round(self._avg_task_il[t],  4),
                "Class-IL forget":  round(forget_c[t], 4) if not np.isnan(forget_c[t]) else float("nan"),
                "Task-IL forget":   round(forget_t[t], 4) if not np.isnan(forget_t[t]) else float("nan"),
            })
        return pd.DataFrame(rows).set_index("Task")

    def __repr__(self) -> str:
        T = len(self._avg_class_il)
        if T == 0:
            return "MetricsTracker(empty)"
        last_c = self._avg_class_il[-1]
        last_t = self._avg_task_il[-1]
        return (
            f"MetricsTracker(tasks_recorded={T}  "
            f"Class-IL={last_c:.3f}  Task-IL={last_t:.3f})"
        )
