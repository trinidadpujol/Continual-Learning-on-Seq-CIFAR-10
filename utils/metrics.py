"""
Métricas de evaluación para aprendizaje continuo sobre Seq-CIFAR-10.

Dos escenarios:
  Class-IL  El modelo clasifica entre las 10 clases sin saber a qué tarea pertenece
            cada muestra. Es el setting más difícil y el principal benchmark del TP.

  Task-IL   El modelo conoce el task_id en test, así que solo necesita distinguir
            entre las 2 clases de esa tarea. Mucho más fácil que Class-IL.

Uso típico:
    test_loaders = [seq_cifar.get_task_loader(k, train=False) for k in range(t + 1)]
    class_il = evaluate_class_il(backbone, test_loaders, device)
    task_il  = evaluate_task_il(backbone, test_loaders, device)
    # → {"acc_per_task": [0.82, 0.61, ...], "avg_acc": 0.715}
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.constants import TASK_CLASSES


@torch.no_grad()
def evaluate_class_il(
    model: nn.Module,
    test_loaders: List[DataLoader],
    device: torch.device,
) -> Dict[str, object]:
    """Class-IL: clasifica entre las 10 clases sin información de tarea."""
    model.eval()
    acc_per_task: List[float] = []

    for task_id, loader in enumerate(test_loaders):
        correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)              # (B, 10), sin restricción
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
    """Task-IL: el task_id es conocido, clasificar solo entre las 2 clases de la tarea.

    Enmascaramos los logits de las clases que no pertenecen a la tarea con -inf
    antes del argmax; en la práctica solo compiten los 2 logits deseados.
    """
    model.eval()
    acc_per_task: List[float] = []

    for task_id, loader in enumerate(test_loaders):
        c0, c1 = TASK_CLASSES[task_id]

        # Máscara fija: 0 para las 2 clases de la tarea, -inf para el resto
        mask = torch.full((10,), float("-inf"), device=device)
        mask[c0] = 0.0
        mask[c1] = 0.0

        correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits  = model(imgs) + mask
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        acc_per_task.append(correct / total)

    return {
        "acc_per_task": acc_per_task,
        "avg_acc":      float(np.mean(acc_per_task)),
    }


def compute_forgetting(acc_matrix: List[List[float]]) -> List[float]:
    """Forgetting por tarea a partir de la matriz de accuracy triangular inferior.

    acc_matrix[i][j] = accuracy en tarea j medida después de entrenar la tarea i.

    F_j = acc_matrix[j][j] - acc_matrix[T-1][j]
    (máxima accuracy lograda en la tarea j, menos la accuracy final)

    La última tarea no tiene forgetting por definición.
    """
    T = len(acc_matrix)
    if T < 2:
        return []
    return [
        acc_matrix[j][j] - acc_matrix[T - 1][j]
        for j in range(T - 1)
    ]


class MetricsTracker:
    """Acumula resultados Class-IL y Task-IL a lo largo de las tareas para un método.

    Uso:
        tracker = MetricsTracker(n_tasks=5)
        tracker.update(t, class_il_result, task_il_result)   # después de cada tarea
        df = tracker.summary_df()
        print(tracker.forgetting("class_il"))
    """

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        self._class_il: List[List[float]] = []
        self._task_il:  List[List[float]] = []
        self._avg_class_il: List[float]   = []
        self._avg_task_il:  List[float]   = []

    def update(
        self,
        task_id: int,
        class_il_result: Dict[str, object],
        task_il_result:  Dict[str, object],
    ) -> None:
        self._class_il.append(list(class_il_result["acc_per_task"]))
        self._task_il.append(list(task_il_result["acc_per_task"]))
        self._avg_class_il.append(float(class_il_result["avg_acc"]))
        self._avg_task_il.append(float(task_il_result["avg_acc"]))

    def acc_matrix(self, scenario: str = "class_il") -> np.ndarray:
        """Matriz de accuracy (tasks_trained x tasks_trained), triángulo superior = NaN."""
        rows = self._class_il if scenario == "class_il" else self._task_il
        T = len(rows)
        mat = np.full((T, T), np.nan)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                mat[i, j] = v
        return mat

    def forgetting(self, scenario: str = "class_il") -> List[float]:
        """Forgetting por tarea. Lista vacía si hay menos de 2 tareas registradas."""
        rows = self._class_il if scenario == "class_il" else self._task_il
        return compute_forgetting(rows)

    def avg_forgetting(self, scenario: str = "class_il") -> float:
        f = self.forgetting(scenario)
        return float(np.mean(f)) if f else 0.0

    def avg_acc_curve(self, scenario: str = "class_il") -> List[float]:
        """Accuracy promedio después de cada tarea (para las curvas del informe)."""
        return self._avg_class_il if scenario == "class_il" else self._avg_task_il

    def summary_df(self):
        """DataFrame de pandas con un resumen por tarea."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas es necesario para summary_df()") from exc

        T = len(self._avg_class_il)
        forget_c = self.forgetting("class_il") + [float("nan")]
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
