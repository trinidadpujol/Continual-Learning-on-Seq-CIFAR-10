"""
ContinualTrainer: orquesta el entrenamiento secuencial tarea por tarea.

Por cada tarea t:
  1. Construye un DataLoader solo para la tarea t (nunca crea loaders de tareas pasadas).
  2. Llama begin_task → train_task → end_task en orden.
  3. Llama eval_fn (si se provee) para registrar métricas Class-IL y Task-IL.

Los métodos con replay acceden a datos pasados exclusivamente a través de su
propio buffer interno. ContinualTrainer no toca el buffer directamente.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

from data.dataset import SeqCIFAR10
from methods.base import BaseMethod

logger = logging.getLogger(__name__)

EvalFn = Callable[[int, BaseMethod], Dict[str, float]]


class ContinualTrainer:
    """Orquestador del protocolo de aprendizaje continuo secuencial.

    method:    Instancia concreta de BaseMethod.
    seq_cifar: Dataset configurado con la división en tareas.
    n_epochs:  Épocas de entrenamiento por tarea.
    eval_fn:   Función opcional eval_fn(task_id, method) → métricas, llamada
               después de cada tarea para registrar Class-IL y Task-IL.
    """

    def __init__(
        self,
        method: BaseMethod,
        seq_cifar: SeqCIFAR10,
        n_epochs: int = 50,
        eval_fn: Optional[EvalFn] = None,
    ):
        self.method    = method
        self.seq_cifar = seq_cifar
        self.n_epochs  = n_epochs
        self.eval_fn   = eval_fn

    def train_all_tasks(self) -> Dict[str, object]:
        """Ejecuta el protocolo completo de entrenamiento secuencial.

        Para cada tarea crea el loader correspondiente (solo esa tarea) y corre
        el ciclo begin_task → train_task → end_task. Registra logs y tiempos.

        Devuelve dict con train_logs, eval_results y task_times.
        """
        train_logs:   Dict[int, Dict] = {}
        eval_results: Dict[int, Dict] = {}
        task_times:   Dict[int, float] = {}

        n_tasks     = self.seq_cifar.n_tasks
        method_name = type(self.method).__name__

        print(f"\n{'='*60}")
        print(f"  ContinualTrainer — {method_name}")
        print(f"  Tasks: {n_tasks}   Epochs/task: {self.n_epochs}")
        print(f"  Replay: {self.method.uses_replay}")
        print(f"{'='*60}")

        for task_id in range(n_tasks):
            t_start = time.perf_counter()

            class_names = self.seq_cifar.get_class_names(task_id)
            n_train     = self.seq_cifar.task_size(task_id, train=True)

            print(f"\n[Task {task_id}] {class_names[0]} + {class_names[1]}"
                  f"  ({n_train} train samples)")
            print(f"  Data source : get_task_loader(task_id={task_id}, train=True)")
            print(f"  Past data   : {'ReplayBuffer only' if self.method.uses_replay and task_id > 0 else 'none'}")

            # Solo se crea el loader de la tarea actual
            train_loader = self.seq_cifar.get_task_loader(task_id, train=True)

            logger.info(
                "[ContinualTrainer] task=%d  method=%s  loader_classes=%s  replay=%s",
                task_id, method_name, class_names, self.method.uses_replay,
            )

            self.method.begin_task(task_id)
            log = self.method.train_task(task_id, train_loader, self.n_epochs)
            self.method.end_task(task_id, train_loader)

            elapsed = time.perf_counter() - t_start
            train_logs[task_id] = log
            task_times[task_id] = elapsed

            final_loss = log["loss"][-1] if log.get("loss") else float("nan")
            final_acc  = log["acc"][-1]  if log.get("acc")  else float("nan")
            print(f"  Done in {elapsed:.1f}s — final loss={final_loss:.4f}  acc={final_acc:.3f}")

            if self.eval_fn is not None:
                metrics = self.eval_fn(task_id, self.method)
                eval_results[task_id] = metrics
                self._print_eval(task_id, metrics)

        print(f"\n{'='*60}")
        print("  Training complete.")
        print(f"{'='*60}\n")

        return {
            "train_logs":   train_logs,
            "eval_results": eval_results,
            "task_times":   task_times,
        }

    @staticmethod
    def _print_eval(task_id: int, metrics: Dict[str, float]) -> None:
        parts = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"  Eval after task {task_id}: {parts}")
