"""
ContinualTrainer — orchestrates the sequential task-by-task training protocol.

Responsibilities
----------------
1. For each task t, obtain a DataLoader for task t ONLY via
   seq_cifar.get_task_loader(t).  Past task loaders are never created here.

2. Call method.begin_task(t) → method.train_task(t, loader, n_epochs)
   → method.end_task(t, loader) in order, with no overlap between tasks.

3. Collect per-task training logs and per-task/per-method evaluation results
   into a structured results dictionary.

4. Log the data source for every task so it is auditable in the notebook
   output (class names, split size, whether replay is active).

Replay contract
---------------
- ContinualTrainer never touches the ReplayBuffer directly.
- Replay-enabled methods (uses_replay=True) sample from their own buffer
  inside train_task / end_task — ContinualTrainer only announces this in logs.
- Past task DataLoaders are never passed to any method.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

from data.dataset import SeqCIFAR10
from methods.base import BaseMethod

logger = logging.getLogger(__name__)


# Type alias: callable that receives (task_id, method) and returns a metrics dict.
EvalFn = Callable[[int, BaseMethod], Dict[str, float]]


class ContinualTrainer:
    """Orchestrates continual learning training across all tasks.

    Args:
        method:    A concrete BaseMethod subclass to train.
        seq_cifar: Configured SeqCIFAR10 instance (provides task-split loaders).
        n_epochs:  Number of training epochs per task.
        eval_fn:   Optional callable ``eval_fn(task_id, method) -> metrics_dict``
                   called after each task finishes.  Receives the task index
                   (0-based, tasks seen so far = 0..task_id) and the method.
                   Useful to evaluate Class-IL / Task-IL after every task.
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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train_all_tasks(self) -> Dict[str, object]:
        """Run the full sequential training protocol.

        For each task t:
          - Creates a DataLoader for task t ONLY (no past loaders created).
          - Calls begin_task → train_task → end_task.
          - Optionally evaluates and records metrics.

        Returns:
            {
              "train_logs":   {task_id: {"loss": [...], "acc": [...]}},
              "eval_results": {task_id: metrics_dict},   # empty if no eval_fn
              "task_times":   {task_id: seconds_float},
            }
        """
        train_logs:   Dict[int, Dict] = {}
        eval_results: Dict[int, Dict] = {}
        task_times:   Dict[int, float] = {}

        n_tasks      = self.seq_cifar.n_tasks
        method_name  = type(self.method).__name__

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

            # ── Obtain the loader for the current task ONLY ───────────
            train_loader = self.seq_cifar.get_task_loader(task_id, train=True)
            # Past task loaders are deliberately NOT created here.
            # ─────────────────────────────────────────────────────────

            logger.info(
                "[ContinualTrainer] task=%d  method=%s  loader_classes=%s  replay=%s",
                task_id, method_name, class_names, self.method.uses_replay,
            )

            # ── Training lifecycle ────────────────────────────────────
            self.method.begin_task(task_id)
            log = self.method.train_task(task_id, train_loader, self.n_epochs)
            self.method.end_task(task_id, train_loader)
            # ─────────────────────────────────────────────────────────

            elapsed = time.perf_counter() - t_start
            train_logs[task_id]  = log
            task_times[task_id]  = elapsed

            final_loss = log["loss"][-1] if log.get("loss") else float("nan")
            final_acc  = log["acc"][-1]  if log.get("acc")  else float("nan")
            print(f"  Done in {elapsed:.1f}s — final loss={final_loss:.4f}  acc={final_acc:.3f}")

            # ── Optional evaluation after each task ───────────────────
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_eval(task_id: int, metrics: Dict[str, float]) -> None:
        parts = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"  Eval after task {task_id}: {parts}")
