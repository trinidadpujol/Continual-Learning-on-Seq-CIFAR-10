"""
Fixed-capacity replay buffer for continual learning.

Design
------
The buffer stores individual samples as CPU tensors (images already normalised
by the task DataLoader) together with their integer class label and the task id
they came from.  Keeping samples normalised means buffer batches are immediately
ready for the model — no extra transform needed at sample time.

Update policy: reservoir sampling (Vitter, 1985)
    After the buffer is full, each new sample replaces a uniformly random
    existing slot with probability capacity / n_seen.  This guarantees that,
    at any point, the buffer is a uniform random sample over ALL samples ever
    offered — regardless of task order or class frequencies.

    Consequence: no manual balancing is needed.  As more tasks are offered the
    buffer naturally shifts toward the distribution of all tasks seen so far.

Usage
-----
    buf = ReplayBuffer(capacity=200)

    # --- end_task: populate from a finished-task DataLoader ---
    buf.update_from_loader(train_loader, task_id=0)

    # --- train_task: mix buffer samples with current-task batch ---
    if len(buf) > 0:
        buf_imgs, buf_labels = buf.sample(32)

    # --- inspect composition ---
    print(buf)
    print(buf.class_counts())   # {0: 21, 1: 19, ...}
    print(buf.task_counts())    # {0: 40, 1: 80, ...}

    # --- use as a Dataset (e.g. for a DataLoader over all buffer) ---
    dataset = buf.as_dataset()
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from data.dataset import TASK_CLASSES


class ReplayBuffer:
    """
    Fixed-capacity replay buffer using reservoir sampling.

    Args:
        capacity: Maximum number of samples the buffer may hold at any time.
    """

    def __init__(self, capacity: int = 200):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = capacity

        # Parallel lists — index i is one stored sample.
        self._imgs:     List[torch.Tensor] = []
        self._labels:   List[int]          = []
        self._task_ids: List[int]          = []

        # Total number of samples ever offered (not just stored).
        # Used to compute replacement probability in reservoir sampling.
        self._n_seen: int = 0

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def update(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
    ) -> None:
        """Insert a batch of samples using reservoir sampling.

        Maintains the invariant that the buffer is a uniform random sample
        over all samples ever offered across all calls to update().

        Args:
            images:  (B, C, H, W) normalised image tensors.
            labels:  (B,) integer class labels.
            task_id: Task index that produced this batch.
        """
        for img, lbl in zip(images.cpu(), labels.cpu().tolist()):
            self._n_seen += 1
            if len(self._imgs) < self.capacity:
                # Buffer not yet full — always accept.
                self._imgs.append(img)
                self._labels.append(int(lbl))
                self._task_ids.append(task_id)
            else:
                # Reservoir sampling: replace slot j with probability
                # capacity / n_seen.
                j = random.randrange(self._n_seen)
                if j < self.capacity:
                    self._imgs[j]     = img
                    self._labels[j]   = int(lbl)
                    self._task_ids[j] = task_id

    def update_from_loader(
        self,
        loader: DataLoader,
        task_id: int,
        max_batches: Optional[int] = None,
    ) -> None:
        """Populate the buffer from a DataLoader after finishing a task.

        Iterates over the loader and feeds every batch through reservoir
        sampling.  Call this inside end_task() to ensure the buffer reflects
        the just-finished task.

        Args:
            loader:      DataLoader for the finished task (normalised tensors).
            task_id:     Index of the finished task.
            max_batches: If set, stop after this many batches (useful for
                         quick testing without iterating the full dataset).
        """
        for i, (imgs, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            self.update(imgs, labels, task_id)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw n samples uniformly at random (with replacement).

        Args:
            n: Number of samples to draw.  If n > len(buffer), samples are
               drawn with replacement so the return size is always exactly n.

        Returns:
            images: (n, C, H, W) tensor
            labels: (n,) int64 tensor

        Raises:
            RuntimeError: if the buffer is empty.
        """
        self._assert_nonempty("sample")
        indices = random.choices(range(len(self._imgs)), k=n)
        imgs   = torch.stack([self._imgs[i]   for i in indices])
        labels = torch.tensor([self._labels[i] for i in indices], dtype=torch.long)
        return imgs, labels

    def sample_with_task(
        self, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Like sample(), but also returns the task_id for each drawn sample.

        Returns:
            images:   (n, C, H, W)
            labels:   (n,)
            task_ids: list of length n
        """
        self._assert_nonempty("sample_with_task")
        indices  = random.choices(range(len(self._imgs)), k=n)
        imgs     = torch.stack([self._imgs[i]     for i in indices])
        labels   = torch.tensor([self._labels[i]  for i in indices], dtype=torch.long)
        task_ids = [self._task_ids[i]              for i in indices]
        return imgs, labels, task_ids

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def class_counts(self) -> Dict[int, int]:
        """Return {class_label: count} for all samples currently stored."""
        return dict(Counter(self._labels))

    def task_counts(self) -> Dict[int, int]:
        """Return {task_id: count} for all samples currently stored."""
        return dict(Counter(self._task_ids))

    def composition_str(self) -> str:
        """Human-readable breakdown by task and class."""
        if len(self) == 0:
            return "ReplayBuffer(empty)"

        lines = [f"ReplayBuffer  size={len(self)}/{self.capacity}  n_seen={self._n_seen}"]
        task_counts = self.task_counts()
        class_counts = self.class_counts()

        for task_id in sorted(task_counts):
            c0, c1 = TASK_CLASSES[task_id]
            n0 = class_counts.get(c0, 0)
            n1 = class_counts.get(c1, 0)
            lines.append(
                f"  task {task_id}: {task_counts[task_id]:4d} samples"
                f"  (class {c0}={n0}, class {c1}={n1})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def as_dataset(self) -> "_BufferDataset":
        """Return a torch Dataset wrapping all stored samples.

        Useful when you want a DataLoader over the entire buffer, e.g. for
        evaluation or for methods that train on a combined current+buffer set.

        Returns:
            Dataset yielding (image_tensor, label) pairs.
        """
        return _BufferDataset(
            list(self._imgs),
            list(self._labels),
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._imgs)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity}, "
            f"stored={len(self)}, n_seen={self._n_seen})"
        )

    def _assert_nonempty(self, caller: str) -> None:
        if len(self) == 0:
            raise RuntimeError(
                f"ReplayBuffer.{caller}() called on an empty buffer.  "
                "Call update() or update_from_loader() first."
            )


# ----------------------------------------------------------------------
# Internal Dataset wrapper
# ----------------------------------------------------------------------

class _BufferDataset(Dataset):
    """Read-only Dataset view of a snapshot of the buffer's contents."""

    def __init__(self, imgs: List[torch.Tensor], labels: List[int]):
        self._imgs   = imgs
        self._labels = labels

    def __len__(self) -> int:
        return len(self._imgs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._imgs[idx], self._labels[idx]
