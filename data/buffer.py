"""
Buffer de replay de capacidad fija usando reservoir sampling (Vitter, 1985).

Guarda muestras como tensores CPU ya normalizados (listas de task_id incluidas
para poder ver la distribución). El reservoir sampling garantiza que en cualquier
momento el buffer es una muestra uniforme de todo lo visto hasta ahora, sin
necesidad de balanceo manual entre tareas.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from data.constants import TASK_CLASSES


class ReplayBuffer:
    """Buffer de replay con reservoir sampling.

    Capacidad fija: cuando está lleno, cada nueva muestra reemplaza una
    posición aleatoria con probabilidad capacity/n_seen.
    """

    def __init__(self, capacity: int = 200):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = capacity

        self._imgs:     List[torch.Tensor] = []
        self._labels:   List[int]          = []
        self._task_ids: List[int]          = []

        # Total de muestras vistas (no solo las almacenadas)
        self._n_seen: int = 0

    def update(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
    ) -> None:
        """Inserta un batch usando reservoir sampling."""
        for img, lbl in zip(images.cpu(), labels.cpu().tolist()):
            self._n_seen += 1
            if len(self._imgs) < self.capacity:
                self._imgs.append(img)
                self._labels.append(int(lbl))
                self._task_ids.append(task_id)
            else:
                # Reemplazar posición aleatoria con prob capacity/n_seen
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
        """Llena el buffer desde un DataLoader al terminar una tarea.

        Llamar esto en end_task() para que el buffer refleje la tarea recién terminada.
        max_batches sirve para tests rápidos sin iterar todo el dataset.
        """
        for i, (imgs, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            self.update(imgs, labels, task_id)

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samplea n muestras con reemplazo. Siempre devuelve exactamente n."""
        self._assert_nonempty("sample")
        indices = random.choices(range(len(self._imgs)), k=n)
        imgs   = torch.stack([self._imgs[i]   for i in indices])
        labels = torch.tensor([self._labels[i] for i in indices], dtype=torch.long)
        return imgs, labels

    def sample_with_task(
        self, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Igual que sample() pero también devuelve los task_ids."""
        self._assert_nonempty("sample_with_task")
        indices  = random.choices(range(len(self._imgs)), k=n)
        imgs     = torch.stack([self._imgs[i]     for i in indices])
        labels   = torch.tensor([self._labels[i]  for i in indices], dtype=torch.long)
        task_ids = [self._task_ids[i]              for i in indices]
        return imgs, labels, task_ids

    def class_counts(self) -> Dict[int, int]:
        return dict(Counter(self._labels))

    def task_counts(self) -> Dict[int, int]:
        return dict(Counter(self._task_ids))

    def composition_str(self) -> str:
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

    def as_dataset(self) -> "_BufferDataset":
        """Dataset wrapping sobre todo el buffer (útil para DataLoaders de evaluación)."""
        return _BufferDataset(
            list(self._imgs),
            list(self._labels),
        )

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
                f"ReplayBuffer.{caller}() llamado sobre buffer vacío. "
                "Llamar update() o update_from_loader() primero."
            )


class _BufferDataset(Dataset):
    """Vista read-only del contenido del buffer como Dataset de PyTorch."""

    def __init__(self, imgs: List[torch.Tensor], labels: List[int]):
        self._imgs   = imgs
        self._labels = labels

    def __len__(self) -> int:
        return len(self._imgs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._imgs[idx], self._labels[idx]
