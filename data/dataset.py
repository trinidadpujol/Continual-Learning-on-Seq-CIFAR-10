"""
Seq-CIFAR-10: CIFAR-10 dividido en 5 tareas secuenciales de 2 clases cada una.

Usamos torchvision para descargar CIFAR-10 la primera vez. Guardamos los índices
por tarea para armar DataLoaders rápido sin recargar todo el dataset cada vez.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from data.constants import TASK_CLASSES, CLASS_NAMES  # noqa: F401
from data.buffer import ReplayBuffer  # noqa: F401

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_supcon_transforms() -> "TwoViewTransform":
    """Augmentaciones para SupCon: dos vistas aleatorias de la misma imagen."""
    augment = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return TwoViewTransform(augment)


class TwoViewTransform:
    """Aplica la misma augmentación dos veces para obtener dos vistas de la imagen.

    Devuelve un tensor (2, C, H, W) → el DataLoader produce batches (B, 2, C, H, W).
    """

    def __init__(self, transform: transforms.Compose):
        self.transform = transform

    def __call__(self, x) -> torch.Tensor:
        v1 = self.transform(x)
        v2 = self.transform(x)
        return torch.stack([v1, v2])


class _TransformSubset(Dataset):
    """Subconjunto del dataset CIFAR-10 crudo (PIL) con transform lazy.

    Necesitamos mantener las imágenes en PIL para poder aplicar distintos
    transforms (incluyendo TwoViewTransform para SupCon).
    """

    def __init__(
        self,
        base_dataset: datasets.CIFAR10,
        indices: List[int],
        transform,
    ):
        self._base = base_dataset
        self._indices = indices
        self._transform = transform

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        img, label = self._base[self._indices[idx]]
        return self._transform(img), label


class SeqCIFAR10:
    """
    CIFAR-10 secuencial para aprendizaje continuo.

    Divide las 10 clases en N tareas (default 5), 2 clases por tarea.
    Pre-computa los índices de cada tarea para construir DataLoaders en O(1).

    Ejemplo de uso:
        seq = SeqCIFAR10(data_root="./data", n_tasks=5)
        loader = seq.get_task_loader(task_id=0, train=True)
        supcon_loader = seq.get_task_loader(task_id=0, train=True, supcon=True)
    """

    N_TASKS = 5
    CLASSES_PER_TASK = 2

    def __init__(
        self,
        data_root: str = "./data",
        n_tasks: int = N_TASKS,
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        self.data_root = data_root
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_classes = TASK_CLASSES[:n_tasks]

        # Cargamos sin transform para poder aplicarlo lazy (PIL images)
        self._raw_train = datasets.CIFAR10(data_root, train=True,  download=True, transform=None)
        self._raw_test  = datasets.CIFAR10(data_root, train=False, download=True, transform=None)

        train_targets = torch.tensor(self._raw_train.targets)
        test_targets  = torch.tensor(self._raw_test.targets)

        self._train_indices: List[List[int]] = []
        self._test_indices:  List[List[int]] = []

        for c0, c1 in self.task_classes:
            train_mask = (train_targets == c0) | (train_targets == c1)
            self._train_indices.append(train_mask.nonzero(as_tuple=True)[0].tolist())

            test_mask = (test_targets == c0) | (test_targets == c1)
            self._test_indices.append(test_mask.nonzero(as_tuple=True)[0].tolist())

    def get_task_loader(
        self,
        task_id: int,
        train: bool = True,
        supcon: bool = False,
    ) -> DataLoader:
        """DataLoader restringido a las dos clases de task_id.

        supcon=True devuelve batches (B, 2, C, H, W) para el pre-entrenamiento contrastivo.
        Solo tiene sentido con train=True.
        """
        if task_id < 0 or task_id >= self.n_tasks:
            raise ValueError(f"task_id must be in [0, {self.n_tasks - 1}], got {task_id}")
        if supcon and not train:
            raise ValueError("supcon=True solo es válido para el split de entrenamiento")

        if train:
            transform = get_supcon_transforms() if supcon else get_transforms(train=True)
            dataset = _TransformSubset(self._raw_train, self._train_indices[task_id], transform)
        else:
            dataset = _TransformSubset(self._raw_test, self._test_indices[task_id], get_transforms(train=False))

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def get_classes(self, task_id: int) -> Tuple[int, int]:
        return self.task_classes[task_id]

    def get_class_names(self, task_id: int) -> Tuple[str, str]:
        c0, c1 = self.task_classes[task_id]
        return CLASS_NAMES[c0], CLASS_NAMES[c1]

    def task_size(self, task_id: int, train: bool = True) -> int:
        indices = self._train_indices if train else self._test_indices
        return len(indices[task_id])
