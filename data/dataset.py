"""
Sequential CIFAR-10 dataset for continual learning.

Splits CIFAR-10 into N sequential tasks (default N=5), each with 2 classes.
Also provides a fixed-size replay buffer with reservoir sampling.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# 10 CIFAR-10 classes grouped into 5 tasks (2 classes per task)
TASK_CLASSES: List[Tuple[int, int]] = [
    (0, 1),   # Task 0: airplane, automobile
    (2, 3),   # Task 1: bird, cat
    (4, 5),   # Task 2: deer, dog
    (6, 7),   # Task 3: frog, horse
    (8, 9),   # Task 4: ship, truck
]

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

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


def get_supcon_transforms() -> transforms.Compose:
    """Two-view augmentation transform for Supervised Contrastive Learning."""
    augment = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    class TwoViewTransform:
        def __call__(self, x):
            return augment(x), augment(x)

    return TwoViewTransform()


class TaskSubset(Dataset):
    """Dataset filtered to a specific pair of classes for one task."""

    def __init__(self, base_dataset: Dataset, class_pair: Tuple[int, int], transform=None):
        self.transform = transform
        indices = [
            i for i, (_, label) in enumerate(base_dataset)
            if label in class_pair
        ]
        self.data = [base_dataset[i] for i in indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img, label = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class SeqCIFAR10:
    """
    Sequential CIFAR-10 for continual learning.

    Usage:
        seq = SeqCIFAR10(data_root="../cifar-10", n_tasks=5)
        train_loader = seq.get_task_loader(task_id=0, train=True)
        test_loader  = seq.get_task_loader(task_id=0, train=False)
    """

    N_TASKS = 5
    CLASSES_PER_TASK = 2

    def __init__(
        self,
        data_root: str = "../cifar-10",
        n_tasks: int = N_TASKS,
        batch_size: int = 128,
        num_workers: int = 2,
        supcon_transforms: bool = False,
    ):
        self.data_root = data_root
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.supcon_transforms = supcon_transforms
        self.task_classes = TASK_CLASSES[:n_tasks]

        # TODO: load raw CIFAR-10 from data_root (extracted from train.7z / test.7z)
        # Placeholder — replace with actual loading logic once images are extracted
        self._train_dataset = None
        self._test_dataset = None

    def get_task_loader(self, task_id: int, train: bool = True) -> DataLoader:
        """Return a DataLoader for the given task split."""
        raise NotImplementedError("Implement after extracting CIFAR-10 images")

    def get_classes(self, task_id: int) -> Tuple[int, int]:
        return self.task_classes[task_id]


class ReplayBuffer:
    """
    Fixed-size replay buffer with reservoir sampling.

    Stores (image_tensor, label) pairs from past tasks.
    """

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._storage: List[Tuple[torch.Tensor, int]] = []
        self._seen = 0

    def update(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """Add a batch of samples using reservoir sampling."""
        for img, lbl in zip(images, labels):
            self._seen += 1
            if len(self._storage) < self.capacity:
                self._storage.append((img.cpu(), int(lbl)))
            else:
                j = random.randint(0, self._seen - 1)
                if j < self.capacity:
                    self._storage[j] = (img.cpu(), int(lbl))

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n items from the buffer (with replacement if needed)."""
        if len(self._storage) == 0:
            raise RuntimeError("Replay buffer is empty")
        items = random.choices(self._storage, k=min(n, len(self._storage)))
        imgs, labels = zip(*items)
        return torch.stack(imgs), torch.tensor(labels)

    def __len__(self) -> int:
        return len(self._storage)
