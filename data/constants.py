"""Shared CIFAR-10 constants used by both dataset.py and buffer.py."""

from typing import List, Tuple

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
