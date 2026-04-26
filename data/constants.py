"""Clases de CIFAR-10 divididas en 5 tareas secuenciales (2 clases por tarea)."""

from typing import List, Tuple

TASK_CLASSES: List[Tuple[int, int]] = [
    (0, 1),   # Tarea 0: airplane, automobile
    (2, 3),   # Tarea 1: bird, cat
    (4, 5),   # Tarea 2: deer, dog
    (6, 7),   # Tarea 3: frog, horse
    (8, 9),   # Tarea 4: ship, truck
]

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
