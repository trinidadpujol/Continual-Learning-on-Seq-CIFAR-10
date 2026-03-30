"""
Visualization utilities for continual learning experiments.

Generates plots saved to imgs/ for the report:
  - Accuracy curves (Class-IL and Task-IL) vs. tasks learned
  - Forgetting curves per task
  - 2D embedding projections (t-SNE / UMAP) at different training stages
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

IMGS_DIR = Path(__file__).resolve().parent.parent / "imgs"


def plot_accuracy_curve(
    results: Dict[str, List[float]],
    scenario: str = "Class-IL",
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot accuracy vs. number of tasks learned for multiple methods.

    Args:
        results:    {method_name: [acc_after_task_1, ..., acc_after_task_N]}
        scenario:   "Class-IL" or "Task-IL" (used in title and filename).
        save_path:  Where to save the figure. Defaults to imgs/<scenario>.png.
    """
    # TODO: implement accuracy curve plot
    raise NotImplementedError


def plot_forgetting_curve(
    forgetting: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot forgetting per task for each method.

    Args:
        forgetting: {method_name: [forgetting_task_0, ..., forgetting_task_N-1]}
        save_path:  Where to save the figure.
    """
    # TODO: implement forgetting curve plot
    raise NotImplementedError


def plot_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stage: str = "final",
    method: str = "tsne",
    save_path: Optional[Path] = None,
) -> None:
    """
    Project backbone embeddings to 2D (t-SNE or UMAP) and plot by class.

    Args:
        model:     Backbone whose .encode() method returns feature vectors.
        loader:    DataLoader for the data to embed.
        device:    Torch device.
        stage:     Label for the training stage (e.g. "init", "mid", "final").
        method:    Dimensionality reduction method: "tsne" or "umap".
        save_path: Where to save the figure.
    """
    # TODO: implement embedding projection and scatter plot
    raise NotImplementedError


def plot_loss_curve(
    losses: List[float],
    title: str = "Training Loss",
    save_path: Optional[Path] = None,
) -> None:
    """Plot and save a training loss curve."""
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True)
    if save_path is None:
        save_path = IMGS_DIR / f"{title.lower().replace(' ', '_')}.png"
    IMGS_DIR.mkdir(exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
