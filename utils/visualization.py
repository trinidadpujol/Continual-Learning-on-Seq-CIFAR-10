"""
Visualization utilities for continual learning experiments.

Generates plots saved to imgs/ for the report:
  - Loss curve during SupCon pre-training
  - 2D embedding projections (t-SNE) at init / midpoint / final training stages
  - Linear probe accuracy summary
  - Accuracy curves (Class-IL and Task-IL) vs. tasks learned
  - Forgetting per task per method
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

IMGS_DIR = Path(__file__).resolve().parent.parent / "imgs"

# Consistent colour palette for up to 10 CIFAR-10 classes
_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _collect_features(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalised encoder features and labels from a DataLoader.

    Args:
        backbone:  Backbone with an ``.encode(x)`` method returning (B, feat_dim).
        loader:    DataLoader yielding (images, labels) batches.
        device:    Torch device.
        n_samples: Cap on total samples collected (None = all).

    Returns:
        features: (N, feat_dim) float32 array.
        labels:   (N,) int64 array.
    """
    backbone.eval()
    all_feats:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    collected = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # Handle two-view batches (B, 2, C, H, W) from SupCon loader
            if x.ndim == 5:
                x = x[:, 0]           # take first view only
            feats = backbone.encode(x).cpu().numpy()
            all_feats.append(feats)
            all_labels.append(y.numpy())
            collected += len(y)
            if n_samples is not None and collected >= n_samples:
                break

    features = np.concatenate(all_feats,  axis=0)
    labels   = np.concatenate(all_labels, axis=0)
    if n_samples is not None:
        features = features[:n_samples]
        labels   = labels[:n_samples]
    return features, labels


def _run_tsne(features: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    """Reduce features to 2D with t-SNE."""
    from sklearn.manifold import TSNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(features) - 1),
        random_state=42,
        max_iter=1000,
    )
    return tsne.fit_transform(features)


def _scatter_2d(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str,
) -> None:
    """Scatter plot coloured by class on the given axes."""
    unique = sorted(set(labels.tolist()))
    for cls in unique:
        mask = labels == cls
        name = class_names[cls] if cls < len(class_names) else str(cls)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=_PALETTE[cls % len(_PALETTE)],
            label=name,
            s=8, alpha=0.7, linewidths=0,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", markerscale=2, fontsize=8)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[visualization] saved → %s", path)


# ── Pre-training visualizations ───────────────────────────────────────────────

def plot_pretrain_loss(
    loss_history: List[float],
    save_path: Optional[Path] = None,
    title: str = "SupCon Pre-training Loss — Task 0",
) -> None:
    """Plot and save the SupCon training loss curve.

    Args:
        loss_history: Per-epoch mean loss values returned by pretrain_supcon().
        save_path:    Where to save.  Defaults to imgs/supcon_pretrain_loss.png.
        title:        Figure title.
    """
    if save_path is None:
        save_path = IMGS_DIR / "supcon_pretrain_loss.png"

    epochs = list(range(1, len(loss_history) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, loss_history, linewidth=1.5, color="#4e79a7", label="train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SupCon Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    _save(fig, Path(save_path))


def plot_embedding_stages(
    stage_paths: Dict[str, str],
    backbone_template: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    save_path: Optional[Path] = None,
    n_samples: int = 1000,
    method: str = "tsne",
) -> None:
    """Visualise backbone embeddings at init / midpoint / final training stage.

    Loads each stage checkpoint, extracts features, reduces to 2D with t-SNE,
    and saves a 1×3 subplot figure.

    Args:
        stage_paths:       Dict with keys "init", "mid", "final" mapping to
                           .pt checkpoint paths produced by pretrain_supcon
                           with save_stages=True.
        backbone_template: A Backbone instance used as a template (its weights
                           are NOT modified — a deep copy is made internally).
        loader:            DataLoader for the data to embed (test split is ideal).
        device:            Torch device.
        class_names:       List of string class names indexed by integer label.
        save_path:         Where to save.  Defaults to
                           imgs/supcon_embeddings_stages.png.
        n_samples:         Maximum samples to embed (for speed).
        method:            "tsne" (default) or "umap" (requires umap-learn).
    """
    if save_path is None:
        save_path = IMGS_DIR / "supcon_embeddings_stages.png"

    stage_order  = ["init", "mid", "final"]
    stage_labels = ["Init (epoch 0)", f"Mid (epoch {_guess_mid(stage_paths)})", "Final"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, key, label in zip(axes, stage_order, stage_labels):
        if key not in stage_paths:
            ax.set_visible(False)
            continue

        # Load weights into a temporary copy — never modify the caller's backbone
        probe = copy.deepcopy(backbone_template)
        ckpt  = torch.load(stage_paths[key], map_location=device)
        probe.load_state_dict(ckpt["backbone_state_dict"])
        probe = probe.to(device)

        feats, lbls = _collect_features(probe, loader, device, n_samples=n_samples)

        logger.info(
            "[plot_embedding_stages] %s: %d samples, running %s…",
            key, len(feats), method.upper(),
        )

        if method == "umap":
            try:
                import umap as umap_lib
                reducer = umap_lib.UMAP(n_components=2, random_state=42)
                coords  = reducer.fit_transform(feats)
            except ImportError:
                logger.warning("umap-learn not installed, falling back to t-SNE")
                coords = _run_tsne(feats)
        else:
            coords = _run_tsne(feats)

        _scatter_2d(ax, coords, lbls, class_names, title=label)

        del probe  # free memory between stages

    fig.suptitle("Backbone embeddings — Task 0 (t-SNE)", fontsize=13)
    fig.tight_layout()
    _save(fig, Path(save_path))


def _guess_mid(stage_paths: Dict[str, str]) -> str:
    """Try to read the epoch number from the midpoint checkpoint."""
    path = stage_paths.get("mid", "")
    if path:
        try:
            ckpt = torch.load(path, map_location="cpu")
            return str(ckpt.get("epoch", "?"))
        except Exception:
            pass
    return "?"


def plot_probe_accuracy(
    probe_history: Dict[str, object],
    save_path: Optional[Path] = None,
    title: str = "Linear Probe — Task 0",
) -> None:
    """Plot linear probe training curves and final test accuracy.

    Args:
        probe_history: Dict returned by train_linear_probe() with keys
                       ``loss_history``, ``train_acc_history``, ``test_acc``.
        save_path:     Where to save.  Defaults to imgs/linear_probe_curves.png.
        title:         Figure title prefix.
    """
    if save_path is None:
        save_path = IMGS_DIR / "linear_probe_curves.png"

    loss_hist = probe_history["loss_history"]
    acc_hist  = probe_history["train_acc_history"]
    test_acc  = probe_history["test_acc"]
    epochs    = list(range(1, len(loss_hist) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, loss_hist, linewidth=1.5, color="#4e79a7")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("CE Loss")
    ax1.set_title(f"{title} — CE Loss"); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, acc_hist, linewidth=1.5, color="#f28e2b", label="train acc")
    ax2.axhline(test_acc, color="#e15759", linestyle="--", linewidth=1.5,
                label=f"test acc = {test_acc:.3f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} — Accuracy"); ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    _save(fig, Path(save_path))


# ── Continual learning comparison plots ──────────────────────────────────────

def plot_accuracy_curve(
    results: Dict[str, List[float]],
    scenario: str = "Class-IL",
    save_path: Optional[Path] = None,
) -> None:
    """Plot accuracy vs. number of tasks learned for multiple methods.

    Args:
        results:   {method_name: [avg_acc_after_task_0, ..., avg_acc_after_task_N]}
        scenario:  "Class-IL" or "Task-IL" (used in title and filename).
        save_path: Where to save.  Defaults to imgs/<scenario>_accuracy.png.
    """
    if save_path is None:
        fname = scenario.lower().replace("-", "_") + "_accuracy.png"
        save_path = IMGS_DIR / fname

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v"]

    for i, (method, accs) in enumerate(results.items()):
        tasks = list(range(1, len(accs) + 1))
        ax.plot(
            tasks, accs,
            marker=markers[i % len(markers)],
            linewidth=1.8, markersize=6,
            label=method,
        )

    ax.set_xlabel("Tasks learned")
    ax.set_ylabel("Average accuracy")
    ax.set_title(f"{scenario} — Average Accuracy vs. Tasks Learned")
    ax.set_xticks(list(range(1, max(len(v) for v in results.values()) + 1)))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, Path(save_path))


def plot_forgetting_curve(
    forgetting: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> None:
    """Plot forgetting per task for each method.

    Args:
        forgetting: {method_name: [forgetting_task_0, ..., forgetting_task_{N-2}]}
                    (task N-1 has no forgetting — it's the last task trained).
        save_path:  Where to save.  Defaults to imgs/forgetting.png.
    """
    if save_path is None:
        save_path = IMGS_DIR / "forgetting.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v"]

    for i, (method, forg) in enumerate(forgetting.items()):
        tasks = list(range(len(forg)))
        ax.plot(
            tasks, forg,
            marker=markers[i % len(markers)],
            linewidth=1.8, markersize=6,
            label=method,
        )

    ax.set_xlabel("Task index")
    ax.set_ylabel("Forgetting")
    ax.set_title("Forgetting per Task (higher = worse)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, Path(save_path))


def plot_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stage: str = "final",
    method: str = "tsne",
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    n_samples: int = 1000,
) -> None:
    """Project backbone embeddings to 2D and plot by class.

    Args:
        model:       Backbone whose ``.encode()`` returns feature vectors.
        loader:      DataLoader for the data to embed.
        device:      Torch device.
        stage:       Label for the training stage (used in title and filename).
        method:      "tsne" or "umap".
        class_names: List of string class names.  Defaults to int strings.
        save_path:   Where to save.  Defaults to imgs/embeddings_<stage>.png.
        n_samples:   Cap on total samples collected.
    """
    if save_path is None:
        save_path = IMGS_DIR / f"embeddings_{stage}.png"

    feats, lbls = _collect_features(model, loader, device, n_samples=n_samples)

    if method == "umap":
        try:
            import umap as umap_lib
            reducer = umap_lib.UMAP(n_components=2, random_state=42)
            coords  = reducer.fit_transform(feats)
        except ImportError:
            logger.warning("umap-learn not installed, falling back to t-SNE")
            coords = _run_tsne(feats)
    else:
        coords = _run_tsne(feats)

    if class_names is None:
        class_names = [str(c) for c in range(max(lbls) + 1)]

    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_2d(ax, coords, lbls, class_names, title=f"Embeddings — {stage}")
    fig.tight_layout()
    _save(fig, Path(save_path))


def plot_loss_curve(
    losses: List[float],
    title: str = "Training Loss",
    save_path: Optional[Path] = None,
) -> None:
    """Plot and save a generic training loss curve."""
    if save_path is None:
        save_path = IMGS_DIR / f"{title.lower().replace(' ', '_')}.png"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, Path(save_path))
