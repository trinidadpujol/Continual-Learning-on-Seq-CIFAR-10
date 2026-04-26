"""
Pre-entrenamiento del backbone con Supervised Contrastive Learning y linear probe.

Flujo de dos etapas:
  1. pretrain_supcon()    — encoder + projector con SupConLoss sobre la Tarea 0
  2. train_linear_probe() — encoder frozen, entrena solo el clasificador lineal con CE

Uso típico:
    pt = pretrain_supcon(backbone, seq_cifar, task_id=0, n_epochs=200, device=device)
    probe = train_linear_probe(backbone, seq_cifar, task_id=0, n_epochs=100, device=device)
    print(probe["test_acc"])
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from data.dataset import SeqCIFAR10
from losses.supcon import SupConLoss
from models.backbone import Backbone

logger = logging.getLogger(__name__)


def pretrain_supcon(
    backbone: Backbone,
    seq_cifar: SeqCIFAR10,
    task_id: int = 0,
    n_epochs: int = 200,
    lr: float = 0.5,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 0,
    save_stages: bool = False,
) -> Dict[str, object]:
    """Pre-entrenamiento con SupCon loss sobre task_id (default: Tarea 0).

    Solo se actualizan encoder y projector; el clasificador no recibe gradientes
    en esta fase y se puede reinicializar después con backbone.reset_classifier().

    save_stages=True guarda snapshots en epoch 0, mitad, y final para las
    visualizaciones de embeddings con t-SNE.

    Devuelve dict con loss_history, checkpoint_path y (si save_stages) stage_paths.
    """
    if device is None:
        device = next(backbone.parameters()).device

    backbone = backbone.to(device)
    backbone.train()

    loader = seq_cifar.get_task_loader(task_id, train=True, supcon=True)
    c0, c1 = seq_cifar.get_class_names(task_id)
    logger.info(
        "[pretrain_supcon] task=%d  classes=(%s, %s)  n_epochs=%d  lr=%.4f",
        task_id, c0, c1, n_epochs, lr,
    )

    criterion = SupConLoss(temperature=temperature).to(device)

    # Solo encoder + projector: no tocamos el clasificador en esta fase
    supcon_params = list(backbone.encoder.parameters()) + \
                    list(backbone.projector.parameters())
    optimizer = torch.optim.SGD(
        supcon_params, lr=lr, momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Snapshot inicial (epoch 0, antes de cualquier entrenamiento)
    stage_paths: Dict[str, str] = {}
    if save_stages:
        init_path = os.path.join(checkpoint_dir, f"supcon_stage_init_task{task_id}.pt")
        _save_checkpoint(backbone, 0, [], init_path, temperature)
        stage_paths["init"] = init_path
        logger.info("[pretrain_supcon] stage checkpoint (init) → %s", init_path)

    mid_epoch = n_epochs // 2

    loss_history: List[float] = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for x, labels in loader:
            x      = x.to(device)
            labels = labels.to(device)

            features = backbone.forward_supcon(x)   # (B, 2, proj_dim)
            loss     = criterion(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        mean_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(mean_loss)

        logger.info(
            "[pretrain_supcon] epoch=%d/%d  loss=%.4f  lr=%.6f",
            epoch + 1, n_epochs, mean_loss, scheduler.get_last_lr()[0],
        )

        # Snapshot del punto medio
        if save_stages and (epoch + 1) == mid_epoch:
            mid_path = os.path.join(checkpoint_dir, f"supcon_stage_mid_task{task_id}.pt")
            _save_checkpoint(backbone, epoch + 1, loss_history, mid_path, temperature)
            stage_paths["mid"] = mid_path
            logger.info("[pretrain_supcon] stage checkpoint (mid) → %s", mid_path)

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(
                checkpoint_dir, f"supcon_task{task_id}_epoch{epoch + 1}.pt"
            )
            _save_checkpoint(backbone, epoch + 1, loss_history, ckpt_path, temperature)
            logger.info("[pretrain_supcon] intermediate checkpoint → %s", ckpt_path)

    # Checkpoint final
    final_path = os.path.join(checkpoint_dir, f"supcon_pretrained_task{task_id}.pt")
    _save_checkpoint(backbone, n_epochs, loss_history, final_path, temperature)
    logger.info("[pretrain_supcon] final checkpoint → %s", final_path)

    result: Dict[str, object] = {
        "loss_history":    loss_history,
        "checkpoint_path": final_path,
    }
    if save_stages:
        stage_paths["final"] = final_path
        result["stage_paths"] = stage_paths
    return result


def train_linear_probe(
    backbone: Backbone,
    seq_cifar: SeqCIFAR10,
    task_id: int = 0,
    n_epochs: int = 100,
    lr: float = 1.0,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, object]:
    """Entrena solo el clasificador lineal con el encoder frozen.

    Debe llamarse después de pretrain_supcon(). El encoder se congela antes de
    empezar y se verifica en cada época para detectar cualquier problema. El
    clasificador se reinicializa al inicio (Kaiming-uniform) para no heredar nada
    de la fase contrastiva.

    Devuelve dict con loss_history, train_acc_history, test_acc, checkpoint_path.

    Lanza RuntimeError si el encoder se descongeló durante el entrenamiento.
    """
    if device is None:
        device = next(backbone.parameters()).device

    backbone = backbone.to(device)

    backbone.freeze_encoder()
    backbone.reset_classifier()

    if not backbone.is_encoder_frozen:
        raise RuntimeError(
            "freeze_encoder() fue llamado pero is_encoder_frozen sigue siendo False."
        )

    c0_name, c1_name = seq_cifar.get_class_names(task_id)
    logger.info(
        "[train_linear_probe] task=%d  classes=(%s, %s)  n_epochs=%d  lr=%.4f",
        task_id, c0_name, c1_name, n_epochs, lr,
    )

    train_loader = seq_cifar.get_task_loader(task_id, train=True,  supcon=False)
    test_loader  = seq_cifar.get_task_loader(task_id, train=False)

    # Solo optimizamos la cabeza lineal
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        backbone.classifier.parameters(),
        lr=lr, momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    os.makedirs(checkpoint_dir, exist_ok=True)

    loss_history:      List[float] = []
    train_acc_history: List[float] = []

    for epoch in range(n_epochs):
        # Verificación de seguridad: el encoder no debe descongelarse nunca
        if not backbone.is_encoder_frozen:
            raise RuntimeError(
                f"El encoder se descongeló en la época {epoch + 1}. "
                "Abortando para evitar contaminación del encoder."
            )

        backbone.eval()             # BN/Dropout en modo eval
        backbone.classifier.train() # solo la cabeza en modo train

        epoch_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                h = backbone._extract_features(x)   # encoder frozen, sin gradientes

            logits = backbone.classifier(h)
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)

        scheduler.step()

        mean_loss = epoch_loss / max(len(train_loader), 1)
        train_acc = correct / max(total, 1)
        loss_history.append(mean_loss)
        train_acc_history.append(train_acc)

        logger.info(
            "[train_linear_probe] epoch=%d/%d  loss=%.4f  train_acc=%.3f  lr=%.6f",
            epoch + 1, n_epochs, mean_loss, train_acc, scheduler.get_last_lr()[0],
        )

    # Evaluación final en test
    backbone.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = backbone(x)
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)
    test_acc = correct / max(total, 1)

    logger.info(
        "[train_linear_probe] task=%d  test_acc=%.3f  (encoder frozen: %s)",
        task_id, test_acc, backbone.is_encoder_frozen,
    )

    ckpt_path = os.path.join(checkpoint_dir, f"linear_probe_task{task_id}.pt")
    torch.save(
        {
            "backbone_state_dict": backbone.state_dict(),
            "loss_history":        loss_history,
            "train_acc_history":   train_acc_history,
            "test_acc":            test_acc,
            "task_id":             task_id,
            "config": {
                "n_epochs":    n_epochs,
                "lr":          lr,
                "feat_dim":    backbone.feat_dim,
                "num_classes": backbone.num_classes,
            },
        },
        ckpt_path,
    )
    logger.info("[train_linear_probe] checkpoint guardado → %s", ckpt_path)

    return {
        "loss_history":      loss_history,
        "train_acc_history": train_acc_history,
        "test_acc":          test_acc,
        "checkpoint_path":   ckpt_path,
    }


def load_pretrained_backbone(
    checkpoint_path: str,
    backbone: Backbone,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Carga un state dict guardado por pretrain_supcon() en el backbone dado."""
    if device is None:
        device = next(backbone.parameters()).device

    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.to(device)

    logger.info(
        "[load_pretrained_backbone] cargado desde %s  (epoch=%d)",
        checkpoint_path, ckpt.get("epoch", -1),
    )
    return ckpt


def _save_checkpoint(
    backbone: Backbone,
    epoch: int,
    loss_history: List[float],
    path: str,
    temperature: float,
) -> None:
    torch.save(
        {
            "backbone_state_dict": backbone.state_dict(),
            "epoch":               epoch,
            "loss_history":        loss_history,
            "config": {
                "temperature": temperature,
                "feat_dim":    backbone.feat_dim,
                "proj_dim":    backbone.proj_dim,
                "num_classes": backbone.num_classes,
            },
        },
        path,
    )
