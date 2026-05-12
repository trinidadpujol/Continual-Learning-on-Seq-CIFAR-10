# Continual Learning on Seq-CIFAR-10

TP1 — I309 Visión Artificial Avanzada · Universidad de San Andrés

## Overview

This project implements and compares continual learning methods on **Sequential CIFAR-10** (Seq-CIFAR-10): CIFAR-10 split into 5 sequential tasks of 2 classes each.
The goal is to mitigate **catastrophic forgetting** while learning tasks one at a time.

### Methods implemented
| Method | Description |
|--------|-------------|
| Naive Fine-Tuning | Baseline — re-trains with CE, no forgetting prevention |
| EWC | Elastic Weight Consolidation (Kirkpatrick et al., 2017) |
| LwF | Learning without Forgetting (Li & Hoiem, 2018) |
| Co²L | Contrastive Continual Learning (Cha et al., ICCV 2021) |

### Evaluation scenarios
- **Class-IL**: model classifies among all 10 classes (no task hint at test time)
- **Task-IL**: model knows the task id and classifies within the 2-class subset

## Repository Structure

```
.
├── data/
│   ├── __init__.py
│   ├── constants.py        # TASK_CLASSES, CLASS_NAMES
│   ├── dataset.py          # SeqCIFAR10, transforms, TwoViewTransform
│   └── buffer.py           # ReplayBuffer
├── models/
│   ├── __init__.py
│   └── backbone.py         # ResNet-18 encoder + projection + classifier head
├── losses/
│   ├── __init__.py
│   ├── supcon.py           # Supervised Contrastive Loss
│   └── distillation.py     # Asymmetric Distillation Loss (Co²L)
├── methods/
│   ├── __init__.py
│   ├── base.py             # Abstract BaseMethod
│   ├── naive.py            # Naive Fine-Tuning
│   ├── ewc.py              # EWC
│   ├── lwf.py              # LwF
│   ├── co2l.py             # Co²L
│   ├── pretrain.py         # pretrain_supcon, train_linear_probe
│   └── trainer.py          # ContinualTrainer (sequential task orchestration)
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # evaluate_class_il, evaluate_task_il, MetricsTracker
│   └── visualization.py    # Accuracy/forgetting curves, t-SNE/UMAP plots
├── checkpoints/            # Saved model weights and result JSONs (created on first run)
├── imgs/                   # Output figures (created on first run)
├── tp1.ipynb               # Main entry point — run all experiments here
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook tp1.ipynb
```

CIFAR-10 is downloaded automatically by torchvision on the first run (into `./data/`).
No manual dataset setup is required.

## Running on Google Colab

The notebook includes a Colab-aware setup cell that:
- Clones the repo if running in `/content`
- Mounts Google Drive and syncs checkpoints to/from `MyDrive/TP1_checkpoints/`

This means training can be interrupted and resumed across Colab sessions — each method saves a per-task checkpoint and skips completed tasks on restart.

## Running Experiments

All experiments are run from `tp1.ipynb`.
The notebook is structured to match the four assignment stages:

1. **Dataset** — build `SeqCIFAR10` task loaders and `ReplayBuffer`
2. **Pre-training** — SupCon pre-training on Task 0 + linear probe
3. **CL methods** — sequential training of Naive / EWC / LwF / Co²L
4. **Comparison** — accuracy curves, forgetting curves, summary table

Each method saves intermediate checkpoints under `checkpoints/` and skips training if results are already saved, so you can re-run cells without redoing expensive training.

## Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Classes | `PascalCase` | `SeqCIFAR10`, `SupConLoss` |
| Functions / variables | `snake_case` | `evaluate_class_il` |
| Files | `snake_case.py` | `dataset.py` |
| Constants | `UPPER_SNAKE_CASE` | `TASK_CLASSES` |

## References

- Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*, PNAS 2017
- Li & Hoiem, *Learning without Forgetting*, TPAMI 2018
- Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020
- Cha et al., *Co²L: Contrastive Continual Learning*, ICCV 2021
