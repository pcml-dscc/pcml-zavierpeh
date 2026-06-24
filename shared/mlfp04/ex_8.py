# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 8 — Deep Learning Foundations.

Contains: synthetic XOR data, synthetic Singapore-medical image data,
reusable training loops, gradient monitoring helpers, ModelVisualizer
output paths. Technique-specific code (model classes, per-file training
loops, scenario narratives) does NOT belong here — it lives per file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kailash_ml import ModelVisualizer

from shared import MLFPDataLoader
from shared.kailash_helpers import get_device, setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT — seeds, device, output dir
# ════════════════════════════════════════════════════════════════════════
setup_environment()
torch.manual_seed(42)
np.random.seed(42)
device = get_device()

OUTPUT_DIR = Path("outputs") / "ex8_deep_learning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared hyperparameters
N_FEATS_XOR = 4
N_XOR_SAMPLES = 200
N_IMG_SAMPLES = 5000
IMG_SIZE = 64
N_CHANNELS = 1
N_CLASSES = 5
BATCH_SIZE = 64

# Kailash visualiser (used by every phase 4 block)
viz = ModelVisualizer()

# ════════════════════════════════════════════════════════════════════════
# DATA — XOR toy problem (Tasks 1-3)
# ════════════════════════════════════════════════════════════════════════


def make_xor_data(
    n_samples: int = N_XOR_SAMPLES, n_features: int = N_FEATS_XOR, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate a synthetic XOR classification task.

    Label is XOR of the sign of features 0 and 1. Features 2..n-1 are noise.
    Returns (X_tensor, y_tensor, y_numpy) on CPU.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float32)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(1)
    return X_t, y_t, y


# ════════════════════════════════════════════════════════════════════════
# DATA — Synthetic Singapore Hospital imaging tensors (Tasks 4-10)
# ════════════════════════════════════════════════════════════════════════
# Scenario: NUH (National University Hospital) chest-film triage. The real
# pipeline uses anonymised 512x512 DICOMs; this exercise uses 64x64 random
# tensors with the same multi-label structure so training completes in
# minutes on a laptop CPU / Colab T4.

SG_HOSPITAL_CLASSES = [
    "pneumonia",
    "effusion",
    "atelectasis",
    "nodule",
    "normal",
]


def make_sg_imaging_data(
    n_samples: int = N_IMG_SAMPLES, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_images, y_labels) as float32 numpy arrays.

    X: (N, 1, 64, 64) — simulated single-channel chest film tensors.
    y: (N, 5) — multi-label (~15% positive per class).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, N_CHANNELS, IMG_SIZE, IMG_SIZE)).astype(
        np.float32
    )
    y = (rng.random((n_samples, N_CLASSES)) > 0.85).astype(np.float32)
    return X, y


def build_sg_loaders(
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Produce (train_loader, test_loader, X_test_np, y_test_np) for the CNN tasks."""
    X, y = make_sg_imaging_data()
    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader, X_te, y_te


# ════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ════════════════════════════════════════════════════════════════════════


def train_xor_net(
    net: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimiser: torch.optim.Optimizer,
    n_epochs: int = 100,
    criterion: nn.Module | None = None,
) -> list[float]:
    """Fit a small binary classifier to XOR data. Returns per-epoch loss."""
    crit = criterion or nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for _ in range(n_epochs):
        optimiser.zero_grad()
        loss = crit(net(X), y)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    return losses


def xor_accuracy(net: nn.Module, X: torch.Tensor, y_np: np.ndarray) -> float:
    """Binary accuracy on XOR data (threshold at 0.5)."""
    net.eval()
    with torch.no_grad():
        probs = torch.sigmoid(net(X)).numpy().flatten()
    return float(((probs > 0.5) == y_np).mean())


def grad_norm(model: nn.Module) -> float:
    """L2 norm of the concatenated gradient vector."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return float(total**0.5)


def train_cnn_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    clip_value: float | None = None,
) -> tuple[float, float]:
    """Train for one epoch on the Singapore imaging loader.

    Returns (mean_loss, mean_grad_norm). If ``clip_value`` is set, the grad
    norm is measured pre-clipping and ``clip_grad_norm_`` is applied.
    """
    model.train()
    losses: list[float] = []
    grads: list[float] = []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimiser.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        grads.append(grad_norm(model))
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimiser.step()
        losses.append(loss.item())
    return float(np.mean(losses)), float(np.mean(grads))


def eval_cnn(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    """Return mean validation loss across the loader."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            losses.append(criterion(model(X_b), y_b).item())
    return float(np.mean(losses))


# ════════════════════════════════════════════════════════════════════════
# CNN BUILDING BLOCKS (reused across files 03, 04, 05)
# ════════════════════════════════════════════════════════════════════════


class ResBlock(nn.Module):
    """Residual block: skip connection preserves gradient flow."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class TriageCNN(nn.Module):
    """CNN for multi-label Singapore hospital triage.

    Architecture: Conv32 -> ResBlock -> Conv64 -> ResBlock -> AdaptiveAvgPool
    -> Dropout -> Linear. Designed for the multi-label BCEWithLogitsLoss
    setup used throughout Exercise 8.
    """

    def __init__(self, n_classes: int = N_CLASSES, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResBlock(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResBlock(64),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.classifier(self.features(x))


def count_params(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ════════════════════════════════════════════════════════════════════════
# DATA LOADER ENTRY POINT
# ════════════════════════════════════════════════════════════════════════
# We expose an MLFPDataLoader handle so student files have a single import
# path even though the tensors are generated on the fly. Real datasets for
# CNN fine-tuning live in Module 5.
loader = MLFPDataLoader()
