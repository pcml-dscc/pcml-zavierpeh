# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 8.4: Optimisers and Learning-Rate Schedulers
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compare SGD, SGD+momentum, Adam, and AdamW on the same task
#   - Apply a cosine-annealing LR schedule with a linear warmup
#   - Track LR and loss together during training
#
# PREREQUISITES: 03_cnn_residual.py
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — optimisers as adaptive gradient rescaling
#   2. Build — optimiser grid and a warmup+cosine scheduler
#   3. Train — short runs + one full run with the scheduler
#   4. Visualise — optimiser curves + schedule trajectory
#   5. Apply — Sea Group fraud model: warmup saved a production rollout
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim

from shared.mlfp04.ex_8 import (
    OUTPUT_DIR,
    SG_HOSPITAL_CLASSES,
    TriageCNN,
    build_sg_loaders,
    device,
    eval_cnn,
    train_cnn_one_epoch,
    viz,
)

print("\n" + "=" * 70)
print("  Optimisers and Schedulers — How Fast Can You Converge?")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Optimisers as per-parameter learning rates
# ════════════════════════════════════════════════════════════════════════
# Adam normalises each parameter by its own gradient variance. AdamW
# fixes weight decay to behave as a true regulariser. Cosine annealing
# reduces LR smoothly; linear warmup prevents Adam from blowing up
# before its variance estimates have stabilised.

train_loader, test_loader, _, _ = build_sg_loaders()
criterion = nn.BCEWithLogitsLoss()
n_classes = len(SG_HOSPITAL_CLASSES)


def fresh_model() -> TriageCNN:
    return TriageCNN(n_classes=n_classes, dropout_rate=0.3).to(device)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD optimiser grid + scheduler factory
# ════════════════════════════════════════════════════════════════════════

# TODO: Fill in the missing lambdas so each value is a callable taking
# `params` and returning the configured optimiser.
optimiser_builders: dict[str, callable] = {
    "SGD lr=0.01": lambda p: optim.SGD(p, lr=0.01),
    "SGD+momentum": lambda p: ____,  # SGD lr=0.01, momentum=0.9
    "Adam lr=1e-3": lambda p: ____,  # Adam lr=1e-3
    "AdamW lr=1e-3": lambda p: ____,  # AdamW lr=1e-3, weight_decay=1e-4
}


def make_warmup_cosine(
    optimiser: optim.Optimizer, warmup_epochs: int, total_epochs: int
) -> optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay to zero."""

    def lr_lambda(epoch: int) -> float:
        # TODO: For the first `warmup_epochs` epochs return a linearly
        # increasing multiplier (epoch+1)/warmup_epochs. After that,
        # return 0.5 * (1 + cos(pi * progress)) where progress is
        # (epoch - warmup_epochs) / (total_epochs - warmup_epochs).
        if epoch < warmup_epochs:
            return ____
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return ____

    return optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN the optimiser grid (5 epochs each)
# ════════════════════════════════════════════════════════════════════════
print("\n--- Optimiser comparison (5 epochs each) ---")
optimiser_histories: dict[str, list[float]] = {}
for name, builder in optimiser_builders.items():
    model = fresh_model()
    optimiser = builder(model.parameters())
    losses: list[float] = []
    for _ in range(5):
        loss, _ = train_cnn_one_epoch(model, train_loader, optimiser, criterion)
        losses.append(loss)
    optimiser_histories[name] = losses
    print(f"  {name:<18}: {losses[0]:.4f} -> {losses[-1]:.4f}")

assert all(
    h[-1] <= h[0] + 1e-3 for h in optimiser_histories.values()
), "Task 3: every optimiser should hold the line on train loss"

# Full run with scheduler
print("\n--- Full run: AdamW + warmup(2) + cosine(10) ---")
total_epochs = 10
warmup_epochs = 2

model = fresh_model()
# TODO: Create an AdamW optimiser with lr=1e-3, weight_decay=1e-4.
optimiser = ____
# TODO: Create the warmup+cosine scheduler via make_warmup_cosine.
scheduler = ____

schedule_history = {"train_loss": [], "val_loss": [], "lr": []}
for epoch in range(total_epochs):
    train_loss, _ = train_cnn_one_epoch(model, train_loader, optimiser, criterion)
    val_loss = eval_cnn(model, test_loader, criterion)
    schedule_history["train_loss"].append(train_loss)
    schedule_history["val_loss"].append(val_loss)
    schedule_history["lr"].append(scheduler.get_last_lr()[0])
    scheduler.step()
    print(
        f"  Epoch {epoch + 1:>2}/{total_epochs}: "
        f"train={train_loss:.4f} val={val_loss:.4f} "
        f"lr={schedule_history['lr'][-1]:.6f}"
    )

assert (
    schedule_history["lr"][-1] < schedule_history["lr"][warmup_epochs]
), "Task 3: cosine tail should be lower than warmup peak"
assert (
    schedule_history["lr"][0] < schedule_history["lr"][warmup_epochs]
), "Task 3: warmup should start below the peak LR"
print("\n[ok] Checkpoint passed — warmup + cosine behave as specified\n")

# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════
fig_opt = viz.training_history(optimiser_histories, x_label="Epoch")
fig_opt.update_layout(title="Optimiser Comparison on TriageCNN")
fig_opt.write_html(OUTPUT_DIR / "04_optimiser_curves.html")

fig_sched = viz.training_history(
    {
        "Train BCE": schedule_history["train_loss"],
        "Val BCE": schedule_history["val_loss"],
        "Learning Rate (x1e3)": [lr * 1000 for lr in schedule_history["lr"]],
    },
    x_label="Epoch",
)
fig_sched.update_layout(title="Warmup(2) + Cosine(10) — Loss and LR Together")
fig_sched.write_html(OUTPUT_DIR / "04_schedule_curves.html")
print("[viz] 04_optimiser_curves.html + 04_schedule_curves.html saved")

# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Sea Group (Shopee) Fraud Scoring
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Shopee Pay's v1 fraud scorer (SGD lr=0.1) diverged ~1 night
# in 10 during the hour-3 loss phase, triggering aborted rollouts worth
# ~S$1,200 each plus 4am pages. Switching to AdamW + 500-step warmup +
# cosine decay produced 0 aborted runs across 180 subsequent training
# nights.
#
# BUSINESS IMPACT: ~S$18K/month recovered aborted-run cost, ~60 eng-hours
# of on-call saved, enabling safe deployment of a S$140M/year fraud model.

# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Compared SGD, SGD+momentum, Adam, AdamW on identical budgets
  [x] Built a linear-warmup + cosine-annealing scheduler
  [x] Trained a CNN for 10 epochs tracking train/val/LR together
  [x] Connected the pattern to Sea Group's production rollout

  Next: 05_regularisation_training.py — dropout, BN, clip, early stop, ONNX.
"""
)
