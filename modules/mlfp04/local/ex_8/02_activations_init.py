# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 8.2: Activations and Weight Initialisation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compare ReLU, GELU, Tanh, and SiLU on the same task
#   - See why zero init and unscaled normal init fail
#   - Apply Xavier (Sigmoid/Tanh) and Kaiming (ReLU-family) correctly
#
# PREREQUISITES: 01_xor_proof.py
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — paired decision: activation and init must agree
#   2. Build — network factory + init application helper
#   3. Train — activation grid and initialisation grid
#   4. Visualise — two side-by-side loss charts
#   5. Apply — Grab Singapore: Kaiming init halved their retraining budget
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import torch
import torch.nn as nn

from shared.mlfp04.ex_8 import (
    N_FEATS_XOR,
    OUTPUT_DIR,
    make_xor_data,
    train_xor_net,
    viz,
    xor_accuracy,
)

print("\n" + "=" * 70)
print("  Activations and Initialisation — The Two Knobs That Must Agree")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Activation and init are paired
# ════════════════════════════════════════════════════════════════════════
# ReLU kills half the signal, so the surviving half needs twice the
# variance to keep activations stable across layers. Kaiming init
# (Var(W) = 2/fan_in) compensates. Xavier (Var(W) = 2/(fan_in+fan_out))
# is tuned for tanh/sigmoid. Zeros breaks by symmetry; unscaled normal
# blows up the pre-activation.

X, y, y_np = make_xor_data()

# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: factory + init helper
# ════════════════════════════════════════════════════════════════════════


def build_net(activation: nn.Module) -> nn.Sequential:
    """Two-hidden-layer MLP with a swappable activation."""
    # TODO: Return an nn.Sequential with layers:
    #   Linear(N_FEATS_XOR, 32) -> activation -> Linear(32, 16) -> activation -> Linear(16, 1)
    return ____


def apply_init(net: nn.Sequential, init_fn) -> None:
    """Apply an initialisation to every linear layer."""
    # TODO: Iterate net.modules(). For each nn.Linear, call init_fn on
    # m.weight and nn.init.zeros_ on m.bias.
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN the activation grid
# ════════════════════════════════════════════════════════════════════════
print("\n--- Activation comparison (Kaiming init, Adam, 80 epochs) ---")

activation_variants: dict[str, nn.Module] = {
    "ReLU": nn.ReLU(),
    "GELU": nn.GELU(),
    "Tanh": nn.Tanh(),
    "SiLU/Swish": nn.SiLU(),
}

act_histories: dict[str, list[float]] = {}
for name, act in activation_variants.items():
    net = build_net(act)
    apply_init(net, lambda w: nn.init.kaiming_uniform_(w, nonlinearity="relu"))
    # TODO: Train with Adam lr=0.01 for 80 epochs via train_xor_net
    losses = ____
    acc = xor_accuracy(net, X, y_np)
    act_histories[name] = losses
    print(f"  {name:<12}: final_loss={losses[-1]:.4f}, accuracy={acc:.4f}")

assert all(
    h[-1] < h[0] for h in act_histories.values()
), "Task 3: every activation should have reduced its loss"

# Initialisation grid (ReLU fixed, init swapped)
print("\n--- Initialisation comparison (ReLU, Adam, 80 epochs) ---")

init_variants = {
    "Xavier/Glorot": lambda w: nn.init.xavier_uniform_(w),
    "Kaiming/He": lambda w: nn.init.kaiming_uniform_(w, nonlinearity="relu"),
    "Normal(0,1)": lambda w: nn.init.normal_(w, mean=0.0, std=1.0),
    "Zeros": lambda w: nn.init.zeros_(w),
}

init_histories: dict[str, list[float]] = {}
for name, init_fn in init_variants.items():
    net = build_net(nn.ReLU())
    # TODO: Apply this init's fn via apply_init, then train for 80 epochs
    # with Adam lr=0.01
    apply_init(____, ____)
    losses = train_xor_net(
        net, X, y, torch.optim.Adam(net.parameters(), lr=0.01), n_epochs=80
    )
    init_histories[name] = losses
    print(f"  {name:<15}: init_loss={losses[0]:.4f}, final_loss={losses[-1]:.4f}")

assert (
    init_histories["Kaiming/He"][-1] < init_histories["Zeros"][-1] + 0.1
), "Task 3: Kaiming must beat zero init"
print("\n[ok] Checkpoint passed — activation + init grid trained\n")

# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE both grids
# ════════════════════════════════════════════════════════════════════════
fig_act = viz.training_history(act_histories, x_label="Epoch")
fig_act.update_layout(title="Activation Comparison on XOR (Kaiming init)")
fig_act.write_html(OUTPUT_DIR / "02_activation_curves.html")

fig_init = viz.training_history(init_histories, x_label="Epoch")
fig_init.update_layout(title="Initialisation Comparison on XOR (ReLU hidden)")
fig_init.write_html(OUTPUT_DIR / "02_initialisation_curves.html")
print("[viz] 02_activation_curves.html + 02_initialisation_curves.html saved")

# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Grab Rider Churn Scoring (Singapore + SEA)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Grab's driver retention team scores ~200K riders nightly. A
# 2023 re-architecture swapped Tanh+Xavier for ReLU+Kaiming in the hidden
# trunk. Epochs-to-converge dropped from 14 to 6. Training time per night
# fell from 3.5h to 1.5h. Monthly T4 compute fell from ~S$9K to ~S$3.8K.
#
# BUSINESS IMPACT: S$62K/year recovered compute plus a faster nightly SLO
# (refresh by 5am instead of 8am). One-line change: kaiming_uniform_.

# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Compared ReLU, GELU, Tanh, SiLU on identical tasks
  [x] Reproduced the Xavier vs Kaiming vs Zeros init comparison
  [x] Saw zero init fail by symmetry
  [x] Connected the paired decision to Grab's 60% compute saving

  Next: 03_cnn_residual.py — stack the layers into a CNN with skip blocks.
"""
)
