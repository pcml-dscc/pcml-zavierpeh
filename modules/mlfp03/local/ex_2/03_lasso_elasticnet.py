# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2.3: Lasso (L1) and ElasticNet
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Fit Lasso at many α values and read the sparsity pattern
#   - Explain L1 geometry: diamond constraint, corner solutions, exact zeros
#   - Use Lasso as built-in feature selection
#   - Blend L1 and L2 with ElasticNet for correlated features
#   - Apply sparse selection to a Singapore insurance fraud scorecard
#
# PREREQUISITES:
#   - 02_ridge_regression.py
#
# ESTIMATED TIME: ~40 minutes
#
# TASKS (5-phase R10):
#   1. Theory — the L1 diamond and why it produces zeros
#   2. Build — Lasso + ElasticNet fits across α and l1_ratio
#   3. Train — sparsity trajectory + ElasticNet sweep
#   4. Visualise — regularisation path
#   5. Apply — AIA Singapore insurance fraud scorecard feature selection
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error

from kailash_ml import ModelVisualizer

from shared.mlfp03.ex_2 import (
    ALPHAS,
    load_credit_data,
    print_header,
    save_html_plot,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — L1 Regularisation
# ════════════════════════════════════════════════════════════════════════
#     min_β  ||y - Xβ||²  +  α · ||β||₁
#
# ||β||₁ ≤ c is a DIAMOND — the MSE ellipse first touches it at a corner,
# where one or more coordinates are EXACTLY zero. Built-in feature
# selection. Bayesian view: Laplace prior on β. ElasticNet mixes L1
# and L2 to keep correlated features together.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD Lasso + ElasticNet fits
# ════════════════════════════════════════════════════════════════════════

print_header("Lasso & ElasticNet on Singapore Credit Data")

X_train, y_train, X_test, y_test, feature_names = load_credit_data()
print(f"Train: {X_train.shape}  Features: {len(feature_names)}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN Lasso across the α sweep
# ════════════════════════════════════════════════════════════════════════

lasso_results: dict[float, dict[str, float]] = {}
for alpha in ALPHAS:
    # TODO: Instantiate Lasso(alpha=alpha, max_iter=10_000), fit, and
    # record train/test MSE, the L1 norm of β (sum of |β_i|), and the
    # number of exact-zero coefficients (|β|<1e-6).
    lasso = ____
    lasso.fit(____, ____)
    train_mse = ____
    test_mse = ____
    l1_norm = float(np.sum(np.abs(lasso.coef_)))
    n_zero = int(np.sum(np.abs(lasso.coef_) < 1e-6))
    lasso_results[alpha] = {
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "l1_norm": l1_norm,
        "n_zero": n_zero,
        "coef": lasso.coef_.copy(),
    }

print(
    f"\n{'alpha':>10} {'train MSE':>12} {'test MSE':>12} "
    f"{'||β||₁':>10} {'zeros':>8}  sparsity"
)
print("-" * 62)
for alpha, r in lasso_results.items():
    pct = r["n_zero"] / len(feature_names) * 100
    print(
        f"{alpha:>10.3f} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} "
        f"{r['l1_norm']:>10.4f} {r['n_zero']:>8}  {pct:>5.0f}%"
    )

best_alpha_lasso, best_lasso_row = min(
    lasso_results.items(), key=lambda x: x[1]["test_mse"]
)
print(
    f"\nBest Lasso α = {best_alpha_lasso}  "
    f"(test MSE = {best_lasso_row['test_mse']:.4f}, "
    f"zeros = {best_lasso_row['n_zero']})"
)

# Show surviving features at best α
best_lasso_model = Lasso(alpha=best_alpha_lasso, max_iter=10_000).fit(X_train, y_train)
kept = sorted(
    [
        (name, coef)
        for name, coef in zip(feature_names, best_lasso_model.coef_)
        if abs(coef) > 1e-6
    ],
    key=lambda x: abs(x[1]),
    reverse=True,
)
print(f"\nLasso-selected features ({len(kept)}):")
for name, coef in kept[:10]:
    sign = "+" if coef > 0 else "-"
    print(f"  {sign} {name:<34} β = {coef:+.4f}")


# ── Checkpoint 1 ───────────────────────────────────────────────────────
assert (
    lasso_results[100.0]["n_zero"] > lasso_results[0.001]["n_zero"]
), "Higher-α Lasso must zero out MORE coefficients"
print("\n[ok] Checkpoint 1 passed — Lasso sparsity increases with α")


# ════════════════════════════════════════════════════════════════════════
# TASK 3b — TRAIN ElasticNet across l1_ratio
# ════════════════════════════════════════════════════════════════════════

print_header("ElasticNet — Mixing L1 and L2 (α fixed at 0.1)")
en_results: dict[float, dict[str, float]] = {}
for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
    # TODO: Instantiate ElasticNet(alpha=0.1, l1_ratio=l1_ratio,
    # max_iter=10_000) and record MSEs + zero-count same as Lasso.
    en = ____
    en.fit(____, ____)
    train_mse = ____
    test_mse = ____
    n_zero = int(np.sum(np.abs(en.coef_) < 1e-6))
    en_results[l1_ratio] = {
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "n_zero": n_zero,
    }

print(f"\n{'l1_ratio':>10} {'train MSE':>12} {'test MSE':>12} {'zeros':>8}")
print("-" * 46)
for l1_ratio, r in en_results.items():
    print(
        f"{l1_ratio:>10.1f} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} "
        f"{r['n_zero']:>8}"
    )


# ── Checkpoint 2 ───────────────────────────────────────────────────────
assert (
    en_results[0.9]["n_zero"] >= en_results[0.1]["n_zero"]
), "l1_ratio=0.9 should zero at least as many coefficients as 0.1"
print("\n[ok] Checkpoint 2 passed — l1_ratio drives sparsity monotonically")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the regularisation path
# ════════════════════════════════════════════════════════════════════════

print_header("Regularisation Path Visualisation")
viz = ModelVisualizer()

# TODO: Build the L1-norm trajectory plot. Use viz.training_history
# with a dict {"||β||₁ (Lasso)": [lasso_results[a]["l1_norm"] for a
# in ALPHAS]} and x_label="Regularisation Strength (α)".
fig_l1 = ____
fig_l1.update_layout(
    title="Lasso: L1 Norm of Coefficients vs α",
    xaxis_type="log",
)
path_l1 = save_html_plot(fig_l1, "lasso_l1_norm_path.html")

# TODO: Build the sparsity plot showing (len(feature_names) - n_zero)
# for each alpha (the number of NON-ZERO coefficients).
fig_sparse = ____
fig_sparse.update_layout(
    title="Lasso: Sparsity vs α — features dropping to zero",
    xaxis_type="log",
)
path_sparse = save_html_plot(fig_sparse, "lasso_sparsity_path.html")

print(f"\nSaved: {path_l1.name}")
print(f"Saved: {path_sparse.name}")


# ── Checkpoint 3 ───────────────────────────────────────────────────────
coef_matrix = np.array([lasso_results[a]["coef"] for a in ALPHAS])
assert coef_matrix.shape == (
    len(ALPHAS),
    len(feature_names),
), "Coefficient matrix should be (n_alphas, n_features)"
print("\n[ok] Checkpoint 3 passed — regularisation path matrix shape correct")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: AIA Singapore Insurance Fraud Scorecard
# ════════════════════════════════════════════════════════════════════════
# AIA Singapore has 240 candidate fraud features but only ~40 are
# "memo-able" for MAS compliance. Lasso picks the subset automatically.
# Business impact: ~S$360K/year saved in compliance, ~S$9.1M/year in
# extra fraud recovery, ~S$800K/year in dropped ETL cost.

print_header("AIA Singapore Fraud Scorecard — Lasso Feature Selection")
print(
    """
Model                      | Features | Annual fraud savings | Compliance cost
---------------------------|----------|----------------------|-----------------
Legacy (hand-picked, 12)   |    12    |         (baseline)   |     S$72K
Lasso (best α, ~40)        |    40    |         +S$9.1M      |    S$240K
Unregularised OLS (240)    |   240    |         +S$9.4M      |    S$1.44M
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print(
    """
======================================================================
  WHAT YOU'VE MASTERED
======================================================================

  [x] Lasso objective: ||y-Xβ||² + α·||β||₁
  [x] Diamond geometry → corner solutions → exact zeros
  [x] ElasticNet (α, l1_ratio) for correlated-feature stability
  [x] Regularisation path as a diagnostic visual

  NEXT: 04_cross_validation.py — nested, time-series, and group CV.
"""
)
