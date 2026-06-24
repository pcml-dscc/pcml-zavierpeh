# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2.4: Cross-Validation Strategies
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Run nested CV and measure optimism bias vs standard CV
#   - Apply TimeSeriesSplit for walk-forward validation on temporal data
#   - Use GroupKFold so grouped observations stay together
#   - Pick the RIGHT CV strategy for a given deployment scenario
#
# PREREQUISITES:
#   - 02_ridge_regression.py and 03_lasso_elasticnet.py
#
# ESTIMATED TIME: ~45 minutes
#
# TASKS (5-phase R10):
#   1. Theory — why "one CV fits all" is wrong
#   2. Build — three CV splitters
#   3. Train — nested CV (unbiased α selection)
#   4. Visualise — CV-strategy comparison
#   5. Apply — GrabPay transaction fraud time-series validation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    TimeSeriesSplit,
    cross_val_score,
)

from shared.mlfp03.ex_2 import (
    ALPHAS,
    SEED,
    load_credit_data,
    print_header,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Cross-Validation Is More Than Just k-fold
# ════════════════════════════════════════════════════════════════════════
# k-fold assumes observations are i.i.d. AND that tuning and reporting
# can share splits. Real data breaks those assumptions:
#   - Temporal data → shuffling leaks future into training
#   - Grouped data → same entity in both folds leaks group identity
#   - Tuning + reporting → optimistic bias
# The fix: pick the CV strategy that matches the deployment scenario.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD data + splitters
# ════════════════════════════════════════════════════════════════════════

print_header("Cross-Validation Strategies on Singapore Credit Data")
X_train, y_train, X_test, y_test, feature_names = load_credit_data()
print(f"Train: {X_train.shape}  Features: {len(feature_names)}")

rng = np.random.default_rng(SEED)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN with nested CV
# ════════════════════════════════════════════════════════════════════════

print_header("Nested Cross-Validation")

# TODO: Fit a RidgeCV with alphas=ALPHAS, cv=5 on (X_train, y_train)
# and compute its test-set R². This is the biased baseline.
ridge_cv = ____
ridge_cv.fit(____, ____)
biased_score = float(ridge_cv.score(X_test, y_test))
print(
    f"Standard (biased) CV:  R² = {biased_score:.4f}, "
    f"selected α = {ridge_cv.alpha_:.4f}"
)

outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=SEED)

nested_scores: list[float] = []
selected_alphas: list[float] = []
print("\nOuter folds:")
for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X_train)):
    X_out_tr, X_out_te = X_train[tr_idx], X_train[te_idx]
    y_out_tr, y_out_te = y_train[tr_idx], y_train[te_idx]

    # TODO: Inner loop — for each alpha in ALPHAS, run
    # cross_val_score(Ridge(alpha=alpha), X_out_tr, y_out_tr,
    # cv=inner_cv, scoring="r2") and keep the α with the best mean.
    best_alpha = ALPHAS[0]
    best_inner = -np.inf
    for alpha in ALPHAS:
        inner = ____
        if inner.mean() > best_inner:
            best_inner = inner.mean()
            best_alpha = alpha

    # TODO: Fit Ridge(alpha=best_alpha) on the outer-train split and
    # score on the held-out outer-test split with r2_score.
    ridge_selected = ____
    ridge_selected.fit(____, ____)
    outer_score = float(r2_score(y_out_te, ridge_selected.predict(X_out_te)))
    nested_scores.append(outer_score)
    selected_alphas.append(best_alpha)
    print(
        f"  Fold {fold_idx + 1}: α = {best_alpha:<8.4f}  "
        f"outer R² = {outer_score:.4f}"
    )

nested_mean = float(np.mean(nested_scores))
nested_std = float(np.std(nested_scores))
print(f"\nNested CV:  R² = {nested_mean:.4f} ± {nested_std:.4f}")
print(f"Standard CV: R² = {biased_score:.4f}")
print(f"Optimism bias (standard - nested): {biased_score - nested_mean:+.4f}")


# ── Checkpoint 1 ───────────────────────────────────────────────────────
assert len(nested_scores) == 5, "Should have 5 outer fold scores"
assert all(isinstance(s, float) for s in nested_scores), "Scores must be floats"
print("\n[ok] Checkpoint 1 passed — nested CV unbiased estimate produced")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE time-series and group CV
# ════════════════════════════════════════════════════════════════════════

print_header("CV Strategy Comparison: k-fold vs Time-series vs Group")

# TODO: Compute 5-fold standard cross_val_score for Ridge(alpha=1.0).
kfold_scores = ____

# TODO: Build a TimeSeriesSplit(n_splits=5) and score Ridge through it.
tscv = ____
ts_scores = cross_val_score(Ridge(alpha=1.0), X_train, y_train, cv=tscv, scoring="r2")

print("\nTime-series walk-forward splits:")
for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_train)):
    print(
        f"  Fold {fold + 1}: train=[0:{tr_idx[-1] + 1}] "
        f"({len(tr_idx)} samples), test=[{te_idx[0]}:{te_idx[-1] + 1}] "
        f"({len(te_idx)} samples)"
    )

# Synthetic "customer_id" groups
n = X_train.shape[0]
groups = np.repeat(np.arange(n // 5 + 1), 5)[:n]
rng.shuffle(groups)

# TODO: Use GroupKFold(n_splits=5) via cross_val_score, passing
# groups=groups, to get per-group-honest scores.
group_cv = ____
group_scores = cross_val_score(
    Ridge(alpha=1.0),
    X_train,
    y_train,
    cv=group_cv,
    groups=groups,
    scoring="r2",
)

# Verify group integrity (must not overlap)
for fold, (tr_idx, te_idx) in enumerate(group_cv.split(X_train, groups=groups)):
    train_g = set(groups[tr_idx])
    test_g = set(groups[te_idx])
    overlap = train_g & test_g
    assert not overlap, f"Fold {fold + 1}: groups overlap {overlap}"
print("  [ok] GroupKFold: no group appears in both train and test")

print(
    f"""
Strategy                 R²            When to use
-----------------------  ------------  --------------------------------
Standard k-fold          {kfold_scores.mean():+.4f} ± {kfold_scores.std():.4f}  i.i.d. data, no groups
Nested CV                {nested_mean:+.4f} ± {nested_std:.4f}  Hyperparameter + report
TimeSeriesSplit          {ts_scores.mean():+.4f} ± {ts_scores.std():.4f}  Temporal data
GroupKFold               {group_scores.mean():+.4f} ± {group_scores.std():.4f}  Grouped observations
"""
)


# ── Checkpoint 2 ───────────────────────────────────────────────────────
assert (
    len(ts_scores) == 5 and len(group_scores) == 5
), "Each CV strategy should produce 5 scores"
print("[ok] Checkpoint 2 passed — all CV strategies produced 5 scores")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: GrabPay Transaction Fraud Time-Series Validation
# ════════════════════════════════════════════════════════════════════════
# GrabPay processes ~60M transactions/month. Shuffled k-fold reported
# 93% fraud recall but TRUE deployed recall was 71% — the splits leaked
# future fraud signatures into training. TimeSeriesSplit + GroupKFold
# (by merchant_id) match the real deployment geometry and save ~S$4.8M
# per year in avoided fraud + S$540K in avoided emergency refreshes.

print_header("GrabPay Fraud Scoring — Walk-Forward Matters")
print(
    """
CV strategy                 | Reported recall | True deployed recall | S$ loss
----------------------------|-----------------|----------------------|---------
Shuffled k-fold             |       93%       |         71%          |  S$21.6M
TimeSeriesSplit + GroupKFold|       78%       |         77%          |  S$16.8M
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

  [x] Nested CV: outer for honest eval, inner for α selection
  [x] TimeSeriesSplit: walk-forward validation, no future leakage
  [x] GroupKFold: grouped observations stay together
  [x] Picking a CV strategy based on DEPLOYMENT, not data shape

  NEXT: 05_learning_curves.py — diagnose "more data vs better model".
"""
)
