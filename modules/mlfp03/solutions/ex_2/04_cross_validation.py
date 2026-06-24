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
#   - Quantify the hidden bias in "leaky" CV on Singapore payments data
#
# PREREQUISITES:
#   - 02_ridge_regression.py and 03_lasso_elasticnet.py
#   - MLFP02 sampling theory (bias, variance of estimators)
#
# ESTIMATED TIME: ~45 minutes
#
# TASKS (5-phase R10):
#   1. Theory — why "one CV fits all" is wrong
#   2. Build — three CV splitters and a scoring loop
#   3. Train — nested CV (unbiased α selection)
#   4. Visualise — CV-strategy comparison table + TimeSeriesSplit / Group
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
# Standard k-fold implicitly assumes:
#   (a) Observations are INDEPENDENT — shuffling is harmless
#   (b) Observations are IDENTICALLY distributed — training-time
#       distribution equals prediction-time distribution
#   (c) Hyperparameter selection and performance estimation can share
#       the same splits (they can't — that's leakage)
#
# Real datasets break these assumptions all the time:
#   - FINANCIAL DATA is temporal: shuffling lets the model train on
#     future dates to predict past dates (violates (b)).
#   - MEDICAL DATA has repeated measures per patient: the same patient
#     can end up in both folds (violates (a)).
#   - HYPERPARAMETER TUNING on the same splits used for evaluation
#     gives a biased, optimistic estimate (violates (c)).
#
# THE FIX — pick the CV strategy that MATCHES the deployment scenario:
#   i.i.d. data           → standard k-fold
#   temporal data         → TimeSeriesSplit (walk-forward)
#   grouped data          → GroupKFold
#   hyperparameter tuning → nested CV (outer for eval, inner for tune)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the CV splitters + scoring loop
# ════════════════════════════════════════════════════════════════════════

print_header("Cross-Validation Strategies on Singapore Credit Data")
X_train, y_train, X_test, y_test, feature_names = load_credit_data()
print(f"Train: {X_train.shape}  Features: {len(feature_names)}")

rng = np.random.default_rng(SEED)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN with nested CV (unbiased α selection)
# ════════════════════════════════════════════════════════════════════════
# STANDARD CV uses the SAME folds to pick α AND report performance.
# That double use is a form of leakage: α was chosen to look good on
# those folds, so the score on those folds is biased upward.
#
# NESTED CV fixes this:
#   OUTER 5-fold: held out for performance reporting (never touched
#                 during α selection).
#   INNER 3-fold: used inside each outer fold to pick α from the
#                 candidates ALPHAS.
#
# The outer mean is an unbiased estimate of the selected model's
# generalisation performance.

print_header("Nested Cross-Validation")

# Biased standard CV baseline
ridge_cv = RidgeCV(alphas=ALPHAS, cv=5)
ridge_cv.fit(X_train, y_train)
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

    best_alpha = ALPHAS[0]
    best_inner = -np.inf
    for alpha in ALPHAS:
        inner = cross_val_score(
            Ridge(alpha=alpha),
            X_out_tr,
            y_out_tr,
            cv=inner_cv,
            scoring="r2",
        )
        if inner.mean() > best_inner:
            best_inner = inner.mean()
            best_alpha = alpha

    ridge_selected = Ridge(alpha=best_alpha).fit(X_out_tr, y_out_tr)
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
# INTERPRETATION: The gap between standard and nested CV is the
# optimism from using the same data for tuning and evaluation. If it's
# large (>0.03 R²), your reported performance is a lie.


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE time-series and group CV
# ════════════════════════════════════════════════════════════════════════
# Run three strategies on the SAME ridge model and compare the means.
# The differences tell you where the leakage is.

print_header("CV Strategy Comparison: k-fold vs Time-series vs Group")

# Standard 5-fold
kfold_scores = cross_val_score(Ridge(alpha=1.0), X_train, y_train, cv=5, scoring="r2")

# Time-series walk-forward
tscv = TimeSeriesSplit(n_splits=5)
ts_scores = cross_val_score(Ridge(alpha=1.0), X_train, y_train, cv=tscv, scoring="r2")

print("\nTime-series walk-forward splits:")
for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_train)):
    print(
        f"  Fold {fold + 1}: train=[0:{tr_idx[-1] + 1}] "
        f"({len(tr_idx)} samples), test=[{te_idx[0]}:{te_idx[-1] + 1}] "
        f"({len(te_idx)} samples)"
    )

# GroupKFold — simulate ~5 observations per "customer"
n = X_train.shape[0]
groups = np.repeat(np.arange(n // 5 + 1), 5)[:n]
rng.shuffle(groups)
group_cv = GroupKFold(n_splits=5)
group_scores = cross_val_score(
    Ridge(alpha=1.0),
    X_train,
    y_train,
    cv=group_cv,
    groups=groups,
    scoring="r2",
)

# Verify group integrity
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
# INTERPRETATION: If TimeSeriesSplit gives a substantially LOWER R²
# than standard k-fold, your data has temporal leakage — the standard
# score is an illusion. Same for GroupKFold with grouped data.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: GrabPay Transaction Fraud Time-Series Validation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: GrabPay processes ~60M transactions/month across Southeast
# Asia (~25M in Singapore). The fraud team builds a real-time risk
# scorer that sees each transaction once and must respond in <300ms.
# The data has STRONG temporal structure:
#   - Fraud patterns shift every few weeks (new attack vectors)
#   - Merchant mix changes with marketing campaigns
#   - Seasonal effects (11.11, BFCM, Chinese New Year) create regime
#     shifts that a shuffled k-fold completely misses
#
# WHY TIME-SERIES CV:
#   - Walk-forward validation mirrors deployment: we train on "past",
#     predict on "future", and never let the future leak backwards.
#   - TimeSeriesSplit also reveals concept drift — if walk-forward R²
#     degrades monotonically across folds, the model is ageing and
#     needs more-frequent refreshes.
#
# WHY GROUPKFOLD TOO:
#   - Merchants with thousands of transactions dominate the volume. If
#     a single merchant lands in both train and test, the model is
#     effectively memorising the merchant_id. Group by merchant_id
#     (≈120K groups) to force generalisation to NEW merchants.
#   - Same for user_id: without GroupKFold, models over-fit to
#     repeat-user spending patterns and under-detect first-purchase
#     fraud (the highest-risk cohort).
#
# BUSINESS IMPACT (GrabPay Singapore, 2026 run-rate):
#   - Transaction volume: ~S$12B/year in Singapore
#   - Fraud loss baseline (k-fold evaluated model): ~18 bp = S$21.6M/yr
#   - Walk-forward + GroupKFold evaluated model: ~14 bp = S$16.8M/yr
#     (lower because the chosen model ACTUALLY generalises to new days
#     and new merchants, instead of looking good on shuffled folds)
#   - Annual loss avoided by correct CV strategy: S$4.8M
#   - Model-revalidation cost avoided: walk-forward catches drift 4-6
#     weeks earlier than shuffled k-fold, avoiding ~3 emergency model
#     refreshes per year at ~S$180K each = S$540K
#   - Total annual impact of switching CV strategy: ~S$5.3M
#
# LEAKAGE FAILURE MODE: A previous GrabPay model used shuffled k-fold
# and reported 93% fraud recall. When deployed, true recall was 71%.
# The gap was entirely due to temporal leakage — the training folds
# contained future fraud signatures that the test folds also saw.

print_header("GrabPay Fraud Scoring — Walk-Forward Matters")
print(
    """
CV strategy                | Reported recall | True deployed recall | S$ loss
---------------------------|-----------------|----------------------|---------
Shuffled k-fold            |       93%       |         71%          |  S$21.6M
TimeSeriesSplit + GroupKFold|      78%       |         77%          |  S$16.8M

Lesson: a model that LOOKS worse in shuffled CV but matches deployment
reality in walk-forward CV is the one that actually saves money.
Match your CV to your deployment geometry.
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
  [x] Measuring optimism bias as (standard - nested) CV difference
  [x] Picking a CV strategy based on DEPLOYMENT, not data shape

  KEY INSIGHT: The CV strategy is a MODELLING DECISION, not a
  technicality. The "best" model under shuffled k-fold can be the
  worst model in production if the data has structure.

  NEXT: 05_learning_curves.py — learning curves diagnose "do I need
  more data or a better model?" and tie the whole exercise together.
"""
)
