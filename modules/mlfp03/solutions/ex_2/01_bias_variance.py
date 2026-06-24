# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2.1: Bias-Variance Trade-off
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Diagnose underfitting vs overfitting from train/test error gaps
#   - Decompose expected test error into Bias², Variance, and irreducible
#     noise via bootstrap resampling
#   - Read a bias-variance curve and pick the "sweet spot" complexity
#   - Connect the bias-variance picture to Singapore credit-risk decisions
#
# PREREQUISITES:
#   - MLFP03 Exercise 1 (feature engineering, sklearn basics)
#   - MLFP02 Module 2 (linear regression, expectation & variance)
#
# ESTIMATED TIME: ~35 minutes
#
# TASKS (5-phase R10):
#   1. Theory — why "more complex" is not always "better"
#   2. Build — polynomial pipelines at increasing degrees
#   3. Train — fit each degree, collect train/test MSE
#   4. Visualise — bias² / variance / noise curve via bootstrap
#   5. Apply — DBS Singapore consumer credit scoring risk
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from shared.mlfp03.ex_2 import (
    SEED,
    make_poly_pipeline,
    make_sine_dataset,
    print_header,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — The Bias-Variance Decomposition
# ════════════════════════════════════════════════════════════════════════
# Every supervised prediction error decomposes into three additive terms:
#
#   E[(y - ŷ)²]  =  Bias²(ŷ)  +  Var(ŷ)  +  σ²
#                   ─────────    ──────     ───
#                   How wrong    How much   Irreducible
#                   the average  the model  noise in y
#                   prediction   wiggles    (we can't
#                   is           between    do better
#                                datasets   than this)
#
# INTUITION:
#   - A model that's too simple (degree 1 line fitting a sine wave) has
#     HIGH BIAS: even the average of many fits is wrong. More data won't
#     fix this — the model class can't express the truth.
#   - A model that's too complex (degree 20 polynomial on 100 points)
#     has HIGH VARIANCE: each random sample produces a wildly different
#     curve. The model memorises noise.
#   - The "sweet spot" balances the two. Cross-validation FINDS this
#     automatically without knowing the true function.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the polynomial experiment
# ════════════════════════════════════════════════════════════════════════
# We use a 1D synthetic sine problem because (a) we KNOW the truth, which
# lets us measure bias directly, and (b) the effect is visually obvious.

print_header("Polynomial Degree Experiment on sin(2πx)")

x_train, y_train, x_test, y_test, noise_variance = make_sine_dataset(
    n=100, noise_sigma=0.2, seed=SEED
)
print(
    f"Train: {x_train.shape[0]} pts  "
    f"Test: {x_test.shape[0]} pts  "
    f"σ² = {noise_variance:.4f}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN polynomial models at many degrees
# ════════════════════════════════════════════════════════════════════════
# For each degree we fit on the training set and record MSE on both the
# training set (how well the model memorised) and the held-out test set
# (how well it generalises). The GAP between the two is the overfit
# penalty — the pricetag the model pays for fitting the training noise.

degree_rows: dict[int, dict[str, float]] = {}
print(f"\n{'Degree':>6} {'Train MSE':>12} {'Test MSE':>12} {'Gap':>10}  Diagnosis")
print("-" * 60)
for degree in [1, 2, 4, 6, 9, 12, 15, 20]:
    model = make_poly_pipeline(degree)
    model.fit(x_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(x_train))
    test_mse = mean_squared_error(y_test, model.predict(x_test))
    gap = test_mse - train_mse

    if degree <= 2:
        diagnosis = "underfit (bias)"
    elif degree <= 6:
        diagnosis = "good fit"
    else:
        diagnosis = "overfit (variance)"

    degree_rows[degree] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "gap": gap,
    }
    print(f"{degree:>6} {train_mse:>12.4f} {test_mse:>12.4f} {gap:>10.4f}  {diagnosis}")


# ── Checkpoint 1 ───────────────────────────────────────────────────────
assert (
    degree_rows[1]["test_mse"] > degree_rows[4]["test_mse"]
), "Degree=1 should have higher test error than degree=4 (underfit)"
assert (
    degree_rows[20]["train_mse"] < degree_rows[2]["train_mse"]
), "Degree=20 should memorise training data (lowest train MSE)"
print("\n[ok] Checkpoint 1 passed — underfit/overfit pattern confirmed")
# INTERPRETATION: Watch the train MSE keep falling as degree increases —
# a higher-degree polynomial can ALWAYS fit training data better. But
# test MSE bottoms out around degree 4-6 then blows up. That blow-up is
# variance: the model fits the noise.


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the bias-variance decomposition
# ════════════════════════════════════════════════════════════════════════
# We estimate Bias² and Variance EMPIRICALLY by bootstrapping:
#   1. Draw a random sample of the training set (with replacement)
#   2. Fit the polynomial
#   3. Predict on the fixed test grid
#   4. Repeat 50 times
#
# Bias²    = mean squared error of the AVERAGE prediction vs truth
# Variance = spread of predictions across the bootstrap replicates
# Noise    = known σ² (0.04 in our synthetic setup)

rng = np.random.default_rng(SEED)


def bias_variance_decomposition(degree: int, n_bootstrap: int = 50) -> dict[str, float]:
    """Estimate Bias², Variance, and total expected error for a polynomial."""
    all_preds = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_train), len(y_train), replace=True)
        model = make_poly_pipeline(degree)
        model.fit(x_train[idx], y_train[idx])
        all_preds.append(model.predict(x_test))

    preds = np.array(all_preds)  # (n_bootstrap, n_test)
    mean_pred = preds.mean(axis=0)

    # Noiseless "truth" at the test points (we KNOW the generating fn)
    y_truth = np.sin(2 * np.pi * x_test.ravel())

    bias_sq = float(np.mean((mean_pred - y_truth) ** 2))
    variance = float(np.mean(preds.var(axis=0)))
    expected = bias_sq + variance + noise_variance

    return {
        "bias_sq": bias_sq,
        "variance": variance,
        "noise": noise_variance,
        "expected_error": expected,
    }


print_header("Bias-Variance Decomposition via Bootstrap")
print(
    f"{'Degree':>6} {'Bias²':>10} {'Variance':>10} {'Noise':>8} "
    f"{'Expected':>12}  Dominant"
)
print("-" * 60)

bv_rows: dict[int, dict[str, float]] = {}
for degree in [1, 2, 3, 6, 10, 15]:
    bv = bias_variance_decomposition(degree, n_bootstrap=40)
    bv_rows[degree] = bv
    dominant = "Bias" if bv["bias_sq"] > bv["variance"] else "Variance"
    print(
        f"{degree:>6} {bv['bias_sq']:>10.4f} {bv['variance']:>10.4f} "
        f"{bv['noise']:>8.4f} {bv['expected_error']:>12.4f}  {dominant}"
    )


# ── Checkpoint 2 ───────────────────────────────────────────────────────
assert (
    bv_rows[1]["bias_sq"] > bv_rows[10]["bias_sq"]
), "Degree=1 should have higher bias² than degree=10"
assert (
    bv_rows[1]["variance"] < bv_rows[15]["variance"]
), "Degree=1 should have lower variance than degree=15"
print("\n[ok] Checkpoint 2 passed — bias-variance decomposition valid")
# INTERPRETATION: At degree=1, Bias² dominates — the model is too rigid
# to curve with the sine wave. At degree=15, Variance dominates — every
# bootstrap sample produces a wildly different polynomial. The sweet
# spot (degree ~3-6) balances the two, which is where expected_error is
# minimised. No amount of data can push expected error below σ² (noise).


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Singapore Consumer Credit Scoring
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS Bank (Singapore) scores ~2M active retail credit-card
# customers every night to decide whether to extend additional credit.
# The risk team has a catalogue of 180 candidate features (income,
# transaction velocity, bureau data, device fingerprints, etc.).
#
# WHY BIAS-VARIANCE MATTERS HERE:
#   - A 5-feature linear score card is INTERPRETABLE but high-bias.
#     Default rate predictions systematically miss mid-risk customers,
#     so DBS either extends credit to future defaulters OR denies good
#     customers. Industry baseline: ~2.4% unnecessary defaults.
#   - A 180-feature gradient-boosted tree with no regularisation is
#     high-variance. On a new batch (e.g. post-pandemic customers) its
#     predictions swing wildly because it memorised correlations in the
#     training window. MAS stress tests show ~40% of these models fail
#     out-of-time validation.
#
# BUSINESS IMPACT (2026 DBS retail credit-card book, S$18B outstanding):
#   - Every 0.10 percentage point reduction in default rate that a
#     well-regularised model achieves vs a high-variance model is worth
#     ~S$18M/year in avoided write-offs.
#   - MAS Notice 637 capital treatment rewards models that generalise:
#     a model with stable out-of-time performance reduces required
#     capital by ~3% of the credit book ≈ S$540M freed up.
#
# CONNECTION: The "sweet spot" in the bias-variance curve is exactly
# what the MAS model-risk team looks for in vetting a scorecard. Too
# simple = systematic underpricing of risk. Too complex = instability
# on new customers. Regularised mid-complexity wins every audit.

print_header("DBS Retail Credit Scoring — Bias/Variance in Context")
print(
    """
Stakeholder         | Concern                      | BV trade-off
--------------------|------------------------------|-----------------
Credit risk team    | Missed defaults              | Too much bias
Model-risk / MAS    | Out-of-time instability      | Too much variance
Branch / CX         | Denied good customers        | Too much bias
Finance / capital   | Buffer volatility            | Too much variance

Takeaway: the bias-variance curve is not an academic exercise — every
stakeholder sits at a different point on it. The "right" degree is the
one that minimises EXPECTED loss for the business, not training loss.
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

  [x] Train/test error gap as an overfit diagnostic
  [x] The E[(y-ŷ)²] = Bias² + Variance + σ² decomposition
  [x] Empirical bias/variance estimation via bootstrap
  [x] Reading the "dominant" term to diagnose a model
  [x] Applying the trade-off to a real Singapore credit decision

  KEY INSIGHT: "Complexity" is not a single number — it's the knob you
  turn to trade bias against variance. Regularisation (next file) is
  another way to turn that same knob without changing the model class.

  NEXT: 02_ridge_regression.py — shrink coefficients toward zero with
  L2 and meet its Bayesian alter-ego (Gaussian prior on weights).
"""
)
