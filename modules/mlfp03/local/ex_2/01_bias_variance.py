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
#   E[(y - ŷ)²]  =  Bias²(ŷ)  +  Var(ŷ)  +  σ²
#
# HIGH BIAS = "even on average we get the wrong answer" (underfit).
# HIGH VARIANCE = "our answer depends wildly on the training sample"
# (overfit). The irreducible σ² is the floor that no model can beat.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the polynomial experiment
# ════════════════════════════════════════════════════════════════════════

print_header("Polynomial Degree Experiment on sin(2πx)")

# TODO: Generate the synthetic 1D dataset. Use make_sine_dataset with
# n=100, noise_sigma=0.2, seed=SEED. It returns a tuple of five values:
# x_train (shape n_train,1), y_train, x_test, y_test, noise_variance.
x_train, y_train, x_test, y_test, noise_variance = ____

print(
    f"Train: {x_train.shape[0]} pts  "
    f"Test: {x_test.shape[0]} pts  "
    f"σ² = {noise_variance:.4f}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN polynomial models at many degrees
# ════════════════════════════════════════════════════════════════════════

degree_rows: dict[int, dict[str, float]] = {}
print(f"\n{'Degree':>6} {'Train MSE':>12} {'Test MSE':>12} {'Gap':>10}  Diagnosis")
print("-" * 60)
for degree in [1, 2, 4, 6, 9, 12, 15, 20]:
    # TODO: Construct the pipeline via make_poly_pipeline(degree), fit
    # it on (x_train, y_train), and compute train/test MSE using
    # sklearn.metrics.mean_squared_error.
    model = ____
    model.fit(____, ____)
    train_mse = ____
    test_mse = ____
    gap = test_mse - train_mse

    # Hint: classify by the `degree` heuristic in the solution comments.
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


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the bias-variance decomposition
# ════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(SEED)


def bias_variance_decomposition(degree: int, n_bootstrap: int = 50) -> dict[str, float]:
    """Estimate Bias², Variance, and total expected error for a polynomial."""
    all_preds = []
    for _ in range(n_bootstrap):
        # TODO: Draw a bootstrap index with rng.choice(len(y_train),
        # len(y_train), replace=True), fit a polynomial pipeline on the
        # resampled data, and append the test-set predictions.
        idx = ____
        model = make_poly_pipeline(degree)
        model.fit(____, ____)
        all_preds.append(model.predict(x_test))

    preds = np.array(all_preds)
    mean_pred = preds.mean(axis=0)

    # Noiseless truth at the test points
    y_truth = np.sin(2 * np.pi * x_test.ravel())

    # TODO: Compute the three components of expected test error.
    # bias_sq  = mean of (mean_pred - y_truth)^2
    # variance = mean of preds.var(axis=0)
    bias_sq = ____
    variance = ____
    expected = bias_sq + variance + noise_variance

    return {
        "bias_sq": float(bias_sq),
        "variance": float(variance),
        "noise": noise_variance,
        "expected_error": float(expected),
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


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Singapore Consumer Credit Scoring
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS Bank (Singapore) scores ~2M active retail credit-card
# customers every night. A 5-feature linear scorecard is interpretable
# but high-bias; a 180-feature unregularised GBT is high-variance and
# fails MAS out-of-time validation.
#
# BUSINESS IMPACT: Every 0.10 pp reduction in default rate from a
# well-regularised model is worth ~S$18M/year on the S$18B retail book.
# MAS Notice 637 capital treatment rewards stable models with ~3% of
# the book (~S$540M) freed regulatory capital.

print_header("DBS Retail Credit Scoring — Bias/Variance in Context")
print(
    """
Stakeholder         | Concern                      | BV trade-off
--------------------|------------------------------|-----------------
Credit risk team    | Missed defaults              | Too much bias
Model-risk / MAS    | Out-of-time instability      | Too much variance
Branch / CX         | Denied good customers        | Too much bias
Finance / capital   | Buffer volatility            | Too much variance
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
  [x] Applying the trade-off to a real Singapore credit decision

  NEXT: 02_ridge_regression.py — shrink coefficients toward zero with
  L2 and meet its Bayesian alter-ego (Gaussian prior on weights).
"""
)
