# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2.2: Ridge Regression (L2) and Its Bayesian Twin
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Fit Ridge regression at many α values and read the shrinkage effect
#   - Explain L2 geometry: sphere constraint, uniform shrinkage, never zero
#   - Connect Ridge to a Gaussian prior on the coefficient vector (MAP)
#   - Use Ridge to stabilise MAS capital calculations at a Singapore bank
#
# PREREQUISITES:
#   - 01_bias_variance.py
#   - MLFP02 Bayesian thinking
#
# ESTIMATED TIME: ~40 minutes
#
# TASKS (5-phase R10):
#   1. Theory — why L2 shrinkage helps (geometry + Bayes)
#   2. Build — Ridge pipelines across the α sweep
#   3. Train — fit each α, collect metrics
#   4. Visualise — coefficient-norm trajectory vs α
#   5. Apply — OCBC SME lending scorecard stabilisation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

from shared.mlfp03.ex_2 import ALPHAS, load_credit_data, print_header

# ════════════════════════════════════════════════════════════════════════
# THEORY — L2 Regularisation
# ════════════════════════════════════════════════════════════════════════
#     min_β  ||y - Xβ||²  +  α · ||β||²
#
# L2 defines a SPHERE constraint. All coefficients shrink smoothly toward
# zero; none become exactly zero. Bayesian view: Ridge = MAP estimate
# with a Gaussian prior N(0, τ²I) on the coefficients, where α = σ²/τ².


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD Ridge pipelines
# ════════════════════════════════════════════════════════════════════════

print_header("Ridge Regression on Singapore Credit Data")

# TODO: Load the Singapore credit training/test split. load_credit_data()
# returns X_train, y_train, X_test, y_test, feature_names.
X_train, y_train, X_test, y_test, feature_names = ____
print(
    f"Train: {X_train.shape[0]} rows  "
    f"Test: {X_test.shape[0]} rows  "
    f"Features: {len(feature_names)}"
)

# TODO: Fit an unregularised OLS baseline for comparison.
ols = ____
ols_norm = float(np.linalg.norm(ols.coef_))
ols_test_mse = mean_squared_error(y_test, ols.predict(X_test))
print(f"\nOLS baseline: ||β||₂ = {ols_norm:.4f}, test MSE = {ols_test_mse:.4f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN Ridge across the α sweep
# ════════════════════════════════════════════════════════════════════════

ridge_results: dict[float, dict[str, float]] = {}
for alpha in ALPHAS:
    # TODO: Instantiate Ridge(alpha=alpha), fit on the training set,
    # and record train_mse, test_mse, coef_norm (L2), and number of
    # near-zero coefficients (|β|<1e-6).
    ridge = ____
    ridge.fit(____, ____)
    train_mse = ____
    test_mse = ____
    coef_norm = float(np.linalg.norm(ridge.coef_))
    n_zero = int(np.sum(np.abs(ridge.coef_) < 1e-6))
    ridge_results[alpha] = {
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "coef_norm": coef_norm,
        "n_zero": n_zero,
    }

print(
    f"\n{'alpha':>10} {'train MSE':>12} {'test MSE':>12} "
    f"{'||β||₂':>10} {'zeros':>8}"
)
print("-" * 56)
for alpha, r in ridge_results.items():
    print(
        f"{alpha:>10.3f} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} "
        f"{r['coef_norm']:>10.4f} {r['n_zero']:>8}"
    )

best_alpha, best_row = min(ridge_results.items(), key=lambda x: x[1]["test_mse"])
print(f"\nBest Ridge α = {best_alpha}  (test MSE = {best_row['test_mse']:.4f})")


# ── Checkpoint 1 ───────────────────────────────────────────────────────
assert (
    ridge_results[1000.0]["coef_norm"] < ridge_results[0.001]["coef_norm"]
), "Higher α must produce a smaller ||β||₂"
assert (
    ridge_results[1.0]["n_zero"] <= 2
), "Ridge should leave at most a handful of exact zeros (L2 ≠ Lasso)"
print("\n[ok] Checkpoint 1 passed — Ridge shrinkage behaviour confirmed")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the Bayesian interpretation
# ════════════════════════════════════════════════════════════════════════
# Compute α implied by a unit-variance prior (τ²=1) and the empirical
# σ² from the OLS residuals, then fit Ridge at that α.

print_header("Bayesian Interpretation: Ridge as MAP with Gaussian Prior")

# TODO: Estimate σ² from the training-set residuals of the OLS fit
# above. sigma_sq = mean_squared_error(y_train, ols.predict(X_train)).
sigma_sq = ____
tau_sq = 1.0
# TODO: Derive alpha_bayes = σ² / τ² and fit Ridge at that α.
alpha_bayes = ____
ridge_bayes = Ridge(alpha=alpha_bayes).fit(X_train, y_train)
bayes_mse = mean_squared_error(y_test, ridge_bayes.predict(X_test))

print(
    f"""
Prior beliefs:
  σ² (noise variance)     = {sigma_sq:.4f}
  τ² (prior variance)     = {tau_sq:.4f}

Implied regularisation:
  α = σ² / τ²            = {alpha_bayes:.4f}

Ridge with Bayesian α:
  Test MSE              = {bayes_mse:.4f}

Compare to sweep-optimal:
  α*                    = {best_alpha}
  Test MSE at α*        = {best_row["test_mse"]:.4f}
"""
)

# Show coefficient shrinkage at a few features
print(f"\n{'Feature':<30} {'OLS':>10} {'Ridge(α*)':>12} {'Ridge(1000)':>13}")
print("-" * 66)
ridge_best = Ridge(alpha=best_alpha).fit(X_train, y_train)
ridge_over = Ridge(alpha=1000.0).fit(X_train, y_train)
for i, name in enumerate(feature_names[:8]):
    print(
        f"{name:<30} {ols.coef_[i]:>10.4f} {ridge_best.coef_[i]:>12.4f} "
        f"{ridge_over.coef_[i]:>13.4f}"
    )


# ── Checkpoint 2 ───────────────────────────────────────────────────────
assert alpha_bayes > 0, "Implied α should be positive"
ridge_best_norm = float(np.linalg.norm(ridge_best.coef_))
# Ridge shrinks the coefficient L2 norm vs OLS for any alpha > 0. When the
# CV-optimal alpha is tiny (well-conditioned data), Ridge ≈ OLS and the two
# norms match up to solver float-noise (Ridge uses cholesky, OLS uses lstsq),
# so compare with a small relative tolerance.
assert ridge_best_norm <= ols_norm * (
    1 + 1e-3
), "OLS coefficients should have at least as large a norm as Ridge"
print("\n[ok] Checkpoint 2 passed — Bayesian Ridge interpretation verified")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: OCBC SME Lending Scorecard Stabilisation
# ════════════════════════════════════════════════════════════════════════
# Ridge's closed form (X'X + αI)⁻¹ is stable even when features are
# near-collinear. OCBC's ~S$32B SME book benefits: Ridge scorecard
# refits move coefficients by <3% quarter-on-quarter vs 18% for OLS.
# Stable refits = ~S$480K/year in model-ops + ~S$2.8M in avoided
# approval holds + ~S$48M in freed regulatory capital under IRB.

print_header("OCBC SME Lending — Ridge for Refit Stability")
print(
    """
Quarter-on-quarter coefficient drift (illustrative):

  Model          |  Mean |Δβ| / |β|  |  MAS status
  ---------------|---------------------|-----------------------
  OLS            |        18.2%        |  flagged ("unstable")
  Ridge (α=1)    |         2.9%        |  approved
  Ridge (α=10)   |         1.4%        |  approved
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

  [x] Ridge objective: ||y-Xβ||² + α·||β||²
  [x] Closed-form stability: (X'X + αI)⁻¹ always invertible
  [x] Uniform shrinkage, almost never exact zeros
  [x] MAP equivalence: Ridge ⇔ Gaussian prior with α = σ²/τ²

  NEXT: 03_lasso_elasticnet.py — L1 sparsity, corner-of-the-diamond
  geometry, and ElasticNet as the pragmatic compromise.
"""
)
