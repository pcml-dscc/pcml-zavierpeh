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
#   - Pick an α that matches a belief about "how big should coefficients be"
#   - Use Ridge to stabilise MAS capital calculations at a Singapore bank
#
# PREREQUISITES:
#   - 01_bias_variance.py (understand why we WANT to constrain complexity)
#   - MLFP02 Bayesian thinking (priors, likelihoods, posteriors)
#
# ESTIMATED TIME: ~40 minutes
#
# TASKS (5-phase R10):
#   1. Theory — why L2 shrinkage helps (geometry + Bayes)
#   2. Build — Ridge pipelines across the α sweep
#   3. Train — fit each α on Singapore credit data, collect metrics
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
# Ridge modifies the ordinary-least-squares objective by adding an L2
# penalty on the coefficient vector:
#
#     min_β  ||y - Xβ||²  +  α · ||β||²
#              ───────         ───────
#              fit the data    penalise large weights
#
# GEOMETRY: The α-term defines a BALL around the origin. As α grows the
# ball shrinks. The optimiser finds the point in that ball where the
# least-squares error is smallest. Because the ball is smooth (no
# corners), the solution's coefficients shrink uniformly — they get
# smaller together but rarely land exactly at zero.
#
# CLOSED FORM: β = (X'X + αI)⁻¹X'y. The αI makes the matrix always
# invertible — Ridge is the standard fix for multicollinearity.
#
# BAYESIAN VIEW: If you assume the likelihood is Gaussian AND place a
# Gaussian prior N(0, τ²I) on β, the MAP (maximum a posteriori) estimate
# is EXACTLY Ridge regression with α = σ²/τ². So:
#   - Large α  ⇔  narrow prior  ⇔  strong belief "coefficients are small"
#   - Small α  ⇔  wide prior    ⇔  weak belief, OLS in the limit
#
# You're not "regularising" — you're encoding a belief about the world.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD Ridge pipelines across an α sweep
# ════════════════════════════════════════════════════════════════════════
# We use the Singapore credit scoring dataset from MLFP02. The target
# is credit_utilization (ratio of balance-to-limit). With ~45 features
# and modest sample size, OLS will overfit and Ridge should help.

print_header("Ridge Regression on Singapore Credit Data")

X_train, y_train, X_test, y_test, feature_names = load_credit_data()
print(
    f"Train: {X_train.shape[0]} rows  "
    f"Test: {X_test.shape[0]} rows  "
    f"Features: {len(feature_names)}"
)

# Baseline OLS for comparison (α → 0)
ols = LinearRegression().fit(X_train, y_train)
ols_norm = float(np.linalg.norm(ols.coef_))
ols_test_mse = mean_squared_error(y_test, ols.predict(X_test))
print(f"\nOLS baseline: ||β||₂ = {ols_norm:.4f}, test MSE = {ols_test_mse:.4f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN Ridge across the α sweep
# ════════════════════════════════════════════════════════════════════════
# For each α we record: train MSE (how hard the model is pulled toward
# OLS), test MSE (how well it generalises), coefficient norm (how much
# shrinkage), and count of near-zero coefficients (Ridge should have
# very few exact zeros, unlike Lasso).

ridge_results: dict[float, dict[str, float]] = {}
for alpha in ALPHAS:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, ridge.predict(X_train))
    test_mse = mean_squared_error(y_test, ridge.predict(X_test))
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
# INTERPRETATION: ||β||₂ drops smoothly as α rises. At α=0.001 Ridge
# barely differs from OLS. At α=1000 the coefficients are all squished
# near zero (the model is nearly a constant prediction).


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the Bayesian interpretation
# ════════════════════════════════════════════════════════════════════════
# We compute α implied by a unit-variance prior (τ²=1) and the empirical
# σ² from the OLS residuals, then compare that Ridge fit against the
# sweep-optimal α. These should be in the same order of magnitude.

print_header("Bayesian Interpretation: Ridge as MAP with Gaussian Prior")

# σ² is the residual variance from the unregularised OLS fit
sigma_sq = float(mean_squared_error(y_train, ols.predict(X_train)))
tau_sq = 1.0  # Unit prior variance — our belief about coefficient spread
alpha_bayes = sigma_sq / tau_sq

ridge_bayes = Ridge(alpha=alpha_bayes).fit(X_train, y_train)
bayes_mse = mean_squared_error(y_test, ridge_bayes.predict(X_test))

print(
    f"""
Prior beliefs:
  σ² (noise variance)     = {sigma_sq:.4f}   (estimated from OLS residuals)
  τ² (prior variance)     = {tau_sq:.4f}    (unit prior — moderate belief)

Implied regularisation:
  α = σ² / τ²            = {alpha_bayes:.4f}

Ridge with Bayesian α:
  Test MSE              = {bayes_mse:.4f}

Compare to sweep-optimal:
  α*                    = {best_alpha}
  Test MSE at α*        = {best_row["test_mse"]:.4f}
"""
)

# Show coefficient shrinkage at several feature names
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
# INTERPRETATION: The Bayesian view turns α from a "tuning knob" into a
# statement of belief. If your domain expert says "I expect effects to
# spread across many features with moderate size," Ridge encodes that
# directly. Disagree? Move τ and α changes accordingly.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: OCBC SME Lending Scorecard Stabilisation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: OCBC's SME lending team in Singapore runs a probability-of-
# default model across ~45,000 active business lines. Features include
# 12-month revenue volatility, director age, sector codes, GIRO
# utilisation, and 30+ bureau features that are strongly correlated
# (e.g. multiple "debt service coverage" variants).
#
# WHY RIDGE IS THE RIGHT TOOL:
#   - OLS on correlated features produces unstable coefficients — a
#     small change in training data swings signs and magnitudes. MAS
#     audits flag this as "lack of model robustness".
#   - Ridge's closed form (X'X + αI)⁻¹ inverts cleanly even when features
#     are near-collinear. The resulting scorecard is STABLE — quarterly
#     refits produce near-identical coefficients, which MAS model-risk
#     examiners specifically look for.
#   - Bayesian framing matches how credit committees THINK: "we believe
#     every bureau feature matters a little, none should dominate" ⇔
#     Gaussian prior with moderate variance ⇔ Ridge at moderate α.
#
# BUSINESS IMPACT (2026 OCBC SME book, S$32B outstanding):
#   - Scorecard refit stability: Ridge refits move coefficients by < 3%
#     quarter-on-quarter vs 18% for unregularised OLS. Each unstable
#     refit triggers a model revalidation costing ~S$120K and a 6-week
#     freeze on new approvals. Annual savings from stable refits alone:
#     ~S$480K in model-ops cost plus ~S$2.8M in avoided origination
#     holds.
#   - Capital efficiency: MAS Internal Ratings-Based (IRB) approval
#     requires demonstrating out-of-time stability. Ridge-based scorecards
#     earn a 15 bp reduction in the PD floor, freeing ~S$48M of
#     regulatory capital that can be redeployed into loan originations.
#
# NOTE: When features are HIGHLY correlated, Ridge keeps ALL of them
# with small weights instead of picking one arbitrarily. This is a
# feature, not a bug — for auditability, banks want every bureau
# feature visible in the coefficient report.

print_header("OCBC SME Lending — Ridge for Refit Stability")
print(
    """
Quarter-on-quarter coefficient drift (illustrative):

  Model          |  Mean |Δβ| / |β|  |  MAS status
  ---------------|---------------------|-----------------------
  OLS            |        18.2%        |  flagged ("unstable")
  Ridge (α=1)    |         2.9%        |  approved
  Ridge (α=10)   |         1.4%        |  approved

Stable refits = fewer revalidations = faster SME approvals =
more revenue on the same credit book.
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
  [x] Why MAS-regulated Singapore banks love Ridge for stability

  KEY INSIGHT: Ridge is the default when you believe "many features
  contribute small amounts." If instead you believe "only a handful
  matter and the rest should be zero", Lasso is the right tool — and
  that's the next file.

  NEXT: 03_lasso_elasticnet.py — L1 sparsity, corner-of-the-diamond
  geometry, and ElasticNet as the pragmatic compromise.
"""
)
