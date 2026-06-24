# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 5.2: Regression Diagnostics
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Detect multicollinearity using Variance Inflation Factor (VIF)
#   - Perform residual diagnostics: normality, heteroscedasticity, patterns
#   - Run the Breusch-Pagan test for heteroscedasticity
#   - Interpret diagnostic plots: residuals vs fitted, Q-Q, histogram
#
# PREREQUISITES: Exercise 5.1 (OLS from scratch, coefficients, R-squared)
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Load data and fit baseline OLS
#   2. Compute VIF for multicollinearity detection
#   3. Residual diagnostics: normality, skewness, kurtosis
#   4. Breusch-Pagan test for heteroscedasticity
#   5. Diagnostic visualisations (four-panel plot)
#
# THEORY:
#   OLS assumes: (1) linearity, (2) no perfect multicollinearity,
#   (3) homoscedastic errors, (4) normally distributed errors.
#   When assumptions break, estimates are still unbiased but SEs
#   are wrong — making t-tests and CIs unreliable.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from shared.mlfp02.ex_5 import (
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    load_hdb_clean,
    build_design_matrix,
    fit_ols,
    compute_vif,
    breusch_pagan,
    save_residual_diagnostics,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Data and Fit Baseline OLS
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MLFP02 Exercise 5.2: Regression Diagnostics")
print("=" * 70)

hdb_clean = load_hdb_clean()
X, y, feature_names = build_design_matrix(hdb_clean)
n_obs, k = X.shape
X_raw = X[:, 1:]  # Without intercept column

fit = fit_ols(X, y)
residuals = fit["residuals"]
y_hat = fit["y_hat"]

print(f"\n  Baseline OLS: R-squared={fit['R2']:.6f}, n={n_obs:,}, k={k}")
print(f"  Residual sigma_hat = ${fit['sigma_hat']:,.0f}")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Diagnostics Matter
# ════════════════════════════════════════════════════════════════════════
# OLS gives you numbers regardless of whether its assumptions hold.
# The coefficients might be unbiased even if assumptions fail, but the
# standard errors — and therefore every t-test, p-value, and confidence
# interval — can be wrong.
#
# Analogy: Imagine measuring the height of buildings with a ruler that
# stretches when it gets warm. Your measurements (coefficients) might
# still be centred on the truth, but your error bars (SEs) are wrong.
# You think you know a building is 50.0 +/- 0.1 metres, but really
# it is 50.0 +/- 2.0 metres. Diagnostics check the ruler.
#
# WHY THIS MATTERS: A bank using an HDB valuation model with wrong
# SEs might approve a mortgage thinking the property is worth
# $500K +/- $10K, when the true uncertainty is $500K +/- $80K. That
# is the difference between a safe loan and a potential write-off.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Multicollinearity: Variance Inflation Factor (VIF)
# ════════════════════════════════════════════════════════════════════════
# VIF_j = 1/(1 - R-squared_j) where R-squared_j is from regressing
# feature j on all other features.
# VIF > 5 = moderate concern, VIF > 10 = serious.

print(f"\n=== Multicollinearity: VIF ===")

vif_results = compute_vif(X_raw, NUMERIC_FEATURES)

print(f"{'Feature':<25} {'VIF':>8} {'Status':>12}")
print("-" * 48)
for feat, vif in vif_results.items():
    status = "OK" if vif < 5 else "MODERATE" if vif < 10 else "HIGH"
    print(f"{feat:<25} {vif:>8.2f} {status:>12}")

# Correlation matrix
print(f"\nCorrelation matrix:")
corr_matrix = np.corrcoef(X_raw.T)
for i, fi in enumerate(NUMERIC_FEATURES):
    for j, fj in enumerate(NUMERIC_FEATURES):
        if j > i:
            print(f"  corr({fi}, {fj}) = {corr_matrix[i, j]:.3f}")

# INTERPRETATION: VIF > 10 means the feature is almost entirely
# predictable from other features — its coefficient estimate is
# unstable and its SE is inflated. Drop one of the collinear features
# or use regularisation (Ridge regression).

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert all(v >= 1.0 for v in vif_results.values()), "VIF must be >= 1"
print("\n--- Checkpoint 2 passed --- VIF computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Residual Diagnostics: Normality and Shape
# ════════════════════════════════════════════════════════════════════════
# Good residuals should be: normally distributed, homoscedastic,
# uncorrelated, and show no patterns.

print(f"\n=== Residual Diagnostics ===")

# 1. Normality of residuals (Shapiro-Wilk on subsample)
residual_sample = np.random.default_rng(42).choice(
    residuals, size=min(5000, len(residuals)), replace=False
)
sw_stat, sw_p = stats.shapiro(residual_sample)
print(f"\n1. Normality (Shapiro-Wilk on subsample):")
print(f"   W={sw_stat:.4f}, p={sw_p:.6f}")
print(
    f"   {'Normal residuals' if sw_p > 0.05 else 'Non-normal residuals — consider robust SE'}"
)

# 2. Skewness and kurtosis
res_skew = stats.skew(residuals)
res_kurt = stats.kurtosis(residuals)
print(f"\n2. Shape:")
print(f"   Skewness: {res_skew:.3f} (0 = symmetric)")
print(f"   Excess kurtosis: {res_kurt:.3f} (0 = Normal tails)")

# 3. Residual summary
print(f"\n3. Residual summary:")
print(f"   Mean: ${residuals.mean():.2f} (should be approx 0)")
print(f"   Std:  ${residuals.std():,.0f}")
print(f"   Min:  ${residuals.min():,.0f}")
print(f"   Max:  ${residuals.max():,.0f}")
print(
    f"   |Residual| > 2*sigma: {np.sum(np.abs(residuals) > 2 * residuals.std()):,} "
    f"({np.mean(np.abs(residuals) > 2 * residuals.std()):.1%})"
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert abs(residuals.mean()) < 1.0, "Residual mean should be approximately zero"
print("\n--- Checkpoint 3 passed --- residual shape assessed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Breusch-Pagan Test for Heteroscedasticity
# ════════════════════════════════════════════════════════════════════════
# Regress squared residuals on X to test if variance depends on predictors.
# H0: variance is constant (homoscedastic)
# H1: variance depends on one or more predictors

print(f"\n=== Heteroscedasticity: Breusch-Pagan ===")

bp_stat, bp_p = breusch_pagan(residuals, X_raw)
print(f"BP statistic: {bp_stat:.2f}, p={bp_p:.6f}")
print(
    f"{'Homoscedastic' if bp_p > 0.05 else 'HETEROSCEDASTIC — variance depends on predictors'}"
)

# INTERPRETATION: If residuals are heteroscedastic, OLS estimates are
# still unbiased but the standard errors are wrong — making t-tests
# and confidence intervals unreliable. Remedies: WLS (next file),
# heteroscedasticity-consistent (HC) standard errors, or log transform.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert bp_stat >= 0, "BP statistic must be non-negative"
print("\n--- Checkpoint 4 passed --- heteroscedasticity tested\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Four-Panel Diagnostic Plot
# ════════════════════════════════════════════════════════════════════════

path = save_residual_diagnostics(
    y_hat=y_hat,
    residuals=residuals,
    feature_col=X_raw[:, 0],
    feature_label="Floor Area (sqm)",
    filename="02_residual_diagnostics.html",
)
print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Mortgage Risk Assessment
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A risk analyst at a Singapore bank reviews the HDB
# valuation model before it is used to approve mortgages. The analyst
# runs these diagnostics and finds:
#
# - VIF is low for all features — no multicollinearity problem.
# - Breusch-Pagan rejects homoscedasticity — expensive flats have
#   higher price variance than cheap ones.
# - Residuals are right-skewed — the model underestimates some
#   expensive flats.
#
# BUSINESS IMPACT: The bank uses the model's confidence intervals to
# set loan-to-value (LTV) ratios. If the SEs are wrong because of
# heteroscedasticity, the bank might approve a 90% LTV loan on a
# property whose true uncertainty is +/- $80K. A $400K flat could
# really be worth $320K, and the bank is exposed. The fix: use WLS
# or robust standard errors (next file).

print(f"\n--- Business Application: Mortgage Risk ---")
if bp_p < 0.05:
    print(f"  WARNING: Heteroscedasticity detected (BP p={bp_p:.4f})")
    print(f"  Standard errors from OLS are unreliable for this data")
    print(f"  Confidence intervals for property valuations may be too narrow")
    print(f"  Recommendation: use WLS (03_weighted_ls.py) for reliable SEs")
else:
    print(f"  Homoscedasticity holds — OLS standard errors are reliable")
    print(f"  Confidence intervals can be trusted for mortgage decisions")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED (5.2)")
print("=" * 70)
print(
    """
  - VIF for multicollinearity detection (VIF > 10 = unstable coefficients)
  - Shapiro-Wilk test for residual normality
  - Skewness and kurtosis as shape diagnostics
  - Breusch-Pagan test for heteroscedasticity
  - Four-panel diagnostic visualisation
  - Business reasoning: why wrong SEs lead to wrong mortgage decisions

  NEXT: In 03_weighted_ls.py you'll fix the heteroscedasticity problem
  using Weighted Least Squares — giving each observation a weight
  inversely proportional to its variance.
"""
)

print("--- Exercise 5.2 complete --- Regression Diagnostics")
