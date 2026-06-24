# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 5.1: OLS from Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Derive and implement OLS using the normal equation beta = (X'X)^{-1}X'y
#   - Interpret regression coefficients with ceteris paribus reasoning
#   - Test coefficient significance using t-statistics and p-values
#   - Compute R-squared, adjusted R-squared, and the F-statistic
#
# PREREQUISITES: Exercises 2-3 (MLE, hypothesis testing, t-statistics)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Load HDB data and engineer numeric features
#   2. Implement OLS from scratch: beta = (X'X)^{-1}X'y
#   3. Interpret coefficients: direction, magnitude, significance
#   4. Compute t-statistics and p-values for every coefficient
#   5. Compute R-squared, adjusted R-squared, F-statistic
#
# DATASET: HDB resale flat transactions (Singapore)
#   Source: data.gov.sg — public housing resale records, 2020+
#   Target: resale_price (SGD)
#   Features: floor_area_sqm, storey_midpoint, remaining_lease_years
#
# THEORY:
#   OLS minimises sum(yi - y_hat_i)^2. Closed-form: beta = (X'X)^{-1}X'y
#   Each beta_j = expected change in y for one-unit change in x_j,
#   holding all others constant (ceteris paribus).
#   t = beta_j / SE(beta_j), testing H0: beta_j = 0.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from shared.mlfp02.ex_5 import (
    NUMERIC_FEATURES,
    TARGET,
    OUTPUT_DIR,
    load_hdb_clean,
    build_design_matrix,
    fit_ols,
    print_coef_table,
    save_actual_vs_predicted,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load and Engineer Features
# ════════════════════════════════════════════════════════════════════════
# Regression requires numeric features. We parse storey range to a
# midpoint and compute remaining lease. Categorical variables will
# be dummy-encoded in 04_model_enrichment.py.

print("=" * 70)
print("  MLFP02 Exercise 5.1: OLS from Scratch")
print("=" * 70)

hdb_clean = load_hdb_clean()
print(f"\n  Data loaded: {hdb_clean.height:,} rows")
print(f"  Features: {', '.join(NUMERIC_FEATURES)}")
print(f"  Target: {TARGET}")

# Summary statistics
for col in [*NUMERIC_FEATURES, TARGET]:
    vals = hdb_clean[col].to_numpy().astype(np.float64)
    print(
        f"  {col}: mean={vals.mean():.1f}, std={vals.std():.1f}, "
        f"range=[{vals.min():.0f}, {vals.max():.0f}]"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert hdb_clean.height > 10_000, f"Expected >10K rows, got {hdb_clean.height}"
assert "storey_midpoint" in hdb_clean.columns, "storey_midpoint must exist"
assert "remaining_lease_years" in hdb_clean.columns, "remaining_lease must exist"
print("\n--- Checkpoint 1 passed --- features engineered\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Normal Equation
# ════════════════════════════════════════════════════════════════════════
# We want to minimise ||y - X*beta||^2.
# Taking the derivative and setting it to zero:
#   d/d(beta) = -2X'(y - X*beta) = 0
#   X'X*beta = X'y
#   beta = (X'X)^{-1} X'y
#
# Analogy: Imagine fitting a flat sheet of paper through a cloud of
# points. The normal equation finds the exact tilt and position that
# minimises the total vertical distance from every point to the sheet.
#
# WHY THIS MATTERS: In HDB valuation, banks and property agents use
# regression to estimate fair market value. Getting the coefficients
# right — and understanding their limits — is the difference between
# a good estimate and a bad loan.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build Design Matrix and Fit OLS
# ════════════════════════════════════════════════════════════════════════

X, y, feature_names = build_design_matrix(hdb_clean)
n_obs, k = X.shape

print(f"\n=== OLS from Scratch ===")
print(f"Design matrix X: {X.shape} (n={n_obs:,}, k={k})")

# TODO: Compute (X'X) — the Gram matrix
# Hint: use X.T @ X
XtX = ____

# TODO: Compute the inverse of (X'X)
# Hint: use np.linalg.inv()
XtX_inv = ____

# TODO: Compute X'y
Xty = ____

# TODO: Compute beta = (X'X)^{-1} X'y
beta_ols = ____

# Predictions and residuals
y_hat = X @ beta_ols
residuals = y - y_hat

print(f"\nOLS Coefficients:")
print(f"{'Feature':<25} {'Coefficient':>14}")
print("-" * 42)
for name, coef in zip(feature_names, beta_ols):
    print(f"{name:<25} {coef:>14,.2f}")

# Verify with numpy lstsq
beta_lstsq, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
print(
    f"\nVerification (numpy lstsq): max |diff| = "
    f"{np.max(np.abs(beta_ols - beta_lstsq)):.2e}"
)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert np.allclose(
    beta_ols, beta_lstsq, atol=1.0
), "OLS from scratch must match numpy lstsq"
assert len(beta_ols) == k, "Should have k coefficients"
print("\n--- Checkpoint 2 passed --- OLS implemented from scratch\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Coefficient Interpretation
# ════════════════════════════════════════════════════════════════════════
# Ceteris paribus: "all else equal, one more sqm of floor area
# is associated with $X higher resale price."

print(f"\n=== Coefficient Interpretation ===")
for i, name in enumerate(feature_names):
    if i == 0:
        print(f"\nIntercept: ${beta_ols[0]:,.0f}")
        print(f"  A flat with all features at zero would cost ${beta_ols[0]:,.0f}")
        print(f"  (Not meaningful — extrapolation beyond data range)")
    else:
        direction = "increases" if beta_ols[i] > 0 else "decreases"
        unit = "sqm" if "area" in name else "storey" if "storey" in name else "year"
        print(f"\n{name}: ${beta_ols[i]:,.0f} per {unit}")
        print(
            f"  Each additional {unit} {direction} resale price by "
            f"${abs(beta_ols[i]):,.0f}"
        )
        print(f"  Holding all other features constant (ceteris paribus)")

# Practical example
example_flat = {
    "floor_area_sqm": 92.0,
    "storey_midpoint": 8.0,
    "remaining_lease_years": 75.0,
}

# TODO: Compute predicted price for the example flat
# Hint: start with beta_ols[0] (intercept), then add beta_ols[i+1] * val
#       for each feature value
predicted = ____

print(f"\n--- Prediction Example ---")
for feat, val in example_flat.items():
    print(f"  {feat} = {val}")
print(f"  Predicted price: ${predicted:,.0f}")
actual_similar = hdb_clean.filter(
    (pl.col("floor_area_sqm").is_between(90, 94))
    & (pl.col("storey_midpoint").is_between(7, 9))
)
if actual_similar.height > 0:
    print(
        f"  Actual mean for similar flats: "
        f"${actual_similar['resale_price'].mean():,.0f}"
    )

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    beta_ols[1] > 0
), "Floor area coefficient should be positive (bigger = more expensive)"
print("\n--- Checkpoint 3 passed --- coefficients interpreted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — t-Statistics and p-Values
# ════════════════════════════════════════════════════════════════════════
# H0: beta_j = 0 (feature j has no linear relationship with price)
# t = beta_j / SE(beta_j), where SE(beta_j) = sqrt(sigma_hat^2 * (X'X)^{-1}_jj)
# sigma_hat^2 = SSR / (n - k) = sum(e_i^2) / (n - k)

# The fit_ols helper computes all statistics from the normal equation
fit = fit_ols(X, y)

print(f"\n=== Coefficient Significance ===")
print(f"Residual sigma_hat = ${fit['sigma_hat']:,.0f}")
print(f"Degrees of freedom: {fit['n'] - fit['k']:,}")
print_coef_table(feature_names, fit)

# INTERPRETATION: The t-statistic tests whether each coefficient is
# significantly different from zero. A large |t| (and small p) means
# the feature has a statistically significant linear relationship with
# price. But statistical significance != practical importance — a
# significant coefficient of $100 per sqm is far less impactful than
# one of $5,000 per sqm.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert all(se > 0 for se in fit["se_beta"]), "All standard errors must be positive"
assert all(0 <= p <= 1 for p in fit["p_values"]), "All p-values must be valid"
print("\n--- Checkpoint 4 passed --- significance testing completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — R-squared, Adjusted R-squared, and F-Statistic
# ════════════════════════════════════════════════════════════════════════
# R-squared = 1 - SSR/SST = proportion of variance explained
# Adjusted R-squared = 1 - (1-R2)(n-1)/(n-k) — penalises more features
# F = (SSE / (k-1)) / (SSR / (n-k))

print(f"\n=== Model Fit Statistics ===")
print(f"SST (total):     {fit['SST']:,.0f}")
print(f"SSR (residual):  {fit['SSR']:,.0f}")
print(f"SSE (explained): {fit['SSE']:,.0f}")
print(
    f"Check: SST = SSR + SSE -> {fit['SST']:,.0f} "
    f"approx {fit['SSR'] + fit['SSE']:,.0f}"
)
print(f"\nR-squared:          {fit['R2']:.6f} ({fit['R2']:.2%} of variance explained)")
print(f"Adjusted R-squared: {fit['adj_R2']:.6f}")
print(f"F-statistic:        {fit['f_stat']:.2f} (p < {fit['f_p_value']:.2e})")
print(f"RMSE:               ${fit['sigma_hat']:,.0f}")
print(f"MAE:                ${np.mean(np.abs(fit['residuals'])):,.0f}")

# INTERPRETATION: R-squared tells us the fraction of price variation
# explained by the three features. The F-test confirms the model is
# significantly better than predicting the mean for everyone. But the
# unexplained variation — location, renovation, market timing — matters.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert 0 < fit["R2"] < 1, "R-squared must be between 0 and 1"
assert fit["adj_R2"] <= fit["R2"], "Adjusted R-squared must be <= R-squared"
assert fit["f_stat"] > 0, "F-statistic must be positive"
assert abs(fit["SST"] - fit["SSR"] - fit["SSE"]) < 1, "SST = SSR + SSE must hold"
print("\n--- Checkpoint 5 passed --- model evaluation completed\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Actual vs Predicted
# ════════════════════════════════════════════════════════════════════════

path = save_actual_vs_predicted(
    y,
    fit["y_hat"],
    title="OLS: Actual vs Predicted Price",
    filename="01_ols_actual_vs_predicted.html",
)
print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — HDB Valuation in Practice
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A property agent at a Singapore real estate firm needs to
# advise a client selling a 4-room flat in Tampines, 92 sqm, floor 8,
# with 75 years of lease remaining. The agent uses this OLS model.
#
# The model predicts a price, but the agent must consider:
# - R-squared shows how much price variation is explained
# - RMSE shows how far off individual predictions can be
# - The intercept is NOT the price of a "zero-area" flat
#
# BUSINESS IMPACT: Overpricing a listing by $50K means it sits on the
# market for months. Underpricing means the client loses $50K. The
# confidence interval from the SE tells the agent: "I am 95% confident
# the contribution of each additional sqm is between $X and $Y."

r2_pct = fit["R2"] * 100
unexplained_pct = 100 - r2_pct
rmse = fit["sigma_hat"]

print(f"\n--- Business Application: HDB Valuation ---")
print(f"  Model explains {r2_pct:.1f}% of price variation")
print(f"  {unexplained_pct:.1f}% is driven by location, renovation, market timing")
print(f"  RMSE = ${rmse:,.0f} — individual predictions can be off by this much")
print(f"  A property agent using this model should always disclose the uncertainty")
print(f"  A $50K mispricing either leaves money on the table or stalls the sale")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED (5.1)")
print("=" * 70)
print(
    """
  - OLS from scratch: beta = (X'X)^{-1}X'y — matrix derivation implemented
  - Ceteris paribus interpretation: "all else equal, one more sqm..."
  - t-statistics: H0 beta_j=0, SE from sigma_hat^2 * (X'X)^{-1}
  - R-squared, adjusted R-squared, F-statistic — model vs intercept-only
  - Practical valuation: understanding what the model can and cannot tell you

  NEXT: In 02_diagnostics.py you'll check whether the OLS assumptions
  actually hold — normality, heteroscedasticity, and multicollinearity.
"""
)

print("--- Exercise 5.1 complete --- OLS from Scratch")
