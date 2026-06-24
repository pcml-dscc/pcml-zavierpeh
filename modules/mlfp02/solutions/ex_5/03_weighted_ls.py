# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 5.3: Weighted Least Squares (WLS)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement WLS when heteroscedasticity is present
#   - Estimate variance weights from fitted values
#   - Compare OLS and WLS coefficients and standard errors
#   - Understand when WLS improves inference vs point estimates
#
# PREREQUISITES: Exercise 5.1-5.2 (OLS, diagnostics, Breusch-Pagan)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load data and fit baseline OLS
#   2. Estimate variance function from residuals
#   3. Implement WLS: beta = (X'WX)^{-1}X'Wy
#   4. Compare OLS and WLS results
#   5. Visualise the effect of weighting
#
# THEORY:
#   When Var(e_i) = sigma_i^2 (not constant), OLS gives unbiased but
#   inefficient estimates. WLS weights each observation by 1/sigma_i^2,
#   giving less weight to high-variance observations.
#   beta_wls = (X'WX)^{-1}X'Wy where W = diag(1/sigma_i^2)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.mlfp02.ex_5 import (
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    load_hdb_clean,
    build_design_matrix,
    fit_ols,
    print_coef_table,
    save_actual_vs_predicted,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Data and Fit Baseline OLS
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MLFP02 Exercise 5.3: Weighted Least Squares")
print("=" * 70)

hdb_clean = load_hdb_clean()
X, y, feature_names = build_design_matrix(hdb_clean)
n_obs, k = X.shape

fit_baseline = fit_ols(X, y)
beta_ols = fit_baseline["beta"]
residuals = fit_baseline["residuals"]
y_hat = fit_baseline["y_hat"]
r_squared = fit_baseline["R2"]
SST = fit_baseline["SST"]
SSR = fit_baseline["SSR"]

print(
    f"\n  Baseline OLS: R-squared={r_squared:.6f}, RMSE=${fit_baseline['sigma_hat']:,.0f}"
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Weight Observations?
# ════════════════════════════════════════════════════════════════════════
# OLS treats every observation equally. But in housing data, a $1.2M
# executive flat has more price variation than a $300K 3-room flat —
# the "noise" is louder for expensive properties.
#
# Analogy: Imagine averaging exam scores from two classes. Class A has
# 30 students with tightly clustered scores (low variance). Class B
# has 30 students with wildly spread scores (high variance). A simple
# average weights them equally, but you would trust Class A's average
# more. WLS does exactly this — it trusts low-variance observations
# more than high-variance ones.
#
# WHY THIS MATTERS: A real estate analytics firm building an automated
# valuation model (AVM) for bank mortgage approvals needs accurate
# confidence intervals. OLS confidence intervals are too narrow for
# expensive properties (underestimating risk) and too wide for cheap
# ones (overestimating risk). WLS fixes the intervals.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Estimate Variance Function
# ════════════════════════════════════════════════════════════════════════
# We model variance as a function of X: Var(e_i) ~ (X @ gamma)^2
# Use |residuals| as a proxy for standard deviation.

print(f"\n=== Estimating Variance Function ===")

abs_resid = np.abs(residuals)
# Fit |residuals| ~ X to get a variance model
w_beta = np.linalg.lstsq(X, abs_resid, rcond=None)[0]
variance_hat = np.maximum((X @ w_beta) ** 2, 1e-6)  # Estimated variance per obs
weights = 1.0 / variance_hat

print(f"  Weight range: [{weights.min():.6f}, {weights.max():.6f}]")
print(f"  Weight ratio (max/min): {weights.max() / weights.min():.1f}x")
print(
    f"  Observations with above-median weight: {np.sum(weights > np.median(weights)):,}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Implement WLS: beta = (X'WX)^{-1}X'Wy
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Weighted Least Squares ===")

W = np.diag(weights)
XtWX = X.T @ W @ X
XtWy = X.T @ W @ y
beta_wls = np.linalg.solve(XtWX, XtWy)

y_hat_wls = X @ beta_wls
residuals_wls = y - y_hat_wls
ssr_wls = float(np.sum(residuals_wls**2))
r2_wls = 1 - ssr_wls / SST

# WLS standard errors
sigma_sq_wls = float(np.sum(weights * residuals_wls**2)) / (n_obs - k)
XtWX_inv = np.linalg.inv(XtWX)
se_wls = np.sqrt(sigma_sq_wls * np.diag(XtWX_inv))

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(beta_wls) == k, "WLS must have same number of coefficients"
assert ssr_wls > 0, "WLS SSR must be positive"
print("--- Checkpoint 3 passed --- WLS fitted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Compare OLS and WLS
# ════════════════════════════════════════════════════════════════════════

print(f"{'Feature':<25} {'OLS beta':>12} {'WLS beta':>12} {'Delta':>10}")
print("-" * 62)
for i, name in enumerate(feature_names):
    delta = beta_wls[i] - beta_ols[i]
    print(f"{name:<25} {beta_ols[i]:>12,.2f} {beta_wls[i]:>12,.2f} {delta:>+10,.2f}")

print(f"\nOLS R-squared  = {r_squared:.6f}")
print(f"WLS R-squared  = {r2_wls:.6f}")
print(f"OLS RMSE = ${np.sqrt(SSR / n_obs):,.0f}")
print(f"WLS RMSE = ${np.sqrt(ssr_wls / n_obs):,.0f}")

# Standard error comparison
print(f"\n{'Feature':<25} {'OLS SE':>12} {'WLS SE':>12}")
print("-" * 52)
for i, name in enumerate(feature_names):
    print(f"{name:<25} {fit_baseline['se_beta'][i]:>12,.2f} {se_wls[i]:>12,.2f}")

# INTERPRETATION: WLS coefficients may differ from OLS when
# heteroscedasticity is present. WLS gives more reliable standard
# errors and confidence intervals. If OLS and WLS coefficients are
# similar, the heteroscedasticity doesn't much affect the point
# estimates — but the SEs are still more trustworthy from WLS.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert all(se > 0 for se in se_wls), "All WLS standard errors must be positive"
print("\n--- Checkpoint 4 passed --- OLS vs WLS compared\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — WLS Actual vs Predicted
# ════════════════════════════════════════════════════════════════════════

path = save_actual_vs_predicted(
    y,
    y_hat_wls,
    title="WLS: Actual vs Predicted Price",
    filename="03_wls_actual_vs_predicted.html",
)
print(f"Saved: {path}")

# --- Residual variance: OLS vs WLS (before/after weighting) ---
sample = min(3000, n_obs)
fig_var = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "OLS Residuals vs Fitted",
        "WLS Residuals vs Fitted",
    ],
)
fig_var.add_trace(
    go.Scatter(
        x=y_hat[:sample].tolist(),
        y=residuals[:sample].tolist(),
        mode="markers",
        marker={"size": 2, "opacity": 0.3, "color": "steelblue"},
        name="OLS",
    ),
    row=1,
    col=1,
)
fig_var.add_trace(
    go.Scatter(
        x=y_hat_wls[:sample].tolist(),
        y=residuals_wls[:sample].tolist(),
        mode="markers",
        marker={"size": 2, "opacity": 0.3, "color": "#D97706"},
        name="WLS",
    ),
    row=1,
    col=2,
)
fig_var.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
fig_var.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
fig_var.update_layout(
    title="Residual Spread: OLS vs WLS — Does Weighting Reduce the Fan Shape?",
    height=400,
    width=900,
    showlegend=False,
)
fig_var.update_xaxes(title_text="Predicted ($)", row=1, col=1)
fig_var.update_xaxes(title_text="Predicted ($)", row=1, col=2)
fig_var.update_yaxes(title_text="Residual ($)", row=1, col=1)
fig_var.update_yaxes(title_text="Residual ($)", row=1, col=2)
path_var = OUTPUT_DIR / "03_wls_residual_comparison.html"
fig_var.write_html(str(path_var))
print(f"Saved: {path_var}")

# --- Predicted-vs-actual comparison: OLS vs WLS side by side ---
fig_cmp = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        f"OLS (R-sq={r_squared:.4f})",
        f"WLS (R-sq={r2_wls:.4f})",
    ],
)
lo, hi = float(y.min()), float(y.max())
for col_idx, (pred, label) in enumerate([(y_hat, "OLS"), (y_hat_wls, "WLS")], start=1):
    fig_cmp.add_trace(
        go.Scatter(
            x=y[:sample].tolist(),
            y=pred[:sample].tolist(),
            mode="markers",
            marker={"size": 2, "opacity": 0.3},
            name=label,
        ),
        row=1,
        col=col_idx,
    )
    fig_cmp.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            showlegend=False,
        ),
        row=1,
        col=col_idx,
    )
fig_cmp.update_layout(
    title="Actual vs Predicted: OLS vs WLS — How Does Weighting Change Predictions?",
    height=400,
    width=900,
)
fig_cmp.update_xaxes(title_text="Actual ($)", row=1, col=1)
fig_cmp.update_xaxes(title_text="Actual ($)", row=1, col=2)
fig_cmp.update_yaxes(title_text="Predicted ($)", row=1, col=1)
fig_cmp.update_yaxes(title_text="Predicted ($)", row=1, col=2)
path_cmp = OUTPUT_DIR / "03_wls_ols_comparison.html"
fig_cmp.write_html(str(path_cmp))
print(f"Saved: {path_cmp}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Automated Valuation Model (AVM) for Mortgage Approval
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A fintech startup in Singapore builds an automated
# valuation model (AVM) for banks. The AVM must provide not just a
# point estimate but a confidence interval: "This flat is worth
# $520K, 95% CI [$480K, $560K]."
#
# With OLS, the CI for a $1.2M executive flat is too narrow (the
# model is overconfident about expensive properties) and the CI for
# a $300K 3-room flat is too wide (the model is underconfident about
# cheap properties). WLS fixes this by weighting observations
# inversely to their variance.
#
# BUSINESS IMPACT: Singapore banks use LTV (loan-to-value) ratios
# of 75-80%. A $520K valuation with $40K uncertainty means the bank
# lends $390K-$416K. If the CI is wrong, the bank either:
# - Lends too much (risk of loss if property value drops)
# - Lends too little (loses the mortgage deal to a competitor)

print(f"\n--- Business Application: AVM Confidence Intervals ---")
# Pick a representative flat and show the difference
example_x = np.array([1.0, 92.0, 8.0, 75.0])  # intercept, area, storey, lease
pred_ols = float(example_x @ beta_ols)
pred_wls = float(example_x @ beta_wls)

# OLS 95% CI
se_pred_ols = float(
    np.sqrt(
        fit_baseline["sigma_hat"] ** 2
        * (example_x @ fit_baseline["XtX_inv"] @ example_x)
    )
)
# WLS 95% CI
se_pred_wls = float(np.sqrt(sigma_sq_wls * (example_x @ XtWX_inv @ example_x)))

print(f"  Example flat: 92 sqm, floor 8, 75 years lease")
print(f"  OLS prediction: ${pred_ols:,.0f} +/- ${1.96 * se_pred_ols:,.0f} (95% CI)")
print(f"  WLS prediction: ${pred_wls:,.0f} +/- ${1.96 * se_pred_wls:,.0f} (95% CI)")
print(f"  The WLS CI better reflects the actual uncertainty for this property")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED (5.3)")
print("=" * 70)
print(
    """
  - Variance function estimation from OLS residuals
  - WLS implementation: beta = (X'WX)^{-1}X'Wy
  - Comparing OLS and WLS: coefficients, SEs, and R-squared
  - When WLS matters: heteroscedastic data with varying noise levels
  - Business reasoning: why confidence intervals matter for mortgage decisions

  NEXT: In 04_model_enrichment.py you'll extend the model with
  polynomial terms, interaction effects, dummy variables, and
  train/test evaluation.
"""
)

print("--- Exercise 5.3 complete --- Weighted Least Squares")
