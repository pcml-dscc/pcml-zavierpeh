# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 6.1: Logistic Regression from Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement the sigmoid function with numerical stability
#   - Build logistic regression via MLE (Bernoulli log-likelihood)
#   - Optimise with scipy L-BFGS-B using an analytical gradient
#   - Compare your from-scratch model with sklearn LogisticRegression
#   - Apply logistic regression to Singapore HDB resale classification
#
# PREREQUISITES: Exercise 5 — OLS, t-statistics, scipy.optimize.minimize
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — from linear regression to the sigmoid boundary
#   2. Build — sigmoid function + negative log-likelihood + gradient
#   3. Train — MLE optimisation + sklearn comparison
#   4. Visualise — sigmoid shape + coefficient agreement
#   5. Apply — HDB valuation: classifying above-median transactions
#
# ─── FRAMEWORK-FIRST EXEMPTION ──────────────────────────────────────────
# This exercise uses raw sklearn.linear_model.LogisticRegression ONCE, as
# a correctness oracle for the from-scratch MLE implementation. This is a
# documented exemption to the framework-first rule, for three reasons:
#
# 1. The pedagogical beat is "build the optimiser by hand, then check
#    it agrees with a trusted reference." Replacing the reference with
#    kailash-ml's TrainingPipeline would compare one abstraction against
#    another abstraction — both wrap the same L-BFGS-B solver, so the
#    agreement would be tautological and the lesson would collapse.
#
# 2. M2 is "Statistical Mastery" — logistic regression is taught here
#    as an inference tool (coefficients, odds ratios, likelihood), NOT
#    as a production classifier. Students meet TrainingPipeline for the
#    first time in M3 ex_7/01, where the engine is the correct primitive.
#
# 3. sklearn.metrics (accuracy_score, confusion_matrix, roc_curve, etc.)
#    used later in this exercise are stateless utility functions, not
#    framework bypasses — kailash-ml consumes them internally.
#
# Forward pointer: See modules/mlfp03/local/ex_7/01 for the first
# canonical use of TrainingPipeline.
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from shared.mlfp02.ex_6 import (
    FEATURE_COLS,
    OUTPUT_DIR,
    build_classification_frame,
    build_design_matrix,
    load_hdb_recent,
    neg_ll_gradient,
    neg_log_likelihood_logistic,
    sigmoid,
    unscale_coefficients,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — From Linear Regression to the Sigmoid Boundary
# ════════════════════════════════════════════════════════════════════════
# Linear regression predicts a continuous value: y = Xb + e. But what
# if y is binary — 0 or 1? The linear model can predict values outside
# [0, 1], which makes no sense for a probability.
#
# The sigmoid function s(z) = 1 / (1 + exp(-z)) squashes any real
# number into (0, 1). Logistic regression wraps linear regression
# inside the sigmoid:
#
#   P(y = 1 | X) = s(Xb) = 1 / (1 + exp(-Xb))
#
# Instead of minimising squared error, we maximise the LIKELIHOOD.
# The Bernoulli log-likelihood is:
#
#   l(b) = sum[y_i log(p_i) + (1 - y_i) log(1 - p_i)]


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: load data and verify the sigmoid
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Logistic Regression from Scratch — HDB Classification")
print("=" * 70)

# TODO: Load HDB transactions (2020+) using the shared helper.
# Hint: load_hdb_recent() returns a polars DataFrame.
hdb_recent = ____

# TODO: Build classification frame with binary target (above/below median).
# Hint: build_classification_frame(hdb_recent) returns (frame, median_price).
frame, median_price = ____
print(f"\n  Data loaded: {hdb_recent.height:,} transactions (2020+)")
print(f"  Median price: ${median_price:,.0f}")

n_total = frame.height
n_positive = frame.filter(pl.col("high_price") == 1).height
print(f"  Total: {n_total:,}, Positive: {n_positive:,} ({n_positive/n_total:.1%})")

# TODO: Build standardised design matrix with intercept.
# Hint: build_design_matrix(frame) returns (X, y, X_mean, X_std, feature_names).
X, y, X_mean, X_std, feature_names = ____
n_obs, n_params = X.shape

# Verify sigmoid properties
print(f"\n--- Sigmoid Properties ---")
test_values = [-10, -5, -1, 0, 1, 5, 10]
for z in test_values:
    s = sigmoid(np.array([z]))[0]
    print(f"  s({z:>3}) = {s:.6f}")

# Key properties
assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10, "s(0) must equal 0.5"
assert sigmoid(np.array([100.0]))[0] > 0.999, "s(large) must be near 1"
assert sigmoid(np.array([-100.0]))[0] < 0.001, "s(very negative) must be near 0"

# TODO: Verify symmetry: s(-z) = 1 - s(z).
# Hint: compute sigmoid of -z_test and compare to 1 - sigmoid(z_test).
z_test = np.array([2.5])
assert abs(sigmoid(-z_test)[0] - (1 - sigmoid(z_test)[0])) < 1e-10, "Symmetry must hold"

# Derivative: s'(z) = s(z)(1 - s(z))
s_val = sigmoid(z_test)[0]
# TODO: Compute the analytical derivative of the sigmoid at z_test.
# Hint: s'(z) = s(z) * (1 - s(z)).
deriv_analytical = ____
deriv_numerical = (sigmoid(z_test + 1e-7)[0] - sigmoid(z_test - 1e-7)[0]) / (2e-7)
print(f"\n  Derivative at z=2.5:")
print(f"  Analytical s'(z) = s(z)(1-s(z)) = {deriv_analytical:.6f}")
print(f"  Numerical:  {deriv_numerical:.6f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert abs(n_positive / n_total - 0.5) < 0.02, "Median split should give ~50/50"
assert X.shape == (n_obs, n_params), "Design matrix shape incorrect"
assert abs(deriv_analytical - deriv_numerical) < 1e-5, "Derivative check must pass"
print("\n[ok] Checkpoint 1 passed — data + sigmoid verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: MLE optimisation + sklearn comparison
# ════════════════════════════════════════════════════════════════════════

# TODO: Set initial guess beta0 as a vector of zeros with length n_params.
# Hint: np.zeros(n_params).
beta0 = ____

# TODO: Optimise the negative log-likelihood using scipy minimize.
# Hint: minimize(neg_log_likelihood_logistic, beta0, args=(X, y),
#                method="L-BFGS-B", jac=neg_ll_gradient,
#                options={"maxiter": 1000, "ftol": 1e-12})
result = ____

beta_scratch = result.x
ll_scratch = -result.fun

print(f"\n=== Logistic Regression from Scratch ===")
print(f"Converged: {result.success}")
print(f"Log-likelihood: {ll_scratch:.2f}")
print(f"\n{'Feature':<20} {'Coefficient':>14}")
print("-" * 38)
for name, coef in zip(feature_names, beta_scratch):
    print(f"{name:<20} {coef:>14.6f}")

# TODO: Compute predicted probabilities using sigmoid(X @ beta_scratch).
p_scratch = ____
y_pred_scratch = (p_scratch >= 0.5).astype(int)
acc_scratch = accuracy_score(y, y_pred_scratch)
print(f"\nAccuracy (from scratch): {acc_scratch:.4f} ({acc_scratch:.1%})")

# Compare with sklearn
X_scaled = X[:, 1:]  # drop intercept — sklearn adds its own
sklearn_model = LogisticRegression(
    penalty=None,  # type: ignore[arg-type]  # sklearn stub types penalty as str; None is valid at runtime
    max_iter=1000,
    solver="lbfgs",
    tol=1e-8,
)
# TODO: Fit sklearn_model on X_scaled and y.
# Hint: sklearn_model.fit(X_scaled, y)
____

beta_sklearn = np.concatenate(
    [
        np.asarray(sklearn_model.intercept_).ravel(),
        np.asarray(sklearn_model.coef_).ravel(),
    ]
)

print(f"\n=== Comparison: Scratch vs sklearn ===")
print(f"{'Feature':<20} {'Scratch':>14} {'sklearn':>14} {'|diff|':>10}")
print("-" * 62)
for i, name in enumerate(feature_names):
    diff = abs(beta_scratch[i] - beta_sklearn[i])
    print(f"{name:<20} {beta_scratch[i]:>14.6f} {beta_sklearn[i]:>14.6f} {diff:>10.6f}")

acc_sklearn = sklearn_model.score(X_scaled, y)
print(f"\nAccuracy (scratch): {acc_scratch:.6f}")
print(f"Accuracy (sklearn): {acc_sklearn:.6f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert result.success, "Logistic regression must converge"
assert acc_scratch > 0.55, f"Accuracy should beat baseline 50%, got {acc_scratch:.1%}"
assert np.allclose(
    beta_scratch, beta_sklearn, atol=0.1
), "Scratch and sklearn coefficients should agree"
print("\n[ok] Checkpoint 2 passed — from-scratch matches sklearn\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: sigmoid shape + coefficient agreement
# ════════════════════════════════════════════════════════════════════════

# Plot 1: Sigmoid function and its derivative
z_range = np.linspace(-8, 8, 200)
sig_vals = sigmoid(z_range)
deriv_vals = sig_vals * (1 - sig_vals)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=z_range, y=sig_vals, name="s(z)"))
fig1.add_trace(
    go.Scatter(
        x=z_range,
        y=deriv_vals,
        name="s'(z) = s(z)(1-s(z))",
        line={"dash": "dash"},
    )
)
fig1.add_hline(y=0.5, line_dash="dot", line_color="grey", annotation_text="P = 0.5")
fig1.update_layout(
    title="Sigmoid Function and Derivative",
    xaxis_title="z = Xb (log-odds)",
    yaxis_title="Value",
)
fig1.write_html(str(OUTPUT_DIR / "sigmoid_properties.html"))
print(f"Saved: {OUTPUT_DIR / 'sigmoid_properties.html'}")

# TODO: Create a grouped bar chart comparing scratch vs sklearn coefficients.
# Hint: use go.Bar with feature_names[1:] as x, beta_scratch[1:] and
#       beta_sklearn[1:] as y values. Set barmode="group".
fig2 = go.Figure()
feat_labels = feature_names[1:]
fig2.add_trace(
    go.Bar(
        name="From scratch",
        x=feat_labels,
        y=____,
    )
)
fig2.add_trace(
    go.Bar(
        name="sklearn",
        x=feat_labels,
        y=____,
    )
)
fig2.update_layout(
    title="Coefficient Comparison: From Scratch vs sklearn",
    yaxis_title="Coefficient (standardised scale)",
    barmode="group",
)
fig2.write_html(str(OUTPUT_DIR / "coefficient_comparison.html"))
print(f"Saved: {OUTPUT_DIR / 'coefficient_comparison.html'}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 3 passed — sigmoid + comparison visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: HDB valuation classification
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore property valuation firm needs to quickly triage
# incoming HDB resale transactions into "high-value" vs "standard"
# categories for resource allocation.

n_weekly = 300
time_per_manual_min = 15
valuer_hourly_rate = 85

# TODO: Calculate weekly manual cost: n_weekly * time_per_manual / 60 * rate.
weekly_manual_cost = ____

# With the model, only uncertain cases (P near 0.5) need human review
p_all = sigmoid(X @ beta_scratch)
uncertain_mask = (p_all >= 0.35) & (p_all <= 0.65)
pct_uncertain = uncertain_mask.mean()
weekly_model_reviews = int(n_weekly * pct_uncertain)
weekly_model_cost = weekly_model_reviews * time_per_manual_min / 60 * valuer_hourly_rate

print(f"\n=== Real-World Application: Property Triage ===")
print(f"  Weekly transactions: {n_weekly}")
print(f"  Manual cost (all reviewed): S${weekly_manual_cost:,.0f}/week")
print(f"  Model-assisted ({pct_uncertain:.0%} uncertain, human review):")
print(f"    Reviews needed: {weekly_model_reviews}")
print(f"    Cost: S${weekly_model_cost:,.0f}/week")
print(
    f"    Savings: S${weekly_manual_cost - weekly_model_cost:,.0f}/week "
    f"(S${(weekly_manual_cost - weekly_model_cost) * 52:,.0f}/year)"
)
print(f"  Model accuracy on clear cases: {acc_scratch:.1%}")
