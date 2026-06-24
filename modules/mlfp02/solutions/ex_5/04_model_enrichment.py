# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 5.4: Model Enrichment and Evaluation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Extend models with polynomial and interaction terms
#   - Apply dummy variable encoding with a base category
#   - Cross-validate with train/test split and compute out-of-sample R-squared
#   - Compare model complexity: simple vs enriched vs categorical
#
# PREREQUISITES: Exercise 5.1-5.3 (OLS, diagnostics, WLS)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Load data and fit baseline OLS
#   2. Add polynomial and interaction terms
#   3. Dummy variable encoding for flat type
#   4. Train/test split: out-of-sample evaluation
#   5. Model comparison and business interpretation
#
# THEORY:
#   Linear regression is "linear in parameters" — you can add x^2,
#   x1*x2, or dummy variables and still use OLS. The question is
#   whether the added complexity improves out-of-sample prediction
#   or just overfits the training data.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from shared.mlfp02.ex_5 import (
    NUMERIC_FEATURES,
    TARGET,
    BASE_FLAT_TYPE,
    OUTPUT_DIR,
    load_hdb_clean,
    build_design_matrix,
    fit_ols,
    print_coef_table,
    save_actual_vs_predicted,
    save_residual_diagnostics,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Data and Fit Baseline
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MLFP02 Exercise 5.4: Model Enrichment and Evaluation")
print("=" * 70)

hdb_clean = load_hdb_clean()
X, y, feature_names = build_design_matrix(hdb_clean)
n_obs, k = X.shape
X_raw = X[:, 1:]  # Without intercept

fit_baseline = fit_ols(X, y)
r_squared = fit_baseline["R2"]
adj_r_squared = fit_baseline["adj_R2"]
SSR = fit_baseline["SSR"]
SST = fit_baseline["SST"]

print(f"\n  Baseline OLS: R-squared={r_squared:.6f}, Adj R-squared={adj_r_squared:.6f}")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Beyond Straight Lines
# ════════════════════════════════════════════════════════════════════════
# "Linear regression" is linear in PARAMETERS, not in features.
# You can transform features any way you like — square them, multiply
# them, take logs — and still use the normal equation. The key
# question: does the extra complexity capture real patterns or just
# noise?
#
# Analogy: Imagine fitting a price model for HDB flats. A straight
# line says "each extra sqm adds $X." But reality might be: small
# flats have a premium per sqm (scarcity), large flats have a
# premium per sqm (luxury). A quadratic term captures this curve.
# An interaction term says "the storey premium is bigger for large
# flats" — a penthouse effect.
#
# WHY THIS MATTERS: A property developer evaluating whether to build
# 100 small flats or 50 large ones needs to know whether the
# price-per-sqm curve is linear, convex, or concave. The polynomial
# and interaction terms answer this directly.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Polynomial and Interaction Terms
# ════════════════════════════════════════════════════════════════════════
# Non-linearity: floor_area^2 captures diminishing/increasing returns
# Interactions: storey * area captures "premium for high-floor large flats"

print(f"\n=== Polynomial and Interaction Terms ===")

area = X_raw[:, 0]
storey = X_raw[:, 1]
lease = X_raw[:, 2]

X_enriched = np.column_stack(
    [
        np.ones(n_obs),
        area,
        storey,
        lease,
        area**2,  # Polynomial: diminishing returns on area
        storey * area,  # Interaction: high-floor premium x size
        lease * area,  # Interaction: lease x size
    ]
)
enriched_names = [
    "intercept",
    "area",
    "storey",
    "lease",
    "area_sq",
    "storey_x_area",
    "lease_x_area",
]
k_enriched = X_enriched.shape[1]

# Fit enriched model
beta_enriched = np.linalg.lstsq(X_enriched, y, rcond=None)[0]
y_hat_enriched = X_enriched @ beta_enriched
resid_enriched = y - y_hat_enriched
ssr_enriched = float(np.sum(resid_enriched**2))
r2_enriched = 1 - ssr_enriched / SST
adj_r2_enriched = 1 - (1 - r2_enriched) * (n_obs - 1) / (n_obs - k_enriched)

# F-test: enriched vs simple model
f_improvement = ((SSR - ssr_enriched) / (k_enriched - k)) / (
    ssr_enriched / (n_obs - k_enriched)
)
f_p_improvement = 1 - stats.f.cdf(
    f_improvement, dfn=k_enriched - k, dfd=n_obs - k_enriched
)

print(f"{'Feature':<20} {'Coefficient':>14}")
print("-" * 38)
for name, coef in zip(enriched_names, beta_enriched):
    print(f"{name:<20} {coef:>14,.4f}")

print(f"\nSimple model:   R-squared={r_squared:.6f}, Adj R-squared={adj_r_squared:.6f}")
print(
    f"Enriched model: R-squared={r2_enriched:.6f}, Adj R-squared={adj_r2_enriched:.6f}"
)
print(f"F-test (enriched vs simple): F={f_improvement:.2f}, p={f_p_improvement:.2e}")
print(
    f"Enriched model is "
    f"{'significantly better' if f_p_improvement < 0.05 else 'NOT significantly better'}"
)

# INTERPRETATION: The area^2 term captures non-linearity — perhaps
# price per sqm increases for very large flats (premium penthouses)
# or decreases (diminishing returns). The interaction storey*area
# captures whether the storey premium is larger for bigger flats.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (
    r2_enriched >= r_squared - 0.001
), "Adding features should not decrease R-squared substantially"
print("\n--- Checkpoint 2 passed --- enriched model built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Dummy Variable Encoding
# ════════════════════════════════════════════════════════════════════════
# Categorical variables -> binary dummies. Drop one category to avoid
# the dummy variable trap (perfect multicollinearity with intercept).

print(f"\n=== Dummy Variable Encoding ===")

flat_types_in_data = sorted(hdb_clean["flat_type"].unique().to_list())
print(f"Flat types: {flat_types_in_data}")

# Use BASE_FLAT_TYPE as base category (most common)
dummy_categories = [ft for ft in flat_types_in_data if ft != BASE_FLAT_TYPE]

# Build dummy columns
dummy_arrays = []
for ft in dummy_categories:
    dummy = (hdb_clean["flat_type"].to_numpy() == ft).astype(np.float64)
    dummy_arrays.append(dummy)

X_with_dummies = np.column_stack(
    [
        np.ones(n_obs),
        X_raw,  # Original numeric features
        np.column_stack(dummy_arrays),  # Dummy variables
    ]
)
dummy_names = (
    ["intercept"]
    + list(NUMERIC_FEATURES)
    + [f"flat_{ft.replace(' ', '_')}" for ft in dummy_categories]
)
k_dummy = X_with_dummies.shape[1]

# Fit model with dummies
beta_dummy = np.linalg.lstsq(X_with_dummies, y, rcond=None)[0]
y_hat_dummy = X_with_dummies @ beta_dummy
ssr_dummy = float(np.sum((y - y_hat_dummy) ** 2))
r2_dummy = 1 - ssr_dummy / SST
adj_r2_dummy = 1 - (1 - r2_dummy) * (n_obs - 1) / (n_obs - k_dummy)

print(f"\nBase category: {BASE_FLAT_TYPE}")
print(f"\n{'Feature':<30} {'Coefficient':>14}")
print("-" * 48)
for name, coef in zip(dummy_names, beta_dummy):
    print(f"{name:<30} {coef:>14,.0f}")

print(
    f"\nModel with dummies: R-squared={r2_dummy:.6f}, Adj R-squared={adj_r2_dummy:.6f}"
)
print(f"Improvement over simple: Delta R-squared={r2_dummy - r_squared:+.6f}")

# INTERPRETATION: Each dummy coefficient represents the price premium
# (or discount) relative to the base category (3 ROOM). For example,
# if the 5 ROOM coefficient is +$150K, then 5-room flats sell for
# $150K more than 3-room flats, all else equal.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert r2_dummy > r_squared, "Adding flat type should improve R-squared"
assert len(dummy_categories) == len(flat_types_in_data) - 1, "Should drop one category"
print("\n--- Checkpoint 3 passed --- dummy encoding completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Train/Test Split: Out-of-Sample Evaluation
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Train/Test Split ===")

rng = np.random.default_rng(seed=42)
n_total_obs = X_with_dummies.shape[0]
indices = rng.permutation(n_total_obs)
split_point = int(0.8 * n_total_obs)

train_idx = indices[:split_point]
test_idx = indices[split_point:]

X_train = X_with_dummies[train_idx]
y_train = y[train_idx]
X_test = X_with_dummies[test_idx]
y_test = y[test_idx]

# Fit on train
beta_train = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

# Evaluate on both
y_train_pred = X_train @ beta_train
y_test_pred = X_test @ beta_train

r2_train = 1 - float(np.sum((y_train - y_train_pred) ** 2)) / float(
    np.sum((y_train - y_train.mean()) ** 2)
)
r2_test = 1 - float(np.sum((y_test - y_test_pred) ** 2)) / float(
    np.sum((y_test - y_test.mean()) ** 2)
)
rmse_train = float(np.sqrt(np.mean((y_train - y_train_pred) ** 2)))
rmse_test = float(np.sqrt(np.mean((y_test - y_test_pred) ** 2)))
mae_train = float(np.mean(np.abs(y_train - y_train_pred)))
mae_test = float(np.mean(np.abs(y_test - y_test_pred)))

print(f"Train: n={len(train_idx):,}")
print(f"Test:  n={len(test_idx):,}")
print(f"\n{'Metric':<12} {'Train':>14} {'Test':>14} {'Delta':>10}")
print("-" * 54)
print(
    f"{'R-squared':<12} {r2_train:>14.6f} {r2_test:>14.6f} {r2_test - r2_train:>+10.6f}"
)
print(
    f"{'RMSE':<12} ${rmse_train:>12,.0f} ${rmse_test:>12,.0f} "
    f"${rmse_test - rmse_train:>+8,.0f}"
)
print(
    f"{'MAE':<12} ${mae_train:>12,.0f} ${mae_test:>12,.0f} "
    f"${mae_test - mae_train:>+8,.0f}"
)

gap = abs(r2_train - r2_test)
print(f"\nTrain-test R-squared gap: {gap:.4f}")
if gap < 0.02:
    print("Minimal overfitting — model generalises well")
elif gap < 0.05:
    print("Slight overfitting — consider regularisation")
else:
    print("OVERFITTING — model is too complex for the data")

# INTERPRETATION: If train R-squared >> test R-squared, the model
# memorises training data instead of learning generalisable patterns.
# The train-test gap tells you whether your model complexity is
# appropriate.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert r2_test > 0, "Out-of-sample R-squared must be positive"
assert (
    r2_train >= r2_test - 0.05
), "Train R-squared should be >= test R-squared (approx)"
print("\n--- Checkpoint 4 passed --- out-of-sample evaluation completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Model Comparison Summary
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Model Comparison Summary ===")
print(f"{'Model':<30} {'R-sq':>10} {'Adj R-sq':>10} {'k':>4}")
print("-" * 58)
print(f"{'Simple (3 features)':<30} {r_squared:>10.6f} {adj_r_squared:>10.6f} {k:>4}")
print(
    f"{'Enriched (poly+interact)':<30} {r2_enriched:>10.6f} {adj_r2_enriched:>10.6f} "
    f"{k_enriched:>4}"
)
print(
    f"{'With flat type dummies':<30} {r2_dummy:>10.6f} {adj_r2_dummy:>10.6f} {k_dummy:>4}"
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
print("\n--- Checkpoint 5 passed --- model comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Test Set Actual vs Predicted
# ════════════════════════════════════════════════════════════════════════

path = save_actual_vs_predicted(
    y_test,
    y_test_pred,
    title="Full Model: Actual vs Predicted (Test Set)",
    filename="04_test_actual_vs_predicted.html",
)
print(f"Saved: {path}")

# --- Polynomial fit curve: price vs floor area with linear + quadratic ---
area_sorted_idx = np.argsort(area)
area_sorted = area[area_sorted_idx]
# Linear prediction (area only): intercept + beta_area * area
beta_baseline = fit_baseline["beta"]
y_linear = beta_baseline[0] + beta_baseline[1] * area_sorted
# Enriched prediction for area dimension (storey=median, lease=median)
med_storey = float(np.median(storey))
med_lease = float(np.median(lease))
y_poly = (
    beta_enriched[0]
    + beta_enriched[1] * area_sorted
    + beta_enriched[2] * med_storey
    + beta_enriched[3] * med_lease
    + beta_enriched[4] * area_sorted**2
    + beta_enriched[5] * med_storey * area_sorted
    + beta_enriched[6] * med_lease * area_sorted
)

sample = min(3000, n_obs)
fig_poly = go.Figure()
fig_poly.add_trace(
    go.Scatter(
        x=area[:sample].tolist(),
        y=y[:sample].tolist(),
        mode="markers",
        marker={"size": 2, "opacity": 0.2, "color": "#94A3B8"},
        name="Data",
    )
)
fig_poly.add_trace(
    go.Scatter(
        x=area_sorted.tolist(),
        y=y_linear.tolist(),
        mode="lines",
        line={"color": "#2563EB", "width": 2},
        name="Linear",
    )
)
fig_poly.add_trace(
    go.Scatter(
        x=area_sorted.tolist(),
        y=y_poly.tolist(),
        mode="lines",
        line={"color": "#DC2626", "width": 2, "dash": "dash"},
        name="Polynomial + Interactions",
    )
)
fig_poly.update_layout(
    title="Price vs Floor Area — Does a Curve Fit Better Than a Line?",
    xaxis_title="Floor Area (sqm)",
    yaxis_title="Predicted Price (SGD)",
    height=450,
)
path_poly = OUTPUT_DIR / "04_polynomial_fit_curves.html"
fig_poly.write_html(str(path_poly))
print(f"Saved: {path_poly}")

# --- Train/test comparison bar chart: R-squared, RMSE, MAE ---
metrics = ["R-squared", "RMSE ($K)", "MAE ($K)"]
train_vals = [r2_train, rmse_train / 1000, mae_train / 1000]
test_vals = [r2_test, rmse_test / 1000, mae_test / 1000]

fig_tt = go.Figure()
fig_tt.add_trace(
    go.Bar(
        x=metrics,
        y=train_vals,
        name="Train",
        marker_color="#2563EB",
    )
)
fig_tt.add_trace(
    go.Bar(
        x=metrics,
        y=test_vals,
        name="Test",
        marker_color="#DC2626",
    )
)
fig_tt.update_layout(
    title="Train vs Test Performance — Is the Model Overfitting?",
    yaxis_title="Value",
    barmode="group",
    height=400,
)
path_tt = OUTPUT_DIR / "04_train_test_comparison.html"
fig_tt.write_html(str(path_tt))
print(f"Saved: {path_tt}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Property Developer Portfolio Decision
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A property developer in Singapore is deciding between two
# sites for a new HDB-style development. Site A allows 100 3-room
# flats (67 sqm each). Site B allows 50 5-room flats (110 sqm each).
# Construction cost is similar per sqm.
#
# Using the dummy-encoded model, the developer can estimate:
# - Revenue from 100 x 3-room flats (base category)
# - Revenue from 50 x 5-room flats (base + dummy coefficient)
#
# The interaction terms tell the developer whether high floors
# command a bigger premium for large flats (the penthouse effect).
# The train/test gap tells the developer how reliable these
# estimates are for properties not in the training data.
#
# BUSINESS IMPACT: A 0.01 difference in R-squared translates to
# millions of dollars in aggregate valuation uncertainty across
# a 50-unit development. The model comparison table directly
# informs which features the developer should market (floor area,
# storey, flat type) and which add noise.

# Find the 5-ROOM dummy coefficient
five_room_idx = None
for i, name in enumerate(dummy_names):
    if "5_ROOM" in name:
        five_room_idx = i
        break

print(f"\n--- Business Application: Developer Portfolio Decision ---")
if five_room_idx is not None:
    five_room_premium = beta_dummy[five_room_idx]
    print(f"  5-ROOM premium over 3-ROOM (base): ${five_room_premium:,.0f}")
    print(f"  Site A (100 x 3-room): revenue driven by base price")
    print(f"  Site B (50 x 5-room):  each unit commands ${five_room_premium:,.0f} more")
    print(f"  But 50 units vs 100 units — the developer must weigh volume vs premium")
else:
    print(f"  5-ROOM flat type not found in data")

print(f"\n  Model reliability:")
print(f"  Train-test R-squared gap: {gap:.4f}")
print(f"  Out-of-sample RMSE: ${rmse_test:,.0f}")
print(f"  These estimates can be off by +/- ${rmse_test:,.0f} per unit")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED (5.4)")
print("=" * 70)
print(
    """
  - Polynomial terms (area^2) and interactions (storey x area)
  - F-test for nested model comparison (simple vs enriched)
  - Dummy encoding with base category to avoid the dummy trap
  - Train/test split: out-of-sample R-squared, RMSE, MAE
  - Model complexity trade-off: more features != better prediction
  - Business reasoning: how model choice affects portfolio decisions
"""
)

print("=" * 70)
print("  EXERCISE 5 COMPLETE — Linear Regression")
print("=" * 70)
print(
    """
  FULL EXERCISE SUMMARY:
  - 5.1: OLS from scratch, coefficient interpretation, significance
  - 5.2: Diagnostics — VIF, residual normality, Breusch-Pagan
  - 5.3: Weighted Least Squares for heteroscedastic data
  - 5.4: Model enrichment, dummy encoding, train/test evaluation

  NEXT: In Exercise 6 you'll build logistic regression for binary
  classification. You'll implement the sigmoid function, maximise
  the Bernoulli log-likelihood, interpret coefficients as odds ratios,
  and perform ANOVA for multi-group comparison.
"""
)
