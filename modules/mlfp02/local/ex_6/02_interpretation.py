# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 6.2: Odds Ratios and Threshold Optimisation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Convert logistic regression coefficients to odds ratios: exp(b)
#   - Interpret odds ratios on the original (unscaled) feature scale
#   - Optimise classification threshold using a domain cost matrix
#   - Compare cost-optimal vs F1-optimal vs default thresholds
#   - Apply threshold optimisation to Singapore HDB valuation risk
#
# PREREQUISITES: Exercise 6.1 — logistic regression from scratch
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — from coefficients to odds ratios
#   2. Build — unscale coefficients + compute odds ratios
#   3. Train — threshold sweep with cost matrix
#   4. Visualise — odds ratio forest plot + cost curve
#   5. Apply — HDB valuation risk: asymmetric error costs
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

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
# THEORY — From Coefficients to Odds Ratios
# ════════════════════════════════════════════════════════════════════════
# Logistic regression models log-odds: log(P/(1-P)) = b0 + b1*x1 + ...
#
# The ODDS of an event is P / (1 - P). When P = 0.75, odds = 3:1.
#
# An ODDS RATIO for feature j is exp(b_j). It answers: "how do the
# odds multiply when x_j increases by one unit?"
#
#   exp(b_j) > 1  -> odds increase (feature promotes the outcome)
#   exp(b_j) = 1  -> no effect
#   exp(b_j) < 1  -> odds decrease (feature inhibits the outcome)
#
# The threshold question is separate from the model. The model outputs
# a probability; the THRESHOLD decides the action.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: fit model and compute odds ratios
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Odds Ratios and Threshold Optimisation")
print("=" * 70)

# Load data and fit logistic regression (same as 6.1)
hdb_recent = load_hdb_recent()
frame, median_price = build_classification_frame(hdb_recent)
X, y, X_mean, X_std, feature_names = build_design_matrix(frame)
n_obs = X.shape[0]

beta0 = np.zeros(X.shape[1])
result = minimize(
    neg_log_likelihood_logistic,
    beta0,
    args=(X, y),
    method="L-BFGS-B",
    jac=neg_ll_gradient,
    options={"maxiter": 1000, "ftol": 1e-12},
)
beta_scratch = result.x
p_scratch = sigmoid(X @ beta_scratch)

# TODO: Convert coefficients to original scale using unscale_coefficients.
# Hint: unscale_coefficients(beta_scratch, X_mean, X_std) returns the
#       original-scale coefficient vector.
beta_original = ____

print(f"\n=== Odds Ratio Interpretation ===")
print(f"\n{'Feature':<20} {'b (original)':>14} {'Odds Ratio':>12} {'Interpretation'}")
print("-" * 80)
for i in range(1, len(feature_names)):
    # TODO: Compute the odds ratio for feature i.
    # Hint: odds ratio = np.exp(beta_original[i]).
    or_val = ____
    name = feature_names[i]
    if or_val > 1:
        interp = f"1-unit increase -> {(or_val-1)*100:.1f}% higher odds"
    else:
        interp = f"1-unit increase -> {(1-or_val)*100:.1f}% lower odds"
    print(f"{name:<20} {beta_original[i]:>14.6f} {or_val:>12.4f} {interp}")

# Practical examples with real-world units
print(f"\n--- Practical Examples ---")
for feat, units, factor in [
    ("floor_area_sqm", "10 sqm", 10),
    ("storey_mid", "5 storeys", 5),
    ("remaining_lease", "10 years", 10),
]:
    idx = feature_names.index(feat)
    # TODO: Compute the odds change for a multi-unit increase.
    # Hint: np.exp(beta_original[idx] * factor).
    or_change = ____
    print(
        f"  +{units} of {feat}: odds multiply by {or_change:.3f} "
        f"({(or_change-1)*100:+.1f}% change)"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert np.exp(beta_original[1]) > 1, "Larger area should increase odds of high price"
print("\n[ok] Checkpoint 1 passed — odds ratios computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: threshold sweep with cost matrix
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Threshold Optimisation ===")

# Cost matrix for HDB property valuation
# FP (predict high, actually low): buyer overpays -> cost = $30K
# FN (predict low, actually high): seller underprices -> cost = $50K
cost_fp = 30_000
cost_fn = 50_000

thresholds = np.linspace(0.1, 0.9, 81)
total_costs = []
accuracies = []
f1_scores_list = []

for t in thresholds:
    y_pred_t = (p_scratch >= t).astype(int)
    # TODO: Compute the confusion matrix for this threshold.
    # Hint: confusion_matrix(y, y_pred_t) returns a 2x2 array.
    cm = ____
    tn, fp, fn, tp = cm.ravel()
    # TODO: Compute the total cost using the cost matrix.
    # Hint: cost = fp * cost_fp + fn * cost_fn.
    cost = ____
    total_costs.append(cost)
    accuracies.append(accuracy_score(y, y_pred_t))
    f1_scores_list.append(f1_score(y, y_pred_t, zero_division=0))

# TODO: Find the index of the cost-minimising threshold.
# Hint: np.argmin(total_costs).
optimal_idx = ____
optimal_threshold = thresholds[optimal_idx]
optimal_cost = total_costs[optimal_idx]

# F1-optimal threshold
f1_idx = np.argmax(f1_scores_list)
f1_threshold = thresholds[f1_idx]

print(f"Cost matrix: FP=${cost_fp:,}, FN=${cost_fn:,}")
print(f"\nOptimal threshold (min cost): {optimal_threshold:.3f}")
print(f"  Total cost: ${optimal_cost:,.0f}")
print(f"  Accuracy at this threshold: {accuracies[optimal_idx]:.4f}")
print(f"\nF1-optimal threshold: {f1_threshold:.3f}")
print(f"  F1 at this threshold: {f1_scores_list[f1_idx]:.4f}")
print(f"\nDefault threshold (0.5):")
print(f"  Cost: ${total_costs[40]:,.0f}")
print(f"  Accuracy: {accuracies[40]:.4f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert 0 < optimal_threshold < 1, "Optimal threshold must be valid"
assert optimal_cost <= total_costs[40], "Optimal cost must be <= default cost"
print("\n[ok] Checkpoint 2 passed — threshold optimisation completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: odds ratio forest plot + cost curve
# ════════════════════════════════════════════════════════════════════════

# Plot 1: Odds ratio forest plot
or_values = [np.exp(beta_original[i]) for i in range(1, len(feature_names))]
or_names = feature_names[1:]

fig1 = go.Figure()
fig1.add_trace(
    go.Bar(
        y=or_names,
        x=or_values,
        orientation="h",
        marker_color=["#2ecc71" if v > 1 else "#e74c3c" for v in or_values],
    )
)
fig1.add_vline(x=1.0, line_dash="dash", line_color="grey", annotation_text="No effect")
fig1.update_layout(
    title="Odds Ratios per Feature (original scale, per unit)",
    xaxis_title="Odds Ratio exp(b)",
    yaxis_title="Feature",
)
fig1.write_html(str(OUTPUT_DIR / "odds_ratios.html"))
print(f"Saved: {OUTPUT_DIR / 'odds_ratios.html'}")

# TODO: Create a cost + accuracy + F1 curve vs threshold.
# Hint: use go.Scatter for each series. Add vertical lines at
#       optimal_threshold and 0.5 using fig.add_vline().
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=thresholds,
        y=[c / 1e6 for c in total_costs],
        name="Total Cost ($M)",
    )
)
fig2.add_trace(
    go.Scatter(
        x=thresholds,
        y=accuracies,
        name="Accuracy",
        yaxis="y2",
        line={"dash": "dash"},
    )
)
fig2.add_trace(
    go.Scatter(
        x=thresholds,
        y=f1_scores_list,
        name="F1 Score",
        yaxis="y2",
        line={"dash": "dot"},
    )
)
fig2.add_vline(
    x=optimal_threshold,
    line_dash="dash",
    annotation_text=f"Cost-optimal t={optimal_threshold:.2f}",
)
fig2.add_vline(
    x=0.5,
    line_dash="dot",
    line_color="red",
    annotation_text="Default t=0.5",
)
fig2.update_layout(
    title="Cost, Accuracy, and F1 vs Classification Threshold",
    xaxis_title="Threshold",
    yaxis_title="Total Cost ($M)",
    yaxis2={"title": "Score", "overlaying": "y", "side": "right", "range": [0, 1]},
)
fig2.write_html(str(OUTPUT_DIR / "threshold_cost.html"))
print(f"Saved: {OUTPUT_DIR / 'threshold_cost.html'}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 3 passed — odds ratios + cost curve visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: HDB valuation risk — asymmetric error costs
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: The Singapore Land Authority (SLA) monitors HDB resale
# transactions for valuation anomalies.
#   - FP (flagged but normal): unnecessary review costs S$800/case
#   - FN (missed anomaly): average overpayment of S$45,000

y_pred_opt = (p_scratch >= optimal_threshold).astype(int)
cm_opt = confusion_matrix(y, y_pred_opt)
tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()

review_cost = 800
missed_cost = 45_000

annual_transactions = n_obs
# TODO: Compute annual FP cost and FN cost at the optimal threshold.
# Hint: annual cost = (count / n_obs) * annual_transactions * per-case cost.
annual_fp_cost = ____
annual_fn_cost = ____

# Compare with default threshold
y_pred_default = (p_scratch >= 0.5).astype(int)
cm_default = confusion_matrix(y, y_pred_default)
tn_d, fp_d, fn_d, tp_d = cm_default.ravel()
annual_fp_cost_d = int(annual_transactions * fp_d / n_obs) * review_cost
annual_fn_cost_d = int(annual_transactions * fn_d / n_obs) * missed_cost

print(f"\n=== Real-World Application: SLA Valuation Anomaly Detection ===")
print(f"  Annual transactions: {annual_transactions:,}")
print(f"\n  Default threshold (0.5):")
print(f"    Unnecessary reviews: {fp_d:,} -> S${annual_fp_cost_d:,.0f}")
print(f"    Missed anomalies:    {fn_d:,} -> S${annual_fn_cost_d:,.0f}")
print(f"    Total risk:          S${annual_fp_cost_d + annual_fn_cost_d:,.0f}")
print(f"\n  Cost-optimal threshold ({optimal_threshold:.3f}):")
print(f"    Unnecessary reviews: {fp_opt:,} -> S${annual_fp_cost:,.0f}")
print(f"    Missed anomalies:    {fn_opt:,} -> S${annual_fn_cost:,.0f}")
print(f"    Total risk:          S${annual_fp_cost + annual_fn_cost:,.0f}")
savings = (annual_fp_cost_d + annual_fn_cost_d) - (annual_fp_cost + annual_fn_cost)
print(f"    Annual saving:       S${savings:,.0f}")
