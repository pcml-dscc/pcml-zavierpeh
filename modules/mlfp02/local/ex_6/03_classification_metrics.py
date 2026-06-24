# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 6.3: Classification Metrics — Confusion Matrix,
#                         ROC, and Precision-Recall
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compute and interpret the confusion matrix (TP, FP, TN, FN)
#   - Derive precision, recall, F1, and accuracy from the matrix
#   - Plot and interpret ROC curves + compute AUC
#   - Plot precision-recall curves for imbalanced-class awareness
#   - Apply classification metrics to Singapore mortgage risk screening
#
# PREREQUISITES: Exercise 6.1 (logistic regression), 6.2 (thresholds)
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — the four quadrants of classification
#   2. Build — confusion matrix + derived metrics
#   3. Train — ROC curve + precision-recall curve
#   4. Visualise — ROC, PR, and annotated confusion matrix
#   5. Apply — DBS mortgage pre-screening (S$ impact)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from shared.mlfp02.ex_6 import (
    FEATURE_COLS,
    OUTPUT_DIR,
    build_classification_frame,
    build_design_matrix,
    load_hdb_recent,
    neg_ll_gradient,
    neg_log_likelihood_logistic,
    sigmoid,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Four Quadrants of Classification
# ════════════════════════════════════════════════════════════════════════
# Every binary classifier maps each observation into one of four cells:
#
#                      Predicted +    Predicted -
#   Actual Positive        TP             FN
#   Actual Negative        FP             TN
#
# From these four counts:
#   Accuracy  = (TP + TN) / N           — overall correctness
#   Precision = TP / (TP + FP)          — "of flagged, how many right?"
#   Recall    = TP / (TP + FN)          — "of actual pos, how many caught?"
#   F1        = 2 * Prec * Rec / (P+R)  — harmonic mean
#
# ROC curve plots TPR vs FPR across all thresholds. AUC measures
# discrimination ability. PR curve is more informative when the
# positive class is rare.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: fit model and compute confusion matrix
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Classification Metrics — Confusion, ROC, Precision-Recall")
print("=" * 70)

# Load data and fit logistic regression
hdb_recent = load_hdb_recent()
frame, median_price = build_classification_frame(hdb_recent)
X, y, X_mean, X_std, feature_names = build_design_matrix(frame)
n_obs = X.shape[0]
n_positive = int(y.sum())

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

# Use a cost-optimal threshold (FN more costly than FP)
cost_fp, cost_fn = 30_000, 50_000
thresholds_sweep = np.linspace(0.1, 0.9, 81)
costs = []
for t in thresholds_sweep:
    cm_t = confusion_matrix(y, (p_scratch >= t).astype(int))
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    costs.append(fp_t * cost_fp + fn_t * cost_fn)
optimal_threshold = thresholds_sweep[np.argmin(costs)]

# TODO: Compute predictions at the optimal threshold.
# Hint: (p_scratch >= optimal_threshold).astype(int).
y_pred_opt = ____

# TODO: Compute the confusion matrix at the optimal threshold.
# Hint: confusion_matrix(y, y_pred_opt).
cm = ____
tn, fp, fn, tp = cm.ravel()

# TODO: Compute precision, recall, F1, and accuracy.
# Hint: precision_score(y, y_pred_opt), recall_score(y, y_pred_opt),
#       f1_score(y, y_pred_opt), accuracy_score(y, y_pred_opt).
prec = ____
rec = ____
f1 = ____
acc = ____

print(f"\n=== Classification Metrics (threshold={optimal_threshold:.3f}) ===")
print(f"\nConfusion Matrix:")
print(f"              Predicted Low  Predicted High")
print(f"  Actual Low    {tn:>10,}    {fp:>10,}")
print(f"  Actual High   {fn:>10,}    {tp:>10,}")
print(f"\n{'Metric':<15} {'Value':>10}")
print("-" * 28)
print(f"{'Accuracy':<15} {acc:>10.4f}")
print(f"{'Precision':<15} {prec:>10.4f}")
print(f"{'Recall':<15} {rec:>10.4f}")
print(f"{'F1 Score':<15} {f1:>10.4f}")
print(f"{'True Positives':<15} {tp:>10,}")
print(f"{'False Positives':<15} {fp:>10,}")
print(f"{'True Negatives':<15} {tn:>10,}")
print(f"{'False Negatives':<15} {fn:>10,}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert tp + fp + tn + fn == n_obs, "Confusion matrix must sum to n"
assert 0 < prec <= 1, "Precision must be valid"
assert 0 < rec <= 1, "Recall must be valid"
print("\n[ok] Checkpoint 1 passed — confusion matrix + metrics computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: ROC curve + precision-recall curve
# ════════════════════════════════════════════════════════════════════════

# TODO: Compute the ROC curve.
# Hint: roc_curve(y, p_scratch) returns (fpr, tpr, roc_thresholds).
fpr, tpr, roc_thresholds = ____

# TODO: Compute AUC from the ROC curve.
# Hint: auc(fpr, tpr).
roc_auc = ____

# Find threshold closest to top-left corner (0, 1)
distances = np.sqrt((0 - fpr) ** 2 + (1 - tpr) ** 2)
roc_optimal_idx = np.argmin(distances)
roc_optimal_threshold = roc_thresholds[roc_optimal_idx]

print(f"\n=== ROC Curve ===")
print(f"AUC = {roc_auc:.4f}")
print(f"ROC-optimal threshold (closest to top-left): {roc_optimal_threshold:.4f}")
print(f"  at FPR={fpr[roc_optimal_idx]:.4f}, TPR={tpr[roc_optimal_idx]:.4f}")
print(f"\nAUC interpretation:")
if roc_auc > 0.9:
    print(f"  Excellent discrimination")
elif roc_auc > 0.8:
    print(f"  Good discrimination")
elif roc_auc > 0.7:
    print(f"  Fair discrimination")
else:
    print(f"  Poor discrimination")

# TODO: Compute the precision-recall curve.
# Hint: precision_recall_curve(y, p_scratch) returns (prec_curve, rec_curve, pr_thresholds).
prec_curve, rec_curve, pr_thresholds = ____

# TODO: Compute PR-AUC.
# Hint: auc(rec_curve, prec_curve) — note the argument order.
pr_auc = ____

print(f"\n=== Precision-Recall Curve ===")
print(f"PR-AUC = {pr_auc:.4f}")
print(f"Baseline (random): {n_positive/n_obs:.4f}")
print(f"PR-AUC / baseline: {pr_auc / (n_positive/n_obs):.2f}x better than random")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert 0.5 <= roc_auc <= 1.0, "AUC must be between 0.5 and 1.0"
assert pr_auc > n_positive / n_obs, "PR-AUC should beat random baseline"
print("\n[ok] Checkpoint 2 passed — ROC + PR curves computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: ROC, PR, and annotated confusion matrix
# ════════════════════════════════════════════════════════════════════════

# Plot 1: ROC curve
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={roc_auc:.3f})"))
fig1.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name="Random",
        line={"dash": "dash", "color": "grey"},
    )
)
fig1.add_trace(
    go.Scatter(
        x=[fpr[roc_optimal_idx]],
        y=[tpr[roc_optimal_idx]],
        mode="markers",
        name=f"Optimal (t={roc_optimal_threshold:.3f})",
        marker={"size": 12, "color": "red"},
    )
)
fig1.update_layout(
    title="ROC Curve — HDB Resale Classification",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
)
fig1.write_html(str(OUTPUT_DIR / "roc_curve.html"))
print(f"Saved: {OUTPUT_DIR / 'roc_curve.html'}")

# Plot 2: Precision-recall curve
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=rec_curve,
        y=prec_curve,
        name=f"PR (AUC={pr_auc:.3f})",
    )
)
fig2.add_hline(
    y=n_positive / n_obs,
    line_dash="dash",
    line_color="grey",
    annotation_text=f"Random baseline ({n_positive/n_obs:.2f})",
)
fig2.update_layout(
    title="Precision-Recall Curve — HDB Resale Classification",
    xaxis_title="Recall",
    yaxis_title="Precision",
)
fig2.write_html(str(OUTPUT_DIR / "precision_recall.html"))
print(f"Saved: {OUTPUT_DIR / 'precision_recall.html'}")

# TODO: Create an annotated confusion matrix heatmap.
# Hint: use go.Heatmap with z=cm, text annotations, colorscale="Blues".
cm_labels = [["TN", "FP"], ["FN", "TP"]]
cm_text = [[f"{cm_labels[r][c]}\n{cm[r,c]:,}" for c in range(2)] for r in range(2)]

fig3 = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=["Predicted Low", "Predicted High"],
        y=["Actual Low", "Actual High"],
        text=cm_text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
    )
)
fig3.update_layout(
    title=f"Confusion Matrix (t={optimal_threshold:.3f})",
    xaxis_title="Predicted",
    yaxis_title="Actual",
)
fig3.write_html(str(OUTPUT_DIR / "confusion_matrix.html"))
print(f"Saved: {OUTPUT_DIR / 'confusion_matrix.html'}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 3 passed — ROC, PR, confusion matrix visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS mortgage pre-screening
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS Bank pre-screens HDB mortgage applications.
# High-value applications go to a senior relationship manager (RM).
#   - FP (standard routed to RM): wasted time, S$200/case
#   - FN (high-value goes auto): lost cross-sell, S$8,000/case

rm_cost = 200
cross_sell_loss = 8_000
annual_apps = 25_000

fn_rate = fn / n_obs
fp_rate = fp / n_obs
# TODO: Compute annual costs from FP and FN rates.
# Hint: annual_rm_waste = int(annual_apps * fp_rate) * rm_cost.
annual_rm_waste = ____
annual_cross_sell_loss = ____

# Compare with default threshold
y_pred_05 = (p_scratch >= 0.5).astype(int)
cm_05 = confusion_matrix(y, y_pred_05)
tn_05, fp_05, fn_05, tp_05 = cm_05.ravel()
annual_rm_waste_05 = int(annual_apps * fp_05 / n_obs) * rm_cost
annual_cross_sell_05 = int(annual_apps * fn_05 / n_obs) * cross_sell_loss

print(f"\n=== Real-World Application: DBS Mortgage Pre-Screening ===")
print(f"  Annual mortgage applications: {annual_apps:,}")
print(f"\n  Default threshold (0.5):")
print(f"    Unnecessary RM routing:    S${annual_rm_waste_05:,.0f}")
print(f"    Lost cross-sell revenue:   S${annual_cross_sell_05:,.0f}")
print(
    f"    Total misclassification cost: S${annual_rm_waste_05 + annual_cross_sell_05:,.0f}"
)
print(f"\n  Cost-optimal threshold ({optimal_threshold:.3f}):")
print(f"    Unnecessary RM routing:    S${annual_rm_waste:,.0f}")
print(f"    Lost cross-sell revenue:   S${annual_cross_sell_loss:,.0f}")
print(
    f"    Total misclassification cost: S${annual_rm_waste + annual_cross_sell_loss:,.0f}"
)
savings = (annual_rm_waste_05 + annual_cross_sell_05) - (
    annual_rm_waste + annual_cross_sell_loss
)
print(f"    Annual saving:             S${savings:,.0f}")
print(f"\n  Model discrimination (AUC): {roc_auc:.4f}")
