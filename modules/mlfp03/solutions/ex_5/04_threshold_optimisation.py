# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.4: Threshold Optimisation from a Cost Matrix
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why threshold=0.5 is almost always wrong for asymmetric costs
#   - The Bayes-optimal threshold formula t* = cost_FP / (cost_FP + cost_FN)
#   - How to sweep the threshold empirically and find the cost minimum
#   - How to translate the chosen threshold into annual S$ savings
#
# PREREQUISITES: 02_sampling_strategies.py (cost-sensitive proba saved)
# ESTIMATED TIME: ~25 min
#
# 5-PHASE STRUCTURE:
#   Theory   — decision theory for asymmetric costs
#   Build    — load saved cost-sensitive probabilities, build threshold grid
#   Train    — (no training — we TUNE a decision rule on fixed probabilities)
#   Visualise — cost vs threshold curve + confusion matrix at best threshold
#   Apply    — Maybank Singapore unsecured-loan underwriting ROI
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from dotenv import load_dotenv

from shared.mlfp03.ex_5 import (
    ANNUAL_APPLICATIONS,
    DEFAULT_COSTS,
    OUTPUT_DIR,
    annual_roi,
    load_credit_splits,
    load_strategy_proba,
    metrics_row,
    print_metrics_table,
    print_roi,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Decision theory for asymmetric cost matrices
# ════════════════════════════════════════════════════════════════════════
# Imagine your model outputs a probability p that an applicant will
# default. You must DECIDE: approve or decline. The default rule is
# "decline if p >= 0.5". Where does 0.5 come from?
#
# It comes from an implicit assumption: the cost of a false positive
# equals the cost of a false negative. In credit scoring, FP costs
# ~S$100 (manual review) and FN costs ~S$10,000 (charge-off). The
# ratio is 100:1. The 0.5 threshold is wrong by two orders of
# magnitude.
#
# THE BAYES-OPTIMAL THRESHOLD:
#
#     t* = cost_FP / (cost_FP + cost_FN)
#
# Expected cost at threshold t:
#     E[cost] = P(y=0) * P(pred=1 | y=0) * cost_FP
#             + P(y=1) * P(pred=0 | y=1) * cost_FN
#
# Minimising over t gives t* above. For our 100:1 cost matrix,
# t* = 100 / 10100 = 0.0099 — roughly 1%. Any applicant with >=1%
# default probability should be declined.
#
# CAVEAT: the theoretical formula assumes the probabilities are
# well-calibrated. If the model outputs scores that are NOT true
# probabilities (most gradient boosters), you should ALSO sweep
# the threshold empirically on a validation set and pick the
# actual argmin of total cost. Exercise 5.5 will fix the calibration
# so the theoretical formula works; this file uses the empirical
# sweep so it's robust to miscalibrated inputs.


# ════════════════════════════════════════════════════════════════════════
# BUILD — load saved probabilities, sweep thresholds
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()

try:
    y_proba = load_strategy_proba("cost_sensitive_scale")
except FileNotFoundError as e:
    raise RuntimeError(
        "Run 02_sampling_strategies.py first — it saves the probabilities."
    ) from e

print("\n" + "=" * 70)
print("  Exercise 5.4 — Threshold Optimisation")
print("=" * 70)
print(f"  Cost matrix: FP=${DEFAULT_COSTS.fp:,.0f}, FN=${DEFAULT_COSTS.fn:,.0f}")
print(f"  Bayes-optimal theoretical threshold: {DEFAULT_COSTS.optimal_threshold:.4f}")


# ════════════════════════════════════════════════════════════════════════
# TRAIN (no training — we TUNE a decision rule on fixed probabilities)
# ════════════════════════════════════════════════════════════════════════

thresholds = np.arange(0.005, 0.500, 0.005)
sweep_rows: list[dict] = []
for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    cost = fp * DEFAULT_COSTS.fp + fn * DEFAULT_COSTS.fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sweep_rows.append(
        {
            "threshold": float(t),
            "total_cost_usd": float(cost),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
        }
    )

sweep_df = pl.DataFrame(sweep_rows)
best_row = min(sweep_rows, key=lambda r: r["total_cost_usd"])
best_t = best_row["threshold"]
default_cost = (
    next(r["total_cost_usd"] for r in sweep_rows if abs(r["threshold"] - 0.5) < 1e-6)
    if any(abs(r["threshold"] - 0.5) < 1e-6 for r in sweep_rows)
    else (((y_proba >= 0.5).astype(int) != y_test).sum() * DEFAULT_COSTS.fn)
)
# Always compute the t=0.5 reference cost directly for robustness
y_pred_default = (y_proba >= 0.5).astype(int)
fp_d = int(((y_pred_default == 1) & (y_test == 0)).sum())
fn_d = int(((y_pred_default == 0) & (y_test == 1)).sum())
default_cost = fp_d * DEFAULT_COSTS.fp + fn_d * DEFAULT_COSTS.fn


# ── Checkpoint 4 ────────────────────────────────────────────────────────
assert 0.0 < best_t < 1.0, "Best threshold must be in (0,1)"
assert (
    best_row["total_cost_usd"] <= default_cost
), "Optimised threshold should not cost more than t=0.5"
print("[ok] Checkpoint 4 — threshold sweep and argmin computed\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — cost curve + confusion matrix at best threshold
# ════════════════════════════════════════════════════════════════════════

print(
    f"\n  {'threshold':>10} {'total_cost':>14} {'precision':>10} "
    f"{'recall':>8} {'fp':>6} {'fn':>6}"
)
print("  " + "─" * 60)
for r in sweep_rows[::10]:
    print(
        f"  {r['threshold']:>10.3f} ${r['total_cost_usd']:>12,.0f} "
        f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['fp']:>6} {r['fn']:>6}"
    )

print(f"\n  Cost at t=0.5:       ${default_cost:,.0f}")
print(f"  Cost at t*={best_t:.3f}: ${best_row['total_cost_usd']:,.0f}")
savings = default_cost - best_row["total_cost_usd"]
saving_pct = savings / max(default_cost, 1)
print(f"  Savings from threshold tuning: ${savings:,.0f} ({saving_pct:.1%})")
print(
    f"  Theoretical t* (Bayes): {DEFAULT_COSTS.optimal_threshold:.4f} — "
    f"empirical: {best_t:.4f}"
)

sweep_df.write_parquet(OUTPUT_DIR / "threshold_sweep.parquet")
print(f"\n  Saved: {OUTPUT_DIR / 'threshold_sweep.parquet'}")

# ── Visual: Threshold vs metrics curve ───────────────────────────────────
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[r["threshold"] for r in sweep_rows],
        y=[r["total_cost_usd"] for r in sweep_rows],
        mode="lines",
        name="Total cost ($)",
        line=dict(color="#ef4444", width=3),
    )
)
fig.add_vline(
    x=best_t, line_dash="dash", line_color="#10b981", annotation_text=f"t*={best_t:.3f}"
)
fig.add_vline(
    x=0.5, line_dash="dot", line_color="#6b7280", annotation_text="t=0.5 (default)"
)
fig.update_layout(
    title="Threshold vs Total Cost: the U-shaped curve (lower = better)",
    xaxis_title="Decision threshold",
    yaxis_title="Total cost ($)",
    height=450,
)
viz_path = OUTPUT_DIR / "ex5_04_threshold_cost_curve.html"
fig.write_html(str(viz_path))
print(f"  Saved: {viz_path}")

# ── Visual: Precision-Recall vs threshold ────────────────────────────────
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=[r["threshold"] for r in sweep_rows],
        y=[r["precision"] for r in sweep_rows],
        mode="lines",
        name="Precision",
        line=dict(color="#6366f1", width=2),
    )
)
fig2.add_trace(
    go.Scatter(
        x=[r["threshold"] for r in sweep_rows],
        y=[r["recall"] for r in sweep_rows],
        mode="lines",
        name="Recall",
        line=dict(color="#f59e0b", width=2),
    )
)
fig2.add_vline(
    x=best_t, line_dash="dash", line_color="#10b981", annotation_text=f"t*={best_t:.3f}"
)
fig2.update_layout(
    title="Precision and Recall vs Threshold (credit default)",
    xaxis_title="Decision threshold",
    yaxis_title="Score",
    height=450,
    legend=dict(orientation="h", y=-0.2),
)
viz_path2 = OUTPUT_DIR / "ex5_04_threshold_pr_curve.html"
fig2.write_html(str(viz_path2))
print(f"  Saved: {viz_path2}")

# ── Visual: Cost matrix heatmap ──────────────────────────────────────────
cost_matrix = np.array([[0, DEFAULT_COSTS.fp], [DEFAULT_COSTS.fn, 0]])
fig3 = go.Figure(
    data=go.Heatmap(
        z=cost_matrix,
        x=["Predicted Negative", "Predicted Positive"],
        y=["Actual Negative", "Actual Positive"],
        text=[
            [f"TN: $0", f"FP: ${DEFAULT_COSTS.fp:,.0f}"],
            [f"FN: ${DEFAULT_COSTS.fn:,.0f}", f"TP: $0"],
        ],
        texttemplate="%{text}",
        colorscale="Reds",
        showscale=True,
    )
)
fig3.update_layout(
    title="Cost Matrix: Asymmetric penalties (FN >> FP)",
    height=400,
)
viz_path3 = OUTPUT_DIR / "ex5_04_cost_matrix_heatmap.html"
fig3.write_html(str(viz_path3))
print(f"  Saved: {viz_path3}")

# Per-strategy metric table at the optimised threshold
row = metrics_row("Cost-sens @ t*", y_test, y_proba, threshold=best_t)
print_metrics_table([row], f"Metrics at optimised threshold t*={best_t:.3f}")

# INTERPRETATION: The cost curve is U-shaped. At t=0.5 the model misses
# most defaults (FN term dominates). At t=0.01 the model flags everyone
# (FP term dominates). The minimum is far below 0.5 because the cost
# matrix is asymmetric.


# ════════════════════════════════════════════════════════════════════════
# APPLY — Maybank Singapore unsecured-loan underwriting ROI
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Maybank Singapore underwrites ~100,000 unsecured personal
# loans per year. Today's rule: decline if scorecard p >= 0.5. The
# underwriting committee asks: "What would it save if we tuned the
# threshold from the cost matrix?"
#
# We can answer that question NOW by scaling the test-set confusion
# matrix to the annual application volume.

roi_default = annual_roi(
    y_test, y_proba, threshold=0.5, annual_volume=ANNUAL_APPLICATIONS
)
roi_best = annual_roi(
    y_test, y_proba, threshold=best_t, annual_volume=ANNUAL_APPLICATIONS
)

print_roi("Annual ROI @ t=0.5", roi_default)
print_roi(f"Annual ROI @ t*={best_t:.3f}", roi_best)

delta = roi_best["annual_savings_usd"] - roi_default["annual_savings_usd"]
print(f"\n  Threshold tuning alone adds: ${delta:,.0f}/year in savings")
print("    No retraining. No new data. No new features. Just changing a")
print("    number in the decision layer. This is usually the highest-ROI")
print("    lever on any imbalanced production model.")

# Persist the annual ROI snapshot for the final comparison in 05
pl.DataFrame(
    [roi_default | {"label": "t=0.5"}, roi_best | {"label": "t*"}]
).write_parquet(OUTPUT_DIR / "threshold_roi.parquet")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.4")
print("=" * 70)
print(
    """
  [x] Derived the Bayes-optimal threshold t* = cost_FP/(cost_FP+cost_FN)
  [x] Ran an empirical sweep to find the argmin of total cost
  [x] Observed the U-shape of the cost-vs-threshold curve
  [x] Scaled the test confusion matrix to annual application volume
  [x] Quantified the dollar savings of threshold tuning alone

  KEY INSIGHT: Threshold tuning is the highest-ROI lever on any
  imbalanced production model. It's free (no retraining) and can add
  seven figures of annual savings on its own.

  Next: 05_calibration.py — Platt and Isotonic calibration make the
  theoretical t* match the empirical t*, and compare every strategy.
"""
)
