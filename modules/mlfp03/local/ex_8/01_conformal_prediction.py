# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 8.1: Conformal Prediction for Credit Default
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Train a calibrated LightGBM credit-default model end-to-end
#   - Apply split conformal prediction for distribution-free uncertainty
#   - Prove the 1-α coverage guarantee on held-out data
#   - Sweep α and read the cost of tighter coverage in set size
#   - Translate "ambiguous prediction set" into a business routing rule
#
# PREREQUISITES: MLFP03 Exercises 1-7, MLFP02 (preprocessing).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory     — why conformal works without distributional assumptions
#   2. Build      — train calibrated LightGBM + compute nonconformity scores
#   3. Train      — calibrate q̂, generate prediction sets, measure coverage
#   4. Visualise  — plot coverage + singleton rate across α values
#   5. Apply      — DBS Bank Singapore: which applicants go to human review
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from shared.mlfp03.ex_8 import (
    OUTPUT_DIR,
    evaluate_classification,
    load_credit_split,
    train_calibrated_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why conformal prediction works
# ════════════════════════════════════════════════════════════════════════
# Every classifier outputs a probability, but a probability is not a
# guarantee. Conformal prediction instead delivers a prediction SET C(x)
# with P(Y ∈ C(X)) ≥ 1 - α — guaranteed, distribution-free, from only
# the exchangeability assumption.
#
# ALGORITHM (split conformal, binary classification):
#   1. Split test into CALIBRATION and EVALUATION halves
#   2. Nonconformity score s_i = 1 - p(y_true | x_i) on calibration
#   3. q̂ = ceil((n+1)(1-α)) / n quantile of s_i
#   4. On new x: include class c iff 1 - p(c|x) ≤ q̂
#
# Singleton = confident auto-decide. Set of size 2 = ambiguous → human.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: train the calibrated baseline model
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 8.1 — Conformal Prediction")
print("=" * 70)

split = load_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]
feature_names = split["feature_names"]

print(f"\nData: train={X_train.shape}, test={X_test.shape}")
print(f"Default rate: {split['default_rate']:.1%}")

# TODO: Train the calibrated model by calling train_calibrated_model(X_train, y_train)
calibrated_model = ____

# TODO: Score the test set — use predict_proba(X_test)[:, 1] for the positive-class probability
y_proba = ____

# TODO: Compute the metric bundle with evaluate_classification(y_test, y_proba)
metrics = ____

print("\n=== Calibrated Model Metrics ===")
for k, v in metrics.items():
    print(f"  {k:<10} {v:.4f}")


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert metrics["auc_roc"] > 0.5, "Task 2: Should beat random"
assert (
    0 < metrics["brier"] < 0.25
), "Task 2: Brier should be reasonable for calibrated model"
print("\n[ok] Checkpoint 1 — calibrated LightGBM trained and evaluated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN the conformal calibrator
# ════════════════════════════════════════════════════════════════════════

n_cal = X_test.shape[0] // 2
X_cal, X_eval = X_test[:n_cal], X_test[n_cal:]
y_cal, y_eval = y_test[:n_cal], y_test[n_cal:]

cal_proba = calibrated_model.predict_proba(X_cal)[:, 1]
# TODO: Nonconformity score = 1 - p(TRUE class | x). Use np.where(y_cal == 1, 1 - cal_proba, cal_proba)
cal_scores = ____

alpha = 0.10  # target 90% coverage
n_cal_size = len(cal_scores)

# TODO: Compute quantile level ceil((n+1)(1-α)) / n, then q_hat = quantile of cal_scores at that level
quantile_level = ____
q_hat = float(np.quantile(cal_scores, min(quantile_level, 1.0)))

print(f"=== Conformal Calibration ===")
print(f"  Calibration set:   {n_cal_size} samples")
print(f"  Target coverage:   {1 - alpha:.0%}")
print(f"  Calibration q̂:     {q_hat:.4f}")

# Build prediction sets on evaluation data
eval_proba = calibrated_model.predict_proba(X_eval)[:, 1]
prediction_sets: list[set[int]] = []
for i in range(len(y_eval)):
    pset: set[int] = set()
    # TODO: Include class 1 iff (1 - eval_proba[i]) <= q_hat
    # TODO: Include class 0 iff eval_proba[i] <= q_hat
    # TODO: If empty, fall back to argmax class (1 if eval_proba[i] >= 0.5 else 0)
    ____
    prediction_sets.append(pset)

# TODO: Coverage = fraction of y_eval[i] in prediction_sets[i]
coverage = ____
avg_set_size = float(np.mean([len(ps) for ps in prediction_sets]))
singleton_rate = float(np.mean([len(ps) == 1 for ps in prediction_sets]))

print(f"\n=== Empirical Results on Evaluation Half ===")
print(f"  Coverage:          {coverage:.4f} (target: {1 - alpha:.2f})")
print(f"  Avg set size:      {avg_set_size:.3f}")
print(f"  Singleton rate:    {singleton_rate:.1%} (auto-decide)")
print(f"  Ambiguous rate:    {1 - singleton_rate:.1%} (route to human)")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert coverage >= (
    1 - alpha - 0.05
), f"Task 3: Coverage {coverage:.4f} should be near target {1 - alpha:.4f}"
assert 0 < avg_set_size <= 2, "Task 3: Set size should be between 0 and 2"
print("\n[ok] Checkpoint 2 — conformal coverage guarantee verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE coverage vs α
# ════════════════════════════════════════════════════════════════════════

print("=== Coverage vs α Sweep ===")
print(f"{'Alpha':>8} {'Target':>10} {'Actual':>10} {'Avg Size':>10} {'Singleton':>12}")
print("─" * 54)

alphas_sweep = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
sweep_rows = []
for a in alphas_sweep:
    q_level = np.ceil((n_cal_size + 1) * (1 - a)) / n_cal_size
    q = float(np.quantile(cal_scores, min(q_level, 1.0)))
    sets = []
    for i in range(len(y_eval)):
        ps: set[int] = set()
        if (1 - eval_proba[i]) <= q:
            ps.add(1)
        if eval_proba[i] <= q:
            ps.add(0)
        if not ps:
            ps.add(1 if eval_proba[i] >= 0.5 else 0)
        sets.append(ps)
    cov = float(np.mean([y_eval[i] in s for i, s in enumerate(sets)]))
    avg_sz = float(np.mean([len(s) for s in sets]))
    single = float(np.mean([len(s) == 1 for s in sets]))
    sweep_rows.append(
        {"alpha": a, "coverage": cov, "avg_size": avg_sz, "singleton": single}
    )
    print(f"{a:>8.2f} {1 - a:>10.2f} {cov:>10.4f} {avg_sz:>10.3f} {single:>11.1%}")

xs = [r["alpha"] for r in sweep_rows]
fig = go.Figure()
# TODO: Add a line trace for empirical coverage (x=xs, y=coverage values)
____
# TODO: Add a dashed reference trace for target 1-α (theoretical)
____
fig.add_trace(
    go.Bar(
        x=xs,
        y=[r["singleton"] for r in sweep_rows],
        name="Singleton rate (auto-decide)",
        marker_color="#10b981",
        opacity=0.55,
        yaxis="y2",
    )
)
fig.update_layout(
    title="Conformal Prediction: Coverage vs α (Singapore credit default)",
    xaxis_title="α (risk budget)",
    yaxis=dict(title="Coverage", range=[0, 1.05]),
    yaxis2=dict(title="Singleton rate", overlaying="y", side="right", range=[0, 1.05]),
    legend=dict(orientation="h", y=-0.2),
    height=500,
)
viz_path = OUTPUT_DIR / "ex8_01_conformal_sweep.html"
fig.write_html(str(viz_path))
print(f"\nSaved: {viz_path}")


# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert len(sweep_rows) == len(alphas_sweep), "Task 4: Should sweep all alphas"
print("\n[ok] Checkpoint 3 — α sweep visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Bank Singapore auto-decision routing
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS Bank's consumer credit team processes ~18,000 unsecured
# loan applications/month. MAS Notice 635 requires automated decisions
# be "explainable and contestable" — high-risk applications need human
# review.
#
# Conformal prediction gives DBS a MATHEMATICAL routing rule:
#   Prediction set = {0}   → auto-approve (90% confident: no default)
#   Prediction set = {1}   → auto-reject  (90% confident: default)
#   Prediction set = {0,1} → HUMAN REVIEW (model refuses to commit)

n_auto = int(round(singleton_rate * len(y_eval)))
n_human = len(y_eval) - n_auto
print(f"=== Routing decision at α=0.10 on {len(y_eval)} held-out applicants ===")
print(f"  Auto-decide (singleton):  {n_auto:>4}  ({singleton_rate:.1%})")
print(f"  Route to human review:    {n_human:>4}  ({1 - singleton_rate:.1%})")

# DOLLAR IMPACT
monthly_apps = 18_000
analyst_hourly = 85.0
time_per_review_h = 0.25
ambiguous_monthly = monthly_apps * (1 - singleton_rate)
status_quo_cost = monthly_apps * time_per_review_h * analyst_hourly
conformal_cost = ambiguous_monthly * time_per_review_h * analyst_hourly
savings = status_quo_cost - conformal_cost
print(f"\n  Analyst cost (status quo, 100% review): S${status_quo_cost:>10,.0f}/mo")
print(f"  Analyst cost (conformal routing):       S${conformal_cost:>10,.0f}/mo")
print(f"  Monthly savings:                         S${savings:>10,.0f}/mo")
print(f"  Annualised:                              S${savings * 12:>10,.0f}/yr")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Trained a calibrated LightGBM credit-default model
  [x] Built split conformal prediction from nonconformity scores
  [x] Verified {coverage:.0%} empirical coverage at α={alpha}
  [x] Swept α and visualised the cost of tighter coverage
  [x] Translated prediction sets into DBS routing with S${savings * 12:,.0f}/yr savings

  KEY INSIGHT: A probability is a guess. A prediction set is a contract.

  Next: 02_drift_monitoring.py — what happens when exchangeability breaks.
"""
)
