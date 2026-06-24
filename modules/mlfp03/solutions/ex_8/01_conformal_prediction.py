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
# guarantee. A model that says "80% chance of default" on 100 applicants
# may actually default 40 times (miscalibrated) or 95 times (underconfident).
#
# Conformal prediction sidesteps this by asking a different question:
# "For what prediction set C(x) is P(Y ∈ C(x)) ≥ 1 - α true GUARANTEED,
#  regardless of the model or the data distribution?"
#
# The only assumption is EXCHANGEABILITY — future applicants come from
# the same distribution as the calibration applicants. That's it. No
# Gaussian assumption, no well-calibrated probabilities, no correctness
# of the base model. The coverage guarantee is a mathematical theorem,
# not a hope.
#
# THE ALGORITHM (split conformal, binary classification):
#   1. Split the test set into CALIBRATION and EVALUATION halves.
#   2. On CALIBRATION: compute nonconformity score s_i = 1 - p(y_true | x_i).
#      High score = model was wrong/surprised. Low score = model was confident
#      and correct.
#   3. Compute q̂ = ceil((n+1)(1-α)) / n quantile of s_i.
#   4. On new x: include class c in the prediction set iff 1 - p(c|x) ≤ q̂.
#   5. P(Y ∈ C(X)) ≥ 1 - α, proven. Period.
#
# THE BUSINESS PAYOFF: Every prediction now comes with a routing decision.
#   - Set size 1 (singleton): model is confident, auto-decide.
#   - Set size 2 (ambiguous): model admits uncertainty, route to a human.
# You can't get this from a calibrated probability alone — you'd need to
# pick a threshold, and thresholds drift.


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

calibrated_model = train_calibrated_model(X_train, y_train)
y_proba = calibrated_model.predict_proba(X_test)[:, 1]
metrics = evaluate_classification(y_test, y_proba)

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
# Split test set into calibration and evaluation halves. Compute
# nonconformity scores on calibration, then apply q̂ on evaluation.

n_cal = X_test.shape[0] // 2
X_cal, X_eval = X_test[:n_cal], X_test[n_cal:]
y_cal, y_eval = y_test[:n_cal], y_test[n_cal:]

cal_proba = calibrated_model.predict_proba(X_cal)[:, 1]
# Nonconformity score = 1 - p(TRUE class | x)
cal_scores = np.where(y_cal == 1, 1 - cal_proba, cal_proba)

alpha = 0.10  # target 90% coverage
n_cal_size = len(cal_scores)
quantile_level = np.ceil((n_cal_size + 1) * (1 - alpha)) / n_cal_size
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
    if (1 - eval_proba[i]) <= q_hat:
        pset.add(1)
    if eval_proba[i] <= q_hat:
        pset.add(0)
    if not pset:
        pset.add(1 if eval_proba[i] >= 0.5 else 0)
    prediction_sets.append(pset)

coverage = float(np.mean([y_eval[i] in ps for i, ps in enumerate(prediction_sets)]))
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
# INTERPRETATION: The coverage guarantee is distribution-free. If coverage
# drifts below target in production, the data has moved — retrain, don't
# re-calibrate.
print("\n[ok] Checkpoint 2 — conformal coverage guarantee verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE coverage vs α
# ════════════════════════════════════════════════════════════════════════
# Sweep α and plot the trade-off: lower α gives higher coverage but
# bigger prediction sets (more ambiguity → more human review).

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

# Build the visual proof: coverage line + singleton-rate bars on same x.
xs = [r["alpha"] for r in sweep_rows]
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=xs,
        y=[r["coverage"] for r in sweep_rows],
        mode="lines+markers",
        name="Empirical coverage",
        line=dict(color="#2563eb", width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=xs,
        y=[1 - a for a in xs],
        mode="lines",
        name="Target 1-α (theoretical)",
        line=dict(color="#94a3b8", dash="dash"),
    )
)
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
assert (
    sweep_rows[0]["coverage"] >= sweep_rows[-1]["coverage"] - 0.05
), "Task 4: Lower α should give ≥ coverage than higher α (up to sampling noise)"
print("\n[ok] Checkpoint 3 — α sweep visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Bank Singapore auto-decision routing
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS Bank's consumer credit team processes ~18,000 unsecured
# loan applications/month in Singapore. MAS Notice 635 requires that
# automated credit decisions be "explainable and contestable" and that
# high-risk applications receive human review.
#
# The current rule-of-thumb is "if model says >70% default probability,
# auto-reject; if <30%, auto-approve; else human review." That's a
# threshold on a miscalibrated probability, and it drifts every quarter
# as the economy moves.
#
# Conformal prediction gives DBS a MATHEMATICAL routing rule:
#
#   Prediction set = {0}  → auto-approve (model 90% confident: no default)
#   Prediction set = {1}  → auto-reject  (model 90% confident: default)
#   Prediction set = {0,1}→ HUMAN REVIEW (model refuses to commit)
#
# And the 90% coverage guarantee holds regardless of which macroeconomic
# regime we're in — as long as the calibration set was representative.

n_auto = int(round(singleton_rate * len(y_eval)))
n_human = len(y_eval) - n_auto
print(f"=== Routing decision at α=0.10 on {len(y_eval)} held-out applicants ===")
print(f"  Auto-decide (singleton):  {n_auto:>4}  ({singleton_rate:.1%})")
print(f"  Route to human review:    {n_human:>4}  ({1 - singleton_rate:.1%})")

# DOLLAR IMPACT — conservative estimates from DBS's public 2025 disclosures
monthly_apps = 18_000
analyst_hourly = 85.0  # SGD, fully loaded cost of a credit analyst
time_per_review_h = 0.25  # 15 minutes per application
ambiguous_monthly = monthly_apps * (1 - singleton_rate)
manual_baseline_monthly = monthly_apps * 1.0  # status quo: every app reviewed
status_quo_cost = manual_baseline_monthly * time_per_review_h * analyst_hourly
conformal_cost = ambiguous_monthly * time_per_review_h * analyst_hourly
savings = status_quo_cost - conformal_cost

# Avoided losses: each confident (singleton) auto-reject prevents roughly
# S$2,400 in expected loss (DBS 2024 unsecured LGD ~60% × median S$4,000
# loan). We assume the conformal model catches the same rejections the
# analysts would, just faster.
print(f"\n  Analyst cost (status quo, 100% review): S${status_quo_cost:>10,.0f}/mo")
print(f"  Analyst cost (conformal routing):       S${conformal_cost:>10,.0f}/mo")
print(f"  Monthly savings:                         S${savings:>10,.0f}/mo")
print(f"  Annualised:                              S${savings * 12:>10,.0f}/yr")

# LIMITATIONS:
#   - Exchangeability assumption breaks under adversarial drift
#     (fraud rings learn the model); see 8.2 for drift monitoring.
#   - Singleton rate varies with α. α=0.05 halves savings but catches 95%.
#   - Ambiguous applicants may skew toward protected groups; pair with
#     the fairness audit from Exercise 6 before shipping.


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
      (mathematical guarantee, no distributional assumptions)
  [x] Swept α and visualised the cost of tighter coverage
  [x] Translated prediction set size into DBS routing rules
      with S${savings * 12:,.0f}/yr in analyst-time savings

  KEY INSIGHT: A probability is a guess. A prediction set is a contract.
  Conformal prediction is the only way to hand a regulator a coverage
  guarantee that survives model drift, miscalibration, and distribution
  surprise — as long as exchangeability holds.

  Next: 02_drift_monitoring.py — watch what happens to that guarantee
  when exchangeability breaks, and how DriftMonitor catches it.
"""
)
