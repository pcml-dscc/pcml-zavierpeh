# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.2: XGBoost on Singapore Credit Data
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Train an XGBoost classifier on a real imbalanced credit dataset
#   - Connect XGBoost hyperparameters back to the theory in 4.1
#   - Use AUC-PR as the primary metric for 12%-positive data
#   - Extract and rank gain-based feature importances
#   - Explain why XGBoost is the default choice for tabular data
#
# PREREQUISITES: Exercise 4.1 (boosting theory, split-gain formula).
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — hyperparameters as theory dials
#   2. Build — XGBoost with course-standard defaults
#   3. Train — fit, time the training, evaluate
#   4. Visualise — feature importance bar chart
#   5. Apply — DBS credit risk team explainability for MAS FEAT
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import plotly.graph_objects as go
from dotenv import load_dotenv

from shared.mlfp03.ex_4 import (
    OUTPUT_DIR,
    evaluate_classifier,
    make_xgboost,
    prepare_credit_split,
    print_metrics,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — XGBoost hyperparameters as theory dials
# ════════════════════════════════════════════════════════════════════════
# learning_rate (η)  → additive-model step size
# max_depth          → tree size / interaction order
# n_estimators       → number of boosting rounds
# reg_lambda (λ)     → L2 on leaf weights (from split-gain formula)
# gamma (γ)          → minimum gain to accept a split
# subsample          → stochastic row sampling per tree
# colsample_bytree   → stochastic column sampling per tree


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the XGBoost classifier
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  XGBoost on Singapore Credit Scoring")
print("=" * 70)

data = prepare_credit_split()
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
feature_names = data["feature_names"]

print(f"\n  Train: {X_train.shape} | Test: {X_test.shape}")
print(f"  Features: {len(feature_names)}")
print(f"  Default rate: {data['default_rate']:.2%}")

# TODO: Build an XGBoost classifier via make_xgboost with:
# n_estimators=500, learning_rate=0.1, max_depth=6
model = ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN on Singapore credit data
# ════════════════════════════════════════════════════════════════════════

print("\n  Training XGBoost (500 rounds, η=0.1, depth=6)...")
t0 = time.perf_counter()

# TODO: Fit the model using X_train, y_train.
# Pass eval_set=[(X_test, y_test)] and verbose=False.
____

train_time = time.perf_counter() - t0

# TODO: Predict class-1 probabilities for X_test.
# Hint: model.predict_proba returns an (n, 2) array — take column [:, 1].
y_proba = ____

# TODO: Call evaluate_classifier(y_test, y_proba) to get the metric bundle.
metrics = ____

print_metrics("XGBoost", metrics, train_time=train_time)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert metrics["auc_roc"] > 0.7, "XGBoost should beat 0.7 AUC-ROC on credit data"
assert metrics["auc_pr"] > 0.3, "XGBoost AUC-PR should clear 0.3 (12% base rate)"
print("\n[ok] Checkpoint 1 passed — XGBoost trained and evaluated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE feature importance
# ════════════════════════════════════════════════════════════════════════
# XGBoost uses gain-based importance by default: the total reduction in
# loss attributed to each feature via its split-gain contributions.

# TODO: Pull the importance vector from the trained model.
# Hint: model.feature_importances_
importances = ____

# TODO: Sort zip(feature_names, importances) by importance descending,
# take top 15.
ranked = ____
top_15 = ranked[:15]

print("  --- Top-15 Features by XGBoost Gain Importance ---")
print(f"  {'Rank':>4}  {'Feature':<30}  {'Gain':>10}")
print("  " + "─" * 50)
for rank, (name, importance) in enumerate(top_15, start=1):
    print(f"  {rank:>4}  {name:<30}  {importance:>10.4f}")

names = [name for name, _ in top_15][::-1]
values = [float(v) for _, v in top_15][::-1]

fig = go.Figure(
    go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(color=values, colorscale="Blues"),
    )
)
fig.update_layout(
    title="XGBoost Feature Importance — Singapore Credit Default",
    xaxis_title="Gain",
    height=520,
)
viz_path = OUTPUT_DIR / "ex4_02_xgboost_feature_importance.html"
fig.write_html(viz_path)
print(f"\n  Saved: {viz_path}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert len(importances) == len(feature_names)
assert sum(importances) > 0
print("\n[ok] Checkpoint 2 passed — feature importance extracted and visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Credit Risk Team Explainability
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS must explain every automated decline to MAS under the
# FEAT principles (Fairness, Ethics, Accountability, Transparency). The
# XGBoost gain importance above is the first artifact the risk team
# delivers. Expected top features: Debt Service Ratio, months since last
# bounced cheque, unsecured credit exposure, employment length.
#
# A suspicious top-3 feature (e.g., postal code region) triggers a bias
# audit under PACT/MAS guidelines before the model ships.
#
# BUSINESS IMPACT: DBS approves ~S$12B/year in unsecured credit. A 1pp
# lift in AUC-PR is worth S$25-40M/year in avoided losses at 40-60% LGD.
# Catching a proxy-discrimination feature pre-launch avoids a ~S$5M MAS
# remediation cost.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Trained XGBoost on real Singapore credit (AUC-PR={metrics['auc_pr']:.4f})
  [x] Connected every hyperparameter to the theory from 4.1
  [x] Used AUC-PR as the primary metric on imbalanced data
  [x] Ranked features by gain importance
  [x] Mapped the ranking to DBS's MAS FEAT compliance workflow

  Next: 03_lightgbm_catboost.py — the same data, two alternative
  libraries, and the decision tree for picking one.
"""
)
