# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.2: XGBoost on Singapore Credit Data
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Train an XGBoost classifier on a real imbalanced credit dataset
#   - Read XGBoost hyperparameters in terms of the theory from 4.1
#     (learning_rate ↔ η, max_depth ↔ tree size, reg_lambda ↔ λ)
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
#   2. Build — XGBoost classifier with course-standard defaults
#   3. Train — fit on Singapore credit data, time the training
#   4. Visualise — feature-importance bar chart + top-15 table
#   5. Apply — DBS credit risk team ranks features to explain decisions
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
# Every XGBoost hyperparameter corresponds directly to a lever in the
# theory from 4.1:
#
#     learning_rate (η)   → additive-model step size. Smaller = slower
#                           convergence, more robust. Production: 0.01-0.05.
#     max_depth           → tree size. Deeper trees capture interactions
#                           but overfit. Production: 4-8.
#     n_estimators        → number of rounds. Pair with early_stopping.
#     reg_lambda (λ)      → L2 on leaf weights. Shrinks leaf predictions
#                           toward zero when H (Hessian sum) is small.
#     gamma (γ)           → minimum gain to accept a split. Structural
#                           pruning: refuses to split small segments.
#     subsample           → fraction of rows used per tree (stochastic
#                           boosting). Adds variance reduction on top of
#                           the bias reduction boosting gives you.
#     colsample_bytree    → fraction of columns sampled per tree. Another
#                           stochastic regulariser.
#
# The XGBoost defaults (learning_rate=0.3, max_depth=6) are aggressive —
# they're tuned to win Kaggle competitions where speed of convergence
# matters. For credit scoring you typically move to learning_rate=0.05
# and rely on early stopping for the final round count.


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
print(f"  Default rate (imbalance): {data['default_rate']:.2%}")

# Course-standard XGBoost (see shared.mlfp03.ex_4.make_xgboost)
model = make_xgboost(n_estimators=500, learning_rate=0.1, max_depth=6)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN on Singapore credit data
# ════════════════════════════════════════════════════════════════════════

print("\n  Training XGBoost (500 rounds, η=0.1, depth=6)...")
t0 = time.perf_counter()
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
train_time = time.perf_counter() - t0

y_proba = model.predict_proba(X_test)[:, 1]
metrics = evaluate_classifier(y_test, y_proba)

print_metrics("XGBoost", metrics, train_time=train_time)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert metrics["auc_roc"] > 0.7, "XGBoost should beat 0.7 AUC-ROC on credit data"
assert (
    metrics["auc_pr"] > 0.3
), "XGBoost AUC-PR should clear the 0.3 bar (12% base rate)"
# INTERPRETATION: AUC-PR is the metric that matters here. With 12% base
# rate, random scoring gives AUC-PR ≈ 0.12 and a model that ranks every
# customer at the population rate gives AUC-ROC ≈ 0.5 — yet a bank would
# still call that "50% accurate". AUC-PR ≥ 0.3 means the model is
# surfacing real signal on the rare positives.
print("\n[ok] Checkpoint 1 passed — XGBoost trained and evaluated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE feature importance
# ════════════════════════════════════════════════════════════════════════
# XGBoost uses gain-based importance by default: the total reduction in
# loss (summed split gains) attributed to each feature. This is the right
# importance metric for this theory — it maps directly to the split-gain
# formula from 4.1.

importances = model.feature_importances_
ranked = sorted(
    zip(feature_names, importances),
    key=lambda pair: pair[1],
    reverse=True,
)
top_15 = ranked[:15]

print("  --- Top-15 Features by XGBoost Gain Importance ---")
print(f"  {'Rank':>4}  {'Feature':<30}  {'Gain':>10}")
print("  " + "─" * 50)
for rank, (name, importance) in enumerate(top_15, start=1):
    print(f"  {rank:>4}  {name:<30}  {importance:>10.4f}")

# Bar chart — saved as HTML so students can scroll/hover
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
    xaxis_title="Gain (total split-loss reduction attributed to feature)",
    yaxis_title="",
    height=520,
)
viz_path = OUTPUT_DIR / "ex4_02_xgboost_feature_importance.html"
fig.write_html(viz_path)
print(f"\n  Saved: {viz_path}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert len(importances) == len(
    feature_names
), "importance vector must match feature count"
assert sum(importances) > 0, "at least one feature must have positive importance"
# INTERPRETATION: A well-behaved gain importance distribution is typically
# dominated by 3-5 features that together account for 50%+ of the total
# gain. If it's flat (every feature ≈ equal), either the model is
# underfitting or the features are nearly-collinear proxies for each
# other. A sharply-peaked distribution means there are a few features
# that do most of the discrimination work.
print("\n[ok] Checkpoint 2 passed — feature importance extracted and visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Credit Risk Team Explainability
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS's credit risk team needs to explain every automated
# decline to the Monetary Authority of Singapore (MAS) under the FEAT
# principles (Fairness, Ethics, Accountability, Transparency) issued by
# the MAS for AI in finance.
#
# The XGBoost feature-importance ranking above is the FIRST artifact the
# team hands to MAS when asked "what is the model looking at?". But it's
# only a starting point — gain importance is population-level; for a
# specific customer's decline, the team uses SHAP values (Exercise 6) to
# break down "why was THIS application declined".
#
# The top features in Singapore credit scoring are almost always:
#   1. Debt Service Ratio (monthly debt / monthly income)
#   2. Months since last bounced cheque / missed payment
#   3. Total unsecured credit exposure
#   4. Employment length
#   5. Age of oldest credit account
#
# When the model puts one of these in the top 3, the team has a clear
# narrative to give MAS. When the model ranks something unexpected in
# top 3 (e.g., postal code region), that's a fairness red flag — it
# triggers a bias audit under PACT/MAS guidelines.
#
# BUSINESS IMPACT: DBS approves ~S$12B in new unsecured credit per year.
# A 1-percentage-point improvement in AUC-PR is worth roughly S$25-40M
# in avoided losses at typical loss-given-default rates of 40-60%. The
# gain importance view above is what lets the risk team quickly answer
# "did the model learn what we told it to learn" — a cheap 5-minute
# sanity check before every re-train goes to production. Finding a
# suspicious top feature BEFORE the model ships is worth S$0 in direct
# savings but avoids a ~S$5M MAS remediation + reputational cost if the
# model shipped with a proxy-discrimination feature in the top 3.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Trained XGBoost on real Singapore credit data (AUC-PR={metrics['auc_pr']:.4f})
  [x] Connected every hyperparameter back to the theory in 4.1
  [x] Used AUC-PR as the primary metric for 12%-positive imbalanced data
  [x] Ranked features by gain importance and interpreted the shape
  [x] Explained how DBS's credit risk team uses the ranking to satisfy
      MAS FEAT principles for AI in finance

  KEY INSIGHT: XGBoost is the default choice for tabular credit/fraud/
  risk data because (a) it handles mixed feature types, (b) the split-
  gain formula gives you structural regularisation for free, and
  (c) gain importance is a defensible, regulator-ready explanation
  artifact out of the box. Start here, compare against LightGBM for
  speed and CatBoost for categorical-heavy data.

  Next: 03_lightgbm_catboost.py — the same data, same metric, two
  alternative libraries, and the decision tree for choosing one.
"""
)
