# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.3: LightGBM and CatBoost — Same Task, Faster Trees
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Train LightGBM and CatBoost on the same Singapore credit data
#   - Explain each library's split-finding strategy
#   - Compare AUC-PR, log loss, and train time across three libraries
#   - Decide which library to use based on data shape
#
# PREREQUISITES: Exercise 4.2 (XGBoost baseline).
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — exact vs histogram vs ordered boosting
#   2. Build — three models with matched hyperparameters
#   3. Train — fit each, record AUC-PR and train time
#   4. Visualise — side-by-side comparison chart
#   5. Apply — FairPrice fraud detection library choice
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import plotly.graph_objects as go
from dotenv import load_dotenv

from shared.mlfp03.ex_4 import (
    OUTPUT_DIR,
    evaluate_classifier,
    make_catboost,
    make_lightgbm,
    make_xgboost,
    prepare_credit_split,
    print_metrics,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Three Libraries, One Gradient-Boosting Recipe
# ════════════════════════════════════════════════════════════════════════
# All three build additive trees on pseudo-residuals. They differ in
# HOW each tree finds the best split:
#
#   XGBoost:  exact (sort every feature, try every threshold)
#   LightGBM: histogram bins + GOSS + leaf-wise growth (faster on big data)
#   CatBoost: ordered boosting (no target leakage on categoricals)
#
# Practical decision tree:
#   > 1M rows, mostly numeric       → LightGBM
#   > High-cardinality categoricals → CatBoost
#   > Medium mixed data             → XGBoost


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD all three models with matched defaults
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  LightGBM + CatBoost vs XGBoost — Singapore Credit Scoring")
print("=" * 70)

data = prepare_credit_split()
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

print(f"\n  Train: {X_train.shape} | Test: {X_test.shape}")
print(f"  Default rate: {data['default_rate']:.2%}")

# TODO: Build a dict of three models using the course-standard factories
# make_xgboost, make_lightgbm, make_catboost.
# Use n_estimators=500 (iterations=500 for CatBoost), learning_rate=0.1,
# max_depth=6 (depth=6 for CatBoost).
models = {
    "XGBoost": ____,
    "LightGBM": ____,
    "CatBoost": ____,
}


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN each library and record time + metrics
# ════════════════════════════════════════════════════════════════════════

results: dict[str, dict] = {}
print("\n  --- Training ---")
for name, model in models.items():
    t0 = time.perf_counter()
    # TODO: Fit the model. CatBoost takes eval_set=(X_test, y_test);
    # XGBoost takes eval_set=[(X_test, y_test)], verbose=False; LightGBM 4.x
    # takes eval_set=[(X_test, y_test)] with NO verbose (its constructor
    # already sets verbose=-1).
    if name == "CatBoost":
        ____
    elif name == "LightGBM":
        ____
    else:
        ____
    train_time = time.perf_counter() - t0

    # TODO: Get class-1 probabilities and evaluate.
    y_proba = ____
    metrics = ____
    results[name] = {"metrics": metrics, "train_time": train_time, "model": model}
    print_metrics(name, metrics, train_time=train_time)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
for name, r in results.items():
    assert r["metrics"]["auc_roc"] > 0.7, f"{name} AUC-ROC should exceed 0.7"
    assert r["metrics"]["auc_pr"] > 0.3, f"{name} AUC-PR should exceed 0.3"
print("\n[ok] Checkpoint 1 passed — all three libraries trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the comparison
# ════════════════════════════════════════════════════════════════════════

names = list(results.keys())
auc_pr_values = [results[n]["metrics"]["auc_pr"] for n in names]
auc_roc_values = [results[n]["metrics"]["auc_roc"] for n in names]
log_loss_values = [results[n]["metrics"]["log_loss"] for n in names]
time_values = [results[n]["train_time"] for n in names]

fig = go.Figure()
# TODO: Add two grouped Bar traces: one for AUC-PR on yaxis y1, one for
# train time on yaxis y2.
____
____
fig.update_layout(
    title="Boosting Library Comparison — Singapore Credit Default",
    barmode="group",
    yaxis=dict(title="AUC-PR", range=[0, max(auc_pr_values) * 1.25]),
    yaxis2=dict(title="Train time (s)", overlaying="y", side="right"),
)
viz_path = OUTPUT_DIR / "ex4_03_library_comparison.html"
fig.write_html(viz_path)
print(f"  Saved: {viz_path}")

print("\n  --- Library Comparison Table ---")
print(
    f"  {'Library':<10} {'AUC-ROC':>10} {'AUC-PR':>10} {'Log Loss':>10} {'Time (s)':>10}"
)
print("  " + "─" * 56)
for n, auc_pr, auc_roc, ll, t in zip(
    names, auc_pr_values, auc_roc_values, log_loss_values, time_values
):
    print(f"  {n:<10} {auc_roc:>10.4f} {auc_pr:>10.4f} {ll:>10.4f} {t:>10.2f}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
best_name = max(results.items(), key=lambda kv: kv[1]["metrics"]["auc_pr"])[0]
fastest_name = min(results.items(), key=lambda kv: kv[1]["train_time"])[0]
assert best_name in results
assert fastest_name in results
print("\n[ok] Checkpoint 2 passed — comparison computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: FairPrice Fraud Detection Library Choice
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: NTUC FairPrice, 160 stores, ~4M transactions/day, 40%
# categorical features including 12,000-level SKU bundles and 9,000-
# level card BINs, 0.2% fraud rate, nightly re-train on 120M rows.
#
# LIBRARY CHOICE: CatBoost — ordered boosting handles high-cardinality
# categoricals without manual target encoding. LightGBM/XGBoost would
# require a cross-validated target encoding pipeline that owners own
# forever. 3-5x train-time penalty is acceptable in a nightly window.
#
# BUSINESS IMPACT: ~S$4.5B card volume/year → S$1.3-2.7M/year in fraud
# losses. AUC-PR 0.65 recovers ~40% → S$800K-1.1M/year avoided losses
# against S$80K/year infrastructure cost. 10-14x ROI.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Trained XGBoost, LightGBM, CatBoost with matched hyperparameters
  [x] Read each library's design (exact / histogram / ordered boosting)
  [x] Compared AUC-PR and train time side-by-side
  [x] Best by AUC-PR: {best_name} | Fastest: {fastest_name}
  [x] Matched library choice to data shape using FairPrice scenario

  Next: 04_boosting_tuning.py — sweeps, heatmaps, and early stopping.
"""
)
