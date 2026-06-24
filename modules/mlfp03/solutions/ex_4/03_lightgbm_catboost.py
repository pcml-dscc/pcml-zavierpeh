# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.3: LightGBM and CatBoost — Same Task, Faster Trees
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Train LightGBM and CatBoost on the same Singapore credit data
#   - Explain LightGBM's histogram-based split finding (why it's fast on
#     large data) and GOSS (gradient-based one-side sampling)
#   - Explain CatBoost's ordered boosting (why it's robust to target
#     leakage on categorical features)
#   - Compare the three libraries on AUC-PR, log loss, and train time
#   - Decide which library to use based on data shape and constraints
#
# PREREQUISITES: Exercise 4.2 (XGBoost baseline on the same dataset).
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — how LightGBM and CatBoost differ from XGBoost
#   2. Build — LightGBM + CatBoost with matched hyperparameters
#   3. Train — fit both, record train time and AUC-PR
#   4. Visualise — side-by-side bar chart with XGBoost as baseline
#   5. Apply — Singapore supermarket chain picks a library for fraud
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
# All three libraries build additive trees on pseudo-residuals — the
# theory from 4.1 applies identically. They differ in HOW they find the
# best split at each round:
#
# XGBoost: exact split finding. Sorts every feature, evaluates every
#   possible split threshold. Accurate but O(n·d·log n) per round.
#
# LightGBM: histogram-based split finding. Buckets each feature into
#   ~256 bins, evaluates splits between bins only. O(n·d) per round —
#   ~5-10x faster on large data. Adds two more tricks:
#     - GOSS (gradient-based one-side sampling): keep all rows with
#       large gradients (hard examples), randomly sample the easy ones.
#     - EFB (exclusive feature bundling): bundle sparse features that
#       never overlap into a single "super-feature" — further speedup
#       on wide sparse data.
#   LightGBM also grows trees leaf-wise (best-first) instead of
#   level-wise, which usually gives lower loss for the same leaf count
#   but can overfit if max_depth is unset.
#
# CatBoost: ordered boosting. The problem CatBoost solves is target
#   leakage on categorical features — "mean-target encoding" (replace
#   each category with its average target) leaks the target into the
#   training set, inflating training accuracy. CatBoost fixes this by
#   computing target statistics only from rows that come BEFORE the
#   current row in a random permutation, then re-permuting each round.
#   Result: best out-of-the-box performance on data with many high-
#   cardinality categoricals, and the least tuning effort of the three.
#
# Practical decision tree:
#   ≫ >1M rows, mostly numeric        → LightGBM (speed wins)
#   ≫ High-cardinality categoricals   → CatBoost (no manual encoding)
#   ≫ Medium data, mixed types        → XGBoost (stable default)
#
# We train all three with matched hyperparameters so the comparison is
# about the LIBRARY, not the hyperparameters.


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

models = {
    "XGBoost": make_xgboost(n_estimators=500, learning_rate=0.1, max_depth=6),
    "LightGBM": make_lightgbm(n_estimators=500, learning_rate=0.1, max_depth=6),
    "CatBoost": make_catboost(iterations=500, learning_rate=0.1, depth=6),
}


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN each library and record time + metrics
# ════════════════════════════════════════════════════════════════════════

results: dict[str, dict] = {}
print("\n  --- Training ---")
for name, model in models.items():
    t0 = time.perf_counter()
    if name == "CatBoost":
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
    elif name == "LightGBM":
        # LightGBM 4.x dropped fit(verbose=); the constructor's verbose=-1 silences it.
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    else:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_time = time.perf_counter() - t0

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_classifier(y_test, y_proba)
    results[name] = {"metrics": metrics, "train_time": train_time, "model": model}
    print_metrics(name, metrics, train_time=train_time)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
for name, r in results.items():
    assert r["metrics"]["auc_roc"] > 0.7, f"{name} AUC-ROC should exceed 0.7"
    assert r["metrics"]["auc_pr"] > 0.3, f"{name} AUC-PR should exceed 0.3"
# INTERPRETATION: On 5K rows the three libraries are usually within 0.01
# AUC-PR of each other — the data is too small to favour one algorithm.
# The interesting signal here is train time: CatBoost is typically 3-5x
# slower than XGBoost on this size because of the ordered-boosting
# permutation overhead. LightGBM is usually fastest.
print("\n[ok] Checkpoint 1 passed — all three libraries trained on credit data\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the comparison
# ════════════════════════════════════════════════════════════════════════
# One chart with two stacked panels: AUC-PR (higher is better) and train
# time (lower is better). Side-by-side bars so the speed/quality trade-
# off jumps out visually.

names = list(results.keys())
auc_pr_values = [results[n]["metrics"]["auc_pr"] for n in names]
auc_roc_values = [results[n]["metrics"]["auc_roc"] for n in names]
log_loss_values = [results[n]["metrics"]["log_loss"] for n in names]
time_values = [results[n]["train_time"] for n in names]

fig = go.Figure()
fig.add_trace(
    go.Bar(name="AUC-PR (higher=better)", x=names, y=auc_pr_values, yaxis="y1")
)
fig.add_trace(
    go.Bar(name="Train time (s, lower=better)", x=names, y=time_values, yaxis="y2")
)
fig.update_layout(
    title="Boosting Library Comparison — Singapore Credit Default",
    barmode="group",
    yaxis=dict(title="AUC-PR", range=[0, max(auc_pr_values) * 1.25]),
    yaxis2=dict(title="Train time (s)", overlaying="y", side="right"),
)
viz_path = OUTPUT_DIR / "ex4_03_library_comparison.html"
fig.write_html(viz_path)
print(f"  Saved: {viz_path}")

# Console table so the reader sees numbers even without opening the HTML
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
assert best_name in results, "best_name must be a trained library"
assert fastest_name in results, "fastest_name must be a trained library"
# INTERPRETATION: Report BOTH numbers ("best by AUC-PR", "fastest"). On a
# 5K-row dataset these are usually different libraries — the fastest is
# often not the most accurate. Knowing both lets you choose: do you pay
# for 0.005 AUC-PR with 3x training time? Depends on how often you
# re-train and how much each default costs.
print("\n[ok] Checkpoint 2 passed — comparison table computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: FairPrice Fraud Detection Library Choice
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: NTUC FairPrice (Singapore's largest grocery retailer) runs a
# real-time fraud-detection model on card-present transactions across
# ~160 stores. The model scores every swipe in <50ms and flags suspects
# for a manager callback before the receipt prints.
#
# Data shape:
#   - Rows: ~4M transactions per day
#   - Features: 60% numeric (amount, velocity, time-of-day), 40%
#     categorical (store, SKU bundles, payment network, card BIN)
#   - Cardinality: store has 160 levels; SKU bundle has ~12,000 levels;
#     card BIN has ~9,000 levels
#   - Class balance: 0.2% fraud rate (severely imbalanced)
#   - Re-train cadence: nightly, on the last 30 days = ~120M rows
#
# LIBRARY CHOICE: CatBoost
#   - High-cardinality categoricals (SKU bundle, card BIN) would require
#     ordinal encoding for XGBoost/LightGBM, leaking ordering info into
#     features that have no natural order.
#   - CatBoost's ordered boosting handles them without manual encoding
#     and without target leakage.
#   - The 3-5x train-time penalty vs LightGBM is acceptable: 120M rows
#     trains overnight on a 16-core box (~4 hours), well inside the
#     nightly window.
#
# WHY NOT LIGHTGBM: LightGBM on 160 store levels and 12K SKU bundles
# requires target encoding — this is the exact failure mode CatBoost was
# built for. You can make it work with careful cross-validated target
# encoding, but you own that pipeline and its bugs forever.
#
# WHY NOT XGBOOST: Same high-cardinality encoding problem, plus XGBoost
# is ~5x slower than LightGBM on 120M rows so it would blow through the
# nightly window.
#
# BUSINESS IMPACT: FairPrice processes ~S$4.5B in card transactions per
# year. Industry benchmarks put card-present fraud at 3-6 basis points
# of sales = S$1.3-2.7M in annual losses. A model with AUC-PR 0.65
# (typical for well-tuned CatBoost on this data shape) recovers ~40% of
# fraud = ~S$800K-1.1M per year in avoided losses, against a ~S$80K/year
# total infrastructure cost for training + scoring. 10-14x ROI before
# accounting for the 0.15% lift in legitimate-transaction approval rates
# from fewer false positives — which is usually the larger business win.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Trained XGBoost, LightGBM, and CatBoost on the same credit data
      with matched hyperparameters
  [x] Read each library's design in terms of split-finding strategy
      (exact vs histogram vs ordered boosting)
  [x] Compared AUC-PR and train time side-by-side
  [x] Identified {best_name} as the best on this dataset by AUC-PR
  [x] Identified {fastest_name} as the fastest to train
  [x] Matched library choice to data shape using the FairPrice scenario

  KEY INSIGHT: The three libraries are not interchangeable. LightGBM is
  the speed winner on large numeric data; CatBoost is the accuracy
  winner on high-cardinality categorical data; XGBoost is the stable
  default when the data is mixed and medium-sized. Pick before you
  tune — tuning cannot rescue a library mismatch.

  Next: 04_boosting_tuning.py — hyperparameter sweeps, learning-rate
  sensitivity, and early stopping on the same dataset.
"""
)
