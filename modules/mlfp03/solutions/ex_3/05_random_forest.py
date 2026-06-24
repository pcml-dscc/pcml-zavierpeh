# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.5: Random Forests (bagging + OOB)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Bagging: bootstrap sampling + feature subsampling
#   - OOB (out-of-bag) estimation as a free cross-validation proxy
#   - The (1 - 1/n)^n -> 1/e result behind OOB coverage
#   - Read Random Forest feature importances and compare to a single tree
#   - Visualise OOB convergence as the forest grows
#   - Apply a robust, drop-in churn model to Singapore e-commerce scale
#
# PREREQUISITES: 04_decision_tree.py
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — bagging, OOB, feature importance
#   2. Build — RF with 200 trees, sqrt feature subsampling
#   3. Train — inspect OOB score and compare to CV
#   4. Visualise — OOB convergence curve + importance bar chart
#   5. Apply — Singapore e-commerce marketplace churn at 250K MAU
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

from shared.mlfp03.ex_3 import (
    build_train_test_split,
    churn_saved_dollars,
    cv_accuracy_f1,
    decision_boundary_mesh,
    fit_and_evaluate,
    get_visualizer,
    OUTPUT_DIR,
    print_classification_report,
    project_2d,
    RANDOM_SEED,
)

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# THEORY — Bagging and OOB Estimation
# ════════════════════════════════════════════════════════════════════════
# A Random Forest is a bag of decision trees where:
#   1. Each tree is trained on a bootstrap sample of the training data
#      (n rows sampled with replacement).
#   2. At every split, the tree considers only a random subset of
#      features (sqrt(n_features) by default for classification).
#
# These two sources of randomness DE-CORRELATE the trees: each one
# makes different mistakes, so averaging their votes cancels noise and
# leaves the signal.
#
# OOB estimation: for a bootstrap sample of size n, each row has a
#     P(NOT in sample) = (1 - 1/n)^n
# probability of being absent. As n -> infinity this tends to 1/e
# ≈ 0.368. So roughly 36.8% of rows are OUT of any given tree's
# training set. We evaluate each row using only the trees that did NOT
# see it — a free cross-validation proxy baked into training itself.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: fit a 200-tree Random Forest
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.5 — Random Forests")
print("=" * 70)

data = build_train_test_split()
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
cv = data["cv"]
feature_names = data["feature_names"]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# Verify the 1/e OOB fraction analytically
n = X_train.shape[0]
oob_fraction_formula = (1 - 1 / n) ** n
print(
    f"\nOOB fraction formula (1 - 1/n)^n for n={n}: "
    f"{oob_fraction_formula:.4f}  (asymptote 1/e = {1/math.e:.4f})"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: inspect OOB score and CV consistency
# ════════════════════════════════════════════════════════════════════════

rf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    oob_score=True,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
rf_result = fit_and_evaluate(
    rf,
    X_train,
    y_train,
    X_test,
    y_test,
    name="RandomForest (200 trees)",
)
rf_model = rf_result["model"]

print(
    f"\n{rf_result['name']}: trained in {rf_result['train_time']:.2f}s | "
    f"accuracy={rf_result['accuracy']:.4f} | "
    f"F1={rf_result['f1']:.4f} | AUC={rf_result['auc_roc']:.4f}"
)
print(f"OOB score: {rf_model.oob_score_:.4f}")
print_classification_report(y_test, rf_result["pred"])

rf_cv_acc, rf_cv_f1 = cv_accuracy_f1(
    RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
    X_train,
    y_train,
    cv,
)
print(
    f"5-fold CV — accuracy: {rf_cv_acc:.4f} | F1: {rf_cv_f1:.4f} "
    f"(OOB should be within 10pp)"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: OOB convergence + importance
# ════════════════════════════════════════════════════════════════════════

print("\n--- OOB convergence vs number of trees ---")
n_trees_grid = [10, 25, 50, 75, 100, 150, 200]
oob_scores: list[float] = []
print(f"{'trees':>8} {'OOB score':>12}")
print("-" * 24)
for n_trees in n_trees_grid:
    rf_tmp = RandomForestClassifier(
        n_estimators=n_trees,
        max_features="sqrt",
        oob_score=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf_tmp.fit(X_train, y_train)
    oob_scores.append(float(rf_tmp.oob_score_))
    print(f"{n_trees:>8} {rf_tmp.oob_score_:>12.4f}")

importances = sorted(
    zip(feature_names, rf_model.feature_importances_),
    key=lambda pair: pair[1],
    reverse=True,
)
print("\n--- Top feature importances ---")
print(f"{'feature':<30} {'importance':>12}")
print("-" * 44)
for name, imp in importances[:10]:
    bar = "#" * int(imp * 50)
    print(f"{name:<30} {imp:>12.4f}  {bar}")

viz = get_visualizer()
fig_oob = viz.training_history(
    {"OOB score": oob_scores},
    x_label="trees (index into n_trees_grid)",
)
fig_oob.update_layout(title="Random Forest — OOB score vs number of trees")
out_oob = OUTPUT_DIR / "ex3_05_rf_oob.html"
fig_oob.write_html(str(out_oob))
print(f"\nSaved: {out_oob}")

fig_imp = viz.training_history(
    {"importance": [imp for _, imp in importances[:10]]},
    x_label="feature rank (top 10)",
)
fig_imp.update_layout(title="Random Forest — top 10 feature importances")
out_imp = OUTPUT_DIR / "ex3_05_rf_importance.html"
fig_imp.write_html(str(out_imp))
print(f"Saved: {out_imp}")

# 2D decision boundary
pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]
rf_2d = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
rf_2d.fit(X_train_2d, y_train)
xx, yy = decision_boundary_mesh(X_train_2d)
Z = rf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
print(
    f"\nDecision mesh shape: {Z.shape} | "
    f"PCA variance captured: {pca_bundle['explained_variance'].sum():.2%}"
)
print("RF boundaries look like 'voted' trees — axis-aligned but smoothed.")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert rf_result["accuracy"] > 0.5, "Random Forest must beat random"
assert rf_model.oob_score_ > 0.5, "OOB score should beat random"
assert abs(rf_model.oob_score_ - rf_cv_acc) < 0.10, "OOB and CV within 10pp"
print("\n[ok] Checkpoint 1 passed — OOB + CV consistent, RF trained and visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: 250K MAU Singapore marketplace churn model
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A mid-market Singapore e-commerce platform wants a single
# production churn model to replace a stack of hand-tuned heuristics.
# Requirements:
#   - Robust across seasonal shifts (11.11, 12.12, Chinese New Year)
#   - Tolerant of mixed feature types and missing values
#   - A single well-understood hyperparameter (number of trees) the
#     retention team can scale up or down based on nightly compute
#     budget
#
# Why Random Forest fits:
#   - Bagging + feature subsampling produce calibrated, robust models
#     with minimal tuning. "A Random Forest of 200 trees on whatever
#     cleaned features we have" is the single most reliable baseline
#     in tabular ML.
#   - OOB gives a trustworthy accuracy estimate without a holdout
#     split, which saves training data for regions with thin coverage
#     (e.g. new campaign cohorts).
#   - Feature importance lists give the retention team a plain-English
#     story for every model refresh.
#
# LIMITATIONS:
#   - Memory: 200 deep trees on 250K customers is a multi-gigabyte
#     model. Move to gradient boosting (Exercise 4) for tighter limits.
#   - Black-box per-prediction: individual predictions don't have a
#     single clean decision path. For that, drop back to a single
#     decision tree (04_decision_tree.py).

true_positives = int(((rf_result["pred"] == 1) & (y_test == 1)).sum())
dollars_saved = churn_saved_dollars(true_positives)
print(f"\nBusiness impact on held-out test set ({len(y_test)} customers):")
print(f"  True positives (churners caught): {true_positives}")
print(f"  Net retention value at 40% offer acceptance: S${dollars_saved:,.2f}")
monthly_scale = dollars_saved * (250_000 / len(y_test))
print(
    f"  Extrapolated to 250K monthly active base: "
    f"S${monthly_scale:,.0f} / month retained value"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Bagging — bootstrap + feature subsampling de-correlates trees
  [x] OOB score as a free cross-validation proxy
  [x] The (1 - 1/n)^n -> 1/e result behind OOB coverage
  [x] Held-out accuracy: {rf_result['accuracy']:.4f}, F1: {rf_result['f1']:.4f}
  [x] OOB convergence curve and feature importance plot
  [x] 250K MAU churn business case
      — S${monthly_scale:,.0f}/month retained value at scale

  Next: 06_model_zoo.py — direct head-to-head comparison across all 5.
"""
)
