# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.5: Random Forests (bagging + OOB)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Bagging: bootstrap sampling + feature subsampling
#   - OOB estimation as a free cross-validation proxy
#   - (1 - 1/n)^n -> 1/e result behind OOB coverage
#   - Feature importance from the forest
#   - 250K MAU Singapore marketplace scale-out
#
# ESTIMATED TIME: ~30 min
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
# THEORY — Bagging and OOB
# ════════════════════════════════════════════════════════════════════════
# Each tree trains on a bootstrap sample of n rows. Each split uses a
# random subset of features (sqrt by default). ~36.8% of rows are OOB
# for any given tree, giving a free cross-validation proxy.


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

n = X_train.shape[0]
oob_fraction_formula = (1 - 1 / n) ** n
print(
    f"OOB fraction (1 - 1/n)^n for n={n}: "
    f"{oob_fraction_formula:.4f}  (asymptote 1/e = {1/math.e:.4f})"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN
# ════════════════════════════════════════════════════════════════════════

# TODO: build RandomForestClassifier with n_estimators=200,
# max_features="sqrt", oob_score=True, random_state=RANDOM_SEED, n_jobs=-1.
# Pass it through fit_and_evaluate with name "RandomForest (200 trees)".
rf = ____
rf_result = ____
rf_model = rf_result["model"]

print(
    f"\n{rf_result['name']}: trained in {rf_result['train_time']:.2f}s | "
    f"accuracy={rf_result['accuracy']:.4f} | F1={rf_result['f1']:.4f}"
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
print(f"5-fold CV — accuracy: {rf_cv_acc:.4f} | F1: {rf_cv_f1:.4f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: OOB convergence + importance
# ════════════════════════════════════════════════════════════════════════

print("\n--- OOB convergence ---")
n_trees_grid = [10, 25, 50, 75, 100, 150, 200]
oob_scores: list[float] = []
print(f"{'trees':>8} {'OOB score':>12}")
print("-" * 24)
for n_trees in n_trees_grid:
    # TODO: fit a RandomForestClassifier with n_estimators=n_trees, sqrt
    # features, oob_score=True, random_state=RANDOM_SEED, n_jobs=-1.
    # Append its oob_score_ to oob_scores.
    rf_tmp = ____
    rf_tmp.fit(X_train, y_train)
    oob_scores.append(float(rf_tmp.oob_score_))
    print(f"{n_trees:>8} {rf_tmp.oob_score_:>12.4f}")

importances = sorted(
    zip(feature_names, rf_model.feature_importances_),
    key=lambda pair: pair[1],
    reverse=True,
)
print("\n--- Top feature importances ---")
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

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]
# TODO: fit RandomForestClassifier(n_estimators=100, max_features="sqrt",
# random_state=RANDOM_SEED, n_jobs=-1) on X_train_2d and predict over the mesh.
rf_2d = ____
xx, yy = decision_boundary_mesh(X_train_2d)
Z = ____
print(f"Decision mesh shape: {Z.shape}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert rf_result["accuracy"] > 0.5
assert rf_model.oob_score_ > 0.5
assert abs(rf_model.oob_score_ - rf_cv_acc) < 0.10, "OOB and CV within 10pp"
print("\n[ok] Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: 250K MAU Singapore marketplace churn model
# ════════════════════════════════════════════════════════════════════════
# Why RF: robust default, no tuning burden, OOB gives honest accuracy
# without a holdout. Limitations: memory footprint and per-prediction
# interpretability.

# TODO: compute true_positives, dollars_saved, and scale to 250K MAU.
true_positives = ____
dollars_saved = ____
monthly_scale = dollars_saved * (250_000 / len(y_test))
print(f"\nTrue positives: {true_positives}")
print(f"Net retention value (test set): S${dollars_saved:,.2f}")
print(f"Monthly at 250K MAU: S${monthly_scale:,.0f}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(
    f"""
  [x] Bagging + feature subsampling = de-correlated trees
  [x] OOB score as a free CV proxy
  [x] Accuracy: {rf_result['accuracy']:.4f}, F1: {rf_result['f1']:.4f}
  [x] 250K MAU business case: S${monthly_scale:,.0f}/month retained value

  Next: 06_model_zoo.py
"""
)
