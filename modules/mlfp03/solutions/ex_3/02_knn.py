# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.2: K-Nearest Neighbors
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Understand instance-based (lazy) learning: no training phase
#   - Sweep k and compare Euclidean / Manhattan / Cosine distance metrics
#   - Recognise the curse of dimensionality and why scaling matters
#   - Visualise the jagged KNN decision boundary in 2D PCA space
#   - Apply KNN to a Singapore e-commerce cold-start churn scenario
#
# PREREQUISITES: 01_svm.py (shared preprocessing pipeline)
#
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — lazy learning, distance metrics, curse of dimensionality
#   2. Build — k sweep, distance metric comparison
#   3. Train — final KNN with the best (k, metric) pair
#   4. Visualise — 2D decision boundary (jagged, instance-specific)
#   5. Apply — cold-start churn flagging for new Singapore marketplaces
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from sklearn.neighbors import KNeighborsClassifier

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
)

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# THEORY — Lazy Learning and Distance
# ════════════════════════════════════════════════════════════════════════
# KNN has NO training phase. Fit() just memorises the training set.
# Predict() searches the k closest training points and takes a majority
# vote.
#
#   Prediction: y_hat = mode({y_j : j in k nearest neighbors of x})
#
# Distance metrics:
#     Euclidean  = sqrt(Σ (x_i - y_i)²)        — sphere of influence
#     Manhattan  = Σ |x_i - y_i|                — axis-aligned grid
#     Cosine     = 1 - (x . y) / (||x|| ||y||)  — direction, not magnitude
#
# CURSE OF DIMENSIONALITY: as the feature count grows, ALL pairs of
# points become roughly equidistant — the notion of "nearest" breaks
# down. KNN is therefore strong on small, low-dimensional data and
# weak once you pass ~50 features.
#
# SCALING IS MANDATORY: a feature with range [0, 100000] will dominate
# every distance computation. The shared preprocessing pipeline already
# applies z-score normalisation before we see the data.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: k sweep + distance metric comparison
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.2 — K-Nearest Neighbors")
print("=" * 70)

data = build_train_test_split()
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
cv = data["cv"]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

K_VALUES = [1, 3, 5, 7, 11, 15, 21, 31]
METRICS = ["euclidean", "manhattan", "cosine"]

print("\n--- k sweep (euclidean distance) ---")
print(f"{'k':>6} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 34)
k_results: dict[int, dict[str, float]] = {}
for k in K_VALUES:
    acc, f1 = cv_accuracy_f1(
        KNeighborsClassifier(n_neighbors=k, metric="euclidean"),
        X_train,
        y_train,
        cv,
    )
    k_results[k] = {"accuracy": acc, "f1": f1}
    print(f"{k:>6} {acc:>14.4f} {f1:>10.4f}")

best_k = max(k_results, key=lambda k: k_results[k]["f1"])
print(f"\nBest k: {best_k}")

print(f"\n--- distance metric sweep (k={best_k}) ---")
print(f"{'metric':<12} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 40)
metric_results: dict[str, dict[str, float]] = {}
for m in METRICS:
    acc, f1 = cv_accuracy_f1(
        KNeighborsClassifier(n_neighbors=best_k, metric=m),
        X_train,
        y_train,
        cv,
    )
    metric_results[m] = {"accuracy": acc, "f1": f1}
    print(f"{m:<12} {acc:>14.4f} {f1:>10.4f}")

best_metric = max(metric_results, key=lambda m: metric_results[m]["f1"])
print(f"\nBest metric: {best_metric}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: final KNN
# ════════════════════════════════════════════════════════════════════════

knn_result = fit_and_evaluate(
    KNeighborsClassifier(n_neighbors=best_k, metric=best_metric),
    X_train,
    y_train,
    X_test,
    y_test,
    name=f"KNN (k={best_k}, {best_metric})",
)

print(
    f"\n{knn_result['name']}: trained in {knn_result['train_time']:.4f}s | "
    f"accuracy={knn_result['accuracy']:.4f} | "
    f"F1={knn_result['f1']:.4f} | AUC={knn_result['auc_roc']:.4f}"
)
print_classification_report(y_test, knn_result["pred"])

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert knn_result["accuracy"] > 0.5, "KNN must beat random"
assert best_k > 1, "Best k should be > 1 (k=1 always overfits)"
print("[ok] Checkpoint 1 passed — KNN trained and evaluated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: 2D decision boundary (expect jagged)
# ════════════════════════════════════════════════════════════════════════

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]

knn_2d = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
knn_2d.fit(X_train_2d, y_train)
xx, yy = decision_boundary_mesh(X_train_2d)
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

viz = get_visualizer()
fig = viz.training_history(
    {
        "k accuracy": [k_results[k]["accuracy"] for k in K_VALUES],
        "k F1": [k_results[k]["f1"] for k in K_VALUES],
    },
    x_label="k (index into K_VALUES)",
)
fig.update_layout(title="KNN: accuracy / F1 vs k")
out = OUTPUT_DIR / "ex3_02_knn_k_sweep.html"
fig.write_html(str(out))
print(f"Saved: {out}")
print(
    f"Decision mesh shape: {Z.shape} | "
    f"PCA variance captured: {pca_bundle['explained_variance'].sum():.2%}"
)
print("KNN boundaries are jagged because every cell votes from its nearest k.")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert Z.shape[0] > 0 and Z.shape[1] > 0, "Decision boundary mesh is empty"
print("[ok] Checkpoint 2 passed — KNN 2D boundary computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: cold-start churn for new Singapore marketplaces
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A new Singapore marketplace (e.g. a niche B2C fashion
# platform) has only a few thousand labelled customers in its first six
# months. The data science team needs a churn model that works from
# day one, even before enough data exists to justify a deep model.
#
# Why KNN fits:
#   - Zero training cost. Add new customers to the reference set and
#     predict immediately.
#   - Low ceremony: no hyperparameters beyond k.
#   - Mid-dimensional (~20 features) means the curse of dimensionality
#     is mild and distance still means something.
#
# LIMITATIONS:
#   - Prediction cost grows linearly with dataset size — at 250K
#     customers, every prediction scans 250K vectors.
#   - Anisotropic feature space: unevenly-scaled or correlated features
#     distort the distance metric. We mitigate via z-score normalisation.
#   - Difficult to interpret individual predictions. Retention teams
#     prefer "feature X was high" over "customer Y was similar to 11
#     past churners".

true_positives = int(((knn_result["pred"] == 1) & (y_test == 1)).sum())
dollars_saved = churn_saved_dollars(true_positives)
print(f"\nBusiness impact on held-out test set ({len(y_test)} customers):")
print(f"  True positives (churners caught): {true_positives}")
print(f"  Net retention value at 40% offer acceptance: S${dollars_saved:,.2f}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Lazy learning: fit() is trivial, predict() does the work
  [x] k selection via CV — k=1 always overfits, large k under-fits
  [x] Distance metric comparison (Euclidean / Manhattan / Cosine)
  [x] Held-out accuracy: {knn_result['accuracy']:.4f}, F1: {knn_result['f1']:.4f}
  [x] Jagged, instance-specific 2D decision boundary
  [x] Cold-start churn business case — S${dollars_saved:,.0f} retained on test

  Next: 03_naive_bayes.py — Bayes theorem applied to classification.
"""
)
