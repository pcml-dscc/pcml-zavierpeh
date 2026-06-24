# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.2: K-Nearest Neighbors
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Lazy (instance-based) learning with no training phase
#   - k sweep + distance metric comparison
#   - Curse of dimensionality and why scaling matters
#   - Jagged 2D decision boundaries
#   - Cold-start churn for a new Singapore marketplace
#
# ESTIMATED TIME: ~25 min
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
# KNN memorises the training set. Prediction = majority vote over the k
# closest neighbors. Scaling is mandatory (z-score already applied by
# the shared pipeline). Euclidean / Manhattan / Cosine are the standard
# metric choices.


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

K_VALUES = [1, 3, 5, 7, 11, 15, 21, 31]
METRICS = ["euclidean", "manhattan", "cosine"]

print("\n--- k sweep (euclidean distance) ---")
print(f"{'k':>6} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 34)
k_results: dict[int, dict[str, float]] = {}
for k in K_VALUES:
    # TODO: build KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    # and call cv_accuracy_f1. Store {"accuracy", "f1"} in k_results[k].
    acc, f1 = ____
    k_results[k] = ____
    print(f"{k:>6} {acc:>14.4f} {f1:>10.4f}")

# TODO: pick the k with the highest F1.
best_k = ____
print(f"\nBest k: {best_k}")

print(f"\n--- distance metric sweep (k={best_k}) ---")
print(f"{'metric':<12} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 40)
metric_results: dict[str, dict[str, float]] = {}
for m in METRICS:
    # TODO: build KNeighborsClassifier(n_neighbors=best_k, metric=m)
    # and cv_accuracy_f1. Store in metric_results[m].
    acc, f1 = ____
    metric_results[m] = ____
    print(f"{m:<12} {acc:>14.4f} {f1:>10.4f}")

best_metric = max(metric_results, key=lambda m: metric_results[m]["f1"])
print(f"\nBest metric: {best_metric}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: final KNN
# ════════════════════════════════════════════════════════════════════════

# TODO: fit_and_evaluate with KNeighborsClassifier(n_neighbors=best_k,
# metric=best_metric) and name f"KNN (k={best_k}, {best_metric})".
knn_result = ____

print(
    f"\n{knn_result['name']}: trained in {knn_result['train_time']:.4f}s | "
    f"accuracy={knn_result['accuracy']:.4f} | F1={knn_result['f1']:.4f}"
)
print_classification_report(y_test, knn_result["pred"])

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert knn_result["accuracy"] > 0.5, "KNN must beat random"
assert best_k > 1, "Best k should be > 1 (k=1 always overfits)"
print("[ok] Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]

# TODO: fit KNN (best_k, best_metric) on X_train_2d and predict over the
# mesh returned by decision_boundary_mesh(X_train_2d).
knn_2d = ____
xx, yy = decision_boundary_mesh(X_train_2d)
Z = ____

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
print(f"Decision mesh shape: {Z.shape}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert Z.shape[0] > 0 and Z.shape[1] > 0
print("[ok] Checkpoint 2 passed — KNN 2D boundary computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: cold-start churn for new Singapore marketplaces
# ════════════════════════════════════════════════════════════════════════
# Why KNN fits: zero training cost, works on a few thousand labelled
# customers, easy to deploy. Limitations: prediction cost scales with
# dataset size; poor interpretability.

# TODO: compute true_positives and dollars_saved.
true_positives = ____
dollars_saved = ____
print(f"\nTrue positives: {true_positives}")
print(f"Net retention value: S${dollars_saved:,.2f}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(
    f"""
  [x] Lazy learning — fit() is trivial, predict() does the work
  [x] CV-driven k selection and metric comparison
  [x] Accuracy: {knn_result['accuracy']:.4f}, F1: {knn_result['f1']:.4f}
  [x] Jagged 2D decision boundary
  [x] Cold-start business case — S${dollars_saved:,.0f} retained

  Next: 03_naive_bayes.py
"""
)
