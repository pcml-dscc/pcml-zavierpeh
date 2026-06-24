# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.4: Decision Trees (Gini from scratch + sklearn)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compute Gini impurity and entropy from scratch in numpy
#   - Simulate the first split by exhaustive search
#   - Verify against sklearn's depth-1 stump
#   - Tune max_depth with CV
#   - Interpret feature importances as an audit artefact
#   - Apply to Singapore PDPA-friendly interpretable churn flagging
#
# ESTIMATED TIME: ~35 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from sklearn.tree import DecisionTreeClassifier, export_text

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
# THEORY — Impurity and Greedy Splitting
# ════════════════════════════════════════════════════════════════════════
# Gini:    G = 1 - Σ p_k²
# Entropy: H = - Σ p_k log₂(p_k)
# IG(s) = impurity(parent) - Σ_j (|C_j| / |parent|) * impurity(C_j)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Gini + best-split from scratch
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.4 — Decision Trees")
print("=" * 70)


def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini impurity: G = 1 - Σ p_k²."""
    # TODO: use np.unique(y, return_counts=True) and compute the formula.
    _, counts = ____
    proportions = ____
    return ____


def entropy(y: np.ndarray) -> float:
    """Compute entropy: H = - Σ p_k log₂(p_k)."""
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    proportions = proportions[proportions > 0]
    return float(-np.sum(proportions * np.log2(proportions)))


# Sanity-check against reference distributions
print("\n--- Gini / Entropy for reference distributions ---")
print(f"{'Distribution':<25} {'Gini':>8} {'Entropy':>8}")
print("-" * 43)
for label, y_demo in [
    ("Pure: [100, 0]", np.array([0] * 100)),
    ("50/50: [50, 50]", np.array([0] * 50 + [1] * 50)),
    ("90/10: [90, 10]", np.array([0] * 90 + [1] * 10)),
    ("70/30: [70, 30]", np.array([0] * 70 + [1] * 30)),
]:
    print(f"  {label:<23} {gini_impurity(y_demo):>8.4f} {entropy(y_demo):>8.4f}")


def best_split_search(X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    """Exhaustive best-split search — returns (feature_idx, threshold, gain)."""
    n = len(y)
    parent_gini = gini_impurity(y)
    best_gain = 0.0
    best_feature_idx = 0
    best_threshold = 0.0
    rng = np.random.default_rng(RANDOM_SEED)

    for feat_idx in range(X.shape[1]):
        sorted_vals = np.unique(X[:, feat_idx])
        if len(sorted_vals) < 2:
            continue
        thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
        if len(thresholds) > 50:
            thresholds = rng.choice(thresholds, 50, replace=False)
        for threshold in thresholds:
            left_mask = X[:, feat_idx] <= threshold
            n_left = int(left_mask.sum())
            n_right = n - n_left
            if n_left == 0 or n_right == 0:
                continue
            # TODO: compute g_left, g_right, weighted_gini, and gain.
            g_left = ____
            g_right = ____
            weighted = ____
            gain = ____
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feat_idx
                best_threshold = float(threshold)
    return best_feature_idx, best_threshold, float(best_gain)


data = build_train_test_split()
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
cv = data["cv"]
feature_names = data["feature_names"]

best_feat_idx, best_threshold, best_gain = best_split_search(X_train, y_train)
best_feat_name = feature_names[best_feat_idx]
print(
    f"\nFrom-scratch best split — feature: {best_feat_name}, "
    f"threshold: {best_threshold:.4f}, gain: {best_gain:.4f}"
)

# Verify against sklearn's depth-1 stump
stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_SEED)
stump.fit(X_train, y_train)
sklearn_feat = feature_names[int(stump.tree_.feature[0])]
print(f"sklearn stump  : feature={sklearn_feat}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert best_gain > 0, "Best split should have positive Gini gain"
assert abs(gini_impurity(np.array([0] * 100))) < 1e-9, "Pure Gini should be 0"
assert abs(gini_impurity(np.array([0] * 50 + [1] * 50)) - 0.5) < 1e-9, "50/50 = 0.5"
print("\n[ok] Checkpoint 1 passed — Gini + best-split verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: sklearn DecisionTreeClassifier with depth tuning
# ════════════════════════════════════════════════════════════════════════

DEPTHS = [2, 3, 5, 7, 10, 15, None]
print("--- max_depth sweep ---")
print(f"{'depth':>8} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 36)
depth_results: dict[str, dict] = {}
for depth in DEPTHS:
    # TODO: build DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED)
    # and call cv_accuracy_f1.
    acc, f1 = ____
    label = str(depth) if depth is not None else "None"
    depth_results[label] = {"depth": depth, "accuracy": acc, "f1": f1}
    print(f"{label:>8} {acc:>14.4f} {f1:>10.4f}")

best_depth_label = max(depth_results, key=lambda d: depth_results[d]["f1"])
best_depth = depth_results[best_depth_label]["depth"]
print(f"\nBest depth: {best_depth_label}")

# TODO: fit_and_evaluate with DecisionTreeClassifier(max_depth=best_depth,
# random_state=RANDOM_SEED). Name: f"DecisionTree (depth={best_depth_label})".
dt_result = ____
dt_model = dt_result["model"]

print(
    f"\n{dt_result['name']}: accuracy={dt_result['accuracy']:.4f} | "
    f"F1={dt_result['f1']:.4f} | AUC={dt_result['auc_roc']:.4f}"
)
print_classification_report(y_test, dt_result["pred"])
print(f"Tree depth: {dt_model.get_depth()}, leaves: {dt_model.get_n_leaves()}")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: tree structure, importance, 2D boundary
# ════════════════════════════════════════════════════════════════════════

print("\n--- Tree structure (first 3 levels) ---")
print(export_text(dt_model, feature_names=feature_names, max_depth=3)[:1200])

importances = sorted(
    zip(feature_names, dt_model.feature_importances_),
    key=lambda pair: pair[1],
    reverse=True,
)
print("\n--- Top feature importances ---")
for name, imp in importances[:10]:
    bar = "#" * int(imp * 50)
    print(f"{name:<30} {imp:>12.4f}  {bar}")

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]
# TODO: fit DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_SEED)
# on X_train_2d and predict over the mesh.
dt_2d = ____
xx, yy = decision_boundary_mesh(X_train_2d)
Z = ____

viz = get_visualizer()
fig = viz.training_history(
    {"importance": [imp for _, imp in importances[:10]]},
    x_label="feature rank (top 10)",
)
fig.update_layout(title="Decision Tree — top 10 feature importances")
out = OUTPUT_DIR / "ex3_04_tree_importance.html"
fig.write_html(str(out))
print(f"\nSaved: {out}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert dt_result["accuracy"] > 0.5
assert Z.shape[0] > 0 and Z.shape[1] > 0
print("[ok] Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: PDPA-friendly interpretable churn
# ════════════════════════════════════════════════════════════════════════
# Every prediction is a walkable rule path. Compliance teams accept
# decision trees as inherently explainable. Limitation: high variance
# and single-tree accuracy ceiling.

# TODO: compute true positives and dollars saved.
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
  [x] Gini and entropy from scratch
  [x] Best-split search verified against sklearn
  [x] max_depth tuned via CV
  [x] Accuracy: {dt_result['accuracy']:.4f}, F1: {dt_result['f1']:.4f}
  [x] Interpretability business case — S${dollars_saved:,.0f} retained

  Next: 05_random_forest.py
"""
)
