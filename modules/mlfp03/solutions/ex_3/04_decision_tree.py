# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.4: Decision Trees (Gini from scratch + sklearn)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compute Gini impurity and entropy from scratch in numpy
#   - Simulate the first split of a decision tree by exhaustive search
#   - Verify the from-scratch split matches sklearn's first split
#   - Tune max_depth with cross-validation
#   - Read a tree's feature-importance ranking as an audit artefact
#   - Use decision trees for interpretable Singapore compliance scenarios
#
# PREREQUISITES: 01_svm.py (shared preprocessing), MLFP01 numpy basics
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — Gini impurity, entropy, information gain
#   2. Build — from-scratch Gini + best-split search
#   3. Train — sklearn DecisionTreeClassifier with depth tuning
#   4. Visualise — tree structure, feature importance, 2D boundary
#   5. Apply — interpretable churn flagging for PDPA audit
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
# THEORY — Impurity, Entropy, and Greedy Splitting
# ════════════════════════════════════════════════════════════════════════
# A decision tree partitions the feature space by repeatedly asking
# "which feature, at which threshold, most reduces class impurity?"
#
# Gini impurity (the default sklearn split criterion):
#     G(node) = 1 - Σ p_k²
#         Pure node (all one class): G = 0
#         Binary 50/50:              G = 0.5  (maximum)
#
# Entropy (ID3/C4.5 criterion):
#     H(node) = - Σ p_k log₂(p_k)
#
# Information gain for a candidate split s that partitions node N into
# children C_1, ..., C_m:
#     IG(s) = impurity(N) - Σ_j (|C_j| / |N|) * impurity(C_j)
#
# The tree is GREEDY: at every node it picks the locally best split,
# without backtracking or global optimisation. This is why an ensemble
# of de-correlated trees (Random Forest, 05_random_forest.py) usually
# beats a single tree.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: from-scratch Gini + best-split search
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.4 — Decision Trees")
print("=" * 70)


def gini_impurity(y: np.ndarray) -> float:
    """Gini impurity: G = 1 - Σ p_k² for k classes."""
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    return float(1.0 - np.sum(proportions**2))


def entropy(y: np.ndarray) -> float:
    """Shannon entropy: H = - Σ p_k log₂(p_k)."""
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    proportions = proportions[proportions > 0]
    return float(-np.sum(proportions * np.log2(proportions)))


# Gini for canonical distributions
print("\n--- Gini / Entropy for reference distributions ---")
print(f"{'Distribution':<25} {'Gini':>8} {'Entropy':>8}")
print("-" * 43)
for label, y_demo in [
    ("Pure: [100, 0]", np.array([0] * 100)),
    ("50/50: [50, 50]", np.array([0] * 50 + [1] * 50)),
    ("90/10: [90, 10]", np.array([0] * 90 + [1] * 10)),
    ("70/30: [70, 30]", np.array([0] * 70 + [1] * 30)),
    ("3-class [40,30,30]", np.array([0] * 40 + [1] * 30 + [2] * 30)),
]:
    print(f"  {label:<23} {gini_impurity(y_demo):>8.4f} {entropy(y_demo):>8.4f}")


def best_split_search(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[int, float, float]:
    """Exhaustive best-split search — returns (feature_idx, threshold, gain).

    For each feature, try midpoints between consecutive unique values as
    candidate thresholds, compute the weighted-child Gini, and track the
    split with the largest reduction in parent impurity. Samples at most
    50 thresholds per feature to keep the search tractable.
    """
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
            g_left = gini_impurity(y[left_mask])
            g_right = gini_impurity(y[~left_mask])
            weighted = (n_left / n) * g_left + (n_right / n) * g_right
            gain = parent_gini - weighted
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

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print("\n--- From-scratch best-split search on training data ---")
best_feat_idx, best_threshold, best_gain = best_split_search(X_train, y_train)
best_feat_name = feature_names[best_feat_idx]
print(
    f"  Feature: {best_feat_name}\n"
    f"  Threshold: {best_threshold:.4f}\n"
    f"  Gini gain: {best_gain:.4f}"
)

# Verify against sklearn's depth-1 stump
stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_SEED)
stump.fit(X_train, y_train)
sklearn_feat = feature_names[int(stump.tree_.feature[0])]
sklearn_thresh = float(stump.tree_.threshold[0])
print(f"\nsklearn stump  : feature={sklearn_feat}, threshold={sklearn_thresh:.4f}")
print(
    "  Match"
    if sklearn_feat == best_feat_name
    else "  Close (search samples thresholds; sklearn uses all)"
)

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
depth_results: dict[str, dict[str, float | int | None]] = {}
for depth in DEPTHS:
    acc, f1 = cv_accuracy_f1(
        DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED),
        X_train,
        y_train,
        cv,
    )
    label = str(depth) if depth is not None else "None"
    depth_results[label] = {"depth": depth, "accuracy": acc, "f1": f1}
    print(f"{label:>8} {acc:>14.4f} {f1:>10.4f}")

best_depth_label = max(depth_results, key=lambda d: depth_results[d]["f1"])  # type: ignore[arg-type]
best_depth = depth_results[best_depth_label]["depth"]
print(f"\nBest depth: {best_depth_label}")

dt_result = fit_and_evaluate(
    DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_SEED),
    X_train,
    y_train,
    X_test,
    y_test,
    name=f"DecisionTree (depth={best_depth_label})",
)
dt_model = dt_result["model"]

print(
    f"\n{dt_result['name']}: trained in {dt_result['train_time']:.4f}s | "
    f"accuracy={dt_result['accuracy']:.4f} | "
    f"F1={dt_result['f1']:.4f} | AUC={dt_result['auc_roc']:.4f}"
)
print_classification_report(y_test, dt_result["pred"])
print(f"Tree depth: {dt_model.get_depth()}, leaves: {dt_model.get_n_leaves()}")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: tree structure, feature importance, 2D boundary
# ════════════════════════════════════════════════════════════════════════

print("\n--- Tree structure (first 3 levels) ---")
print(export_text(dt_model, feature_names=feature_names, max_depth=3)[:1500])

importances = sorted(
    zip(feature_names, dt_model.feature_importances_),
    key=lambda pair: pair[1],
    reverse=True,
)
print("\n--- Top feature importances ---")
print(f"{'feature':<30} {'importance':>12}")
print("-" * 44)
for name, imp in importances[:10]:
    bar = "#" * int(imp * 50)
    print(f"{name:<30} {imp:>12.4f}  {bar}")

# 2D decision boundary
pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]
dt_2d = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_SEED)
dt_2d.fit(X_train_2d, y_train)
xx, yy = decision_boundary_mesh(X_train_2d)
Z = dt_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

viz = get_visualizer()
imp_plot = {"importance": [imp for _, imp in importances[:10]]}
fig = viz.training_history(imp_plot, x_label="feature rank (top 10)")
fig.update_layout(title="Decision Tree — top 10 feature importances")
out = OUTPUT_DIR / "ex3_04_tree_importance.html"
fig.write_html(str(out))
print(f"\nSaved: {out}")
print(
    f"Decision mesh shape: {Z.shape} | "
    f"PCA variance captured: {pca_bundle['explained_variance'].sum():.2%}"
)
print("Decision-tree boundaries are AXIS-ALIGNED rectangles — no diagonal lines.")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert dt_result["accuracy"] > 0.5, "Decision tree must beat random"
assert Z.shape[0] > 0 and Z.shape[1] > 0, "Decision boundary mesh is empty"
print("[ok] Checkpoint 2 passed — tree trained and visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: interpretable churn flagging for PDPA audit
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Singapore's PDPA and MAS guidance on automated decisioning
# require a firm to explain any customer-impacting model output. A
# marketing retention model that recommends a S$18 offer is low-risk,
# but a credit or lending use case is not. Regulators accept decision
# trees as inherently interpretable — every prediction walks a sequence
# of feature-threshold rules that a human can read.
#
# Why a decision tree fits:
#   - Every prediction is a path: "recency > 35 days AND cart
#     abandonments > 3 AND support tickets > 0 -> flag for retention".
#   - Audit tool output is the same as the model itself.
#   - Feature importance gives a single-column risk summary.
#
# LIMITATIONS:
#   - High variance: a small change in training data can flip the top
#     split and produce a dramatically different tree.
#   - Depth-tuned trees still underperform ensembles on raw accuracy.
#   - The single-tree recall is capped by axis-aligned splits —
#     diagonal decision surfaces need many steps.

true_positives = int(((dt_result["pred"] == 1) & (y_test == 1)).sum())
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
  [x] Gini impurity and entropy computed from scratch
  [x] Best-split exhaustive search, verified against sklearn
  [x] max_depth tuned via CV
  [x] Held-out accuracy: {dt_result['accuracy']:.4f}, F1: {dt_result['f1']:.4f}
  [x] Tree feature importance as an audit artefact
  [x] Axis-aligned 2D decision boundary
  [x] Interpretability / compliance business case
      — S${dollars_saved:,.0f} retained on the held-out test fold

  Next: 05_random_forest.py — bag many de-correlated trees for OOB.
"""
)
