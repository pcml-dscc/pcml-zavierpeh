# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.3: Gaussian Naive Bayes
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Bayes theorem applied to classification
#   - The "naive" conditional independence assumption
#   - Reading class priors and class-conditional Gaussian parameters
#   - Smooth quadratic decision boundaries
#   - High-volume Singapore marketplace triage as the fit-for-purpose use
#
# ESTIMATED TIME: ~25 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from sklearn.naive_bayes import GaussianNB

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
# THEORY — Bayes Theorem and the Naive Assumption
# ════════════════════════════════════════════════════════════════════════
#   P(y | x_1, ..., x_n)  proportional to  P(y) * prod_i P(x_i | y)
# The independence assumption is almost always false but still produces
# strong baselines. GaussianNB assumes each P(x_i | y) is a Gaussian.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: fit GaussianNB
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.3 — Gaussian Naive Bayes")
print("=" * 70)

data = build_train_test_split()
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
cv = data["cv"]
feature_names = data["feature_names"]

# TODO: fit_and_evaluate(GaussianNB(), X_train, y_train, X_test, y_test,
# name="GaussianNB"). Unpack into nb_result.
nb_result = ____
nb_model = nb_result["model"]

print(
    f"\n{nb_result['name']}: accuracy={nb_result['accuracy']:.4f} | "
    f"F1={nb_result['f1']:.4f} | AUC={nb_result['auc_roc']:.4f}"
)
print_classification_report(y_test, nb_result["pred"])

nb_cv_acc, nb_cv_f1 = cv_accuracy_f1(GaussianNB(), X_train, y_train, cv)
print(f"5-fold CV — accuracy: {nb_cv_acc:.4f} | F1: {nb_cv_f1:.4f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: inspect priors and class-conditional means
# ════════════════════════════════════════════════════════════════════════

print(
    f"\nClass priors: "
    f"P(retained)={nb_model.class_prior_[0]:.4f}, "
    f"P(churned)={nb_model.class_prior_[1]:.4f}"
)
print("\nClass-conditional means (top 10 features):")
print(f"{'Feature':<30} {'Retained':>10} {'Churned':>10} {'|Diff|':>10}")
print("-" * 64)
for i, name in enumerate(feature_names[:10]):
    mu_retained = nb_model.theta_[0, i]
    mu_churned = nb_model.theta_[1, i]
    print(
        f"{name:<30} {mu_retained:>10.4f} {mu_churned:>10.4f} "
        f"{abs(mu_churned - mu_retained):>10.4f}"
    )

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert nb_result["accuracy"] > 0.5
assert (
    abs(nb_model.class_prior_[0] + nb_model.class_prior_[1] - 1.0) < 1e-6
), "Priors must sum to 1"
print("\n[ok] Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: 2D boundary (expect smooth, quadratic)
# ════════════════════════════════════════════════════════════════════════

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]

# TODO: fit GaussianNB on X_train_2d and predict over the mesh.
nb_2d = ____
xx, yy = decision_boundary_mesh(X_train_2d)
Z = ____

viz = get_visualizer()
class_means = {
    "retained": list(nb_model.theta_[0, :10]),
    "churned": list(nb_model.theta_[1, :10]),
}
fig = viz.training_history(class_means, x_label="feature index")
fig.update_layout(title="GaussianNB: class-conditional means (top 10 features)")
out = OUTPUT_DIR / "ex3_03_nb_means.html"
fig.write_html(str(out))
print(f"Saved: {out}")
print(f"Decision mesh shape: {Z.shape}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert Z.shape[0] > 0 and Z.shape[1] > 0
print("[ok] Checkpoint 2 passed — GaussianNB 2D boundary\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: high-volume Singapore marketplace triage
# ════════════════════════════════════════════════════════════════════════
# Why NB: O(n) training, microsecond prediction, transparent priors.
# Limitations: correlated features double-count, long-tailed counts
# break the Gaussian assumption.

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
  [x] Bayes theorem rewritten as a classifier
  [x] Class priors and per-feature Gaussian parameters
  [x] Accuracy: {nb_result['accuracy']:.4f}, F1: {nb_result['f1']:.4f}
  [x] Smooth quadratic 2D boundary
  [x] High-volume business case — S${dollars_saved:,.0f} retained

  Next: 04_decision_tree.py
"""
)
