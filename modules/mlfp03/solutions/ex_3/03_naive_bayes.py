# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.3: Gaussian Naive Bayes
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Apply Bayes theorem to a classification problem
#   - Understand the "naive" conditional independence assumption
#   - Read class priors and class-conditional means off a GaussianNB
#   - Visualise the Gaussian decision boundary (approximately linear)
#   - Use Naive Bayes as a lightning-fast baseline for Singapore
#     high-volume e-commerce triage
#
# PREREQUISITES: 01_svm.py, MLFP02 Bayesian thinking
#
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — Bayes theorem, conditional independence, Gaussian likelihood
#   2. Build — fit a GaussianNB (no hyperparameters to sweep)
#   3. Train — inspect class priors and class-conditional parameters
#   4. Visualise — 2D decision boundary (Gaussian => quadratic shape)
#   5. Apply — high-volume triage for Singapore marketplace reviews
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
# Bayes theorem, applied to a class y given features x_1, ..., x_n:
#
#     P(y | x_1, ..., x_n)  proportional to  P(y) * prod_i P(x_i | y)
#
# The "naive" step is the independence assumption:
#     P(x_1, x_2, ..., x_n | y) = prod_i P(x_i | y)
# i.e. we pretend every feature is independent of every other feature
# given the class label. This is almost always false in practice (a
# customer's num_orders and total_spend are correlated), yet Naive Bayes
# still produces strong baselines — especially for high-dimensional
# tasks where the decision boundary is mostly linear.
#
# GAUSSIAN Naive Bayes further assumes that each feature's conditional
# distribution P(x_i | y) is a Gaussian with class-specific mean μ_iy and
# variance σ²_iy, which the classifier estimates by maximum likelihood
# from the training data.
#
# Why it is fast: training is a single pass over the data to compute
# per-class means and variances. Prediction is a vectorised log-
# likelihood scoring. No gradient descent, no iterative optimisation.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: fit GaussianNB (no hyperparameters)
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.3 — Gaussian Naive Bayes")
print("=" * 70)

data = build_train_test_split()
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
cv = data["cv"]
feature_names = data["feature_names"]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

nb_result = fit_and_evaluate(
    GaussianNB(),
    X_train,
    y_train,
    X_test,
    y_test,
    name="GaussianNB",
)
nb_model = nb_result["model"]

print(
    f"\n{nb_result['name']}: trained in {nb_result['train_time']:.4f}s | "
    f"accuracy={nb_result['accuracy']:.4f} | "
    f"F1={nb_result['f1']:.4f} | AUC={nb_result['auc_roc']:.4f}"
)
print_classification_report(y_test, nb_result["pred"])

nb_cv_acc, nb_cv_f1 = cv_accuracy_f1(GaussianNB(), X_train, y_train, cv)
print(f"5-fold CV — accuracy: {nb_cv_acc:.4f} | F1: {nb_cv_f1:.4f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN phase: inspect the learned parameters
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
assert nb_result["accuracy"] > 0.5, "GaussianNB must beat random"
assert (
    abs(nb_model.class_prior_[0] + nb_model.class_prior_[1] - 1.0) < 1e-6
), "Priors must sum to 1"
print("\n[ok] Checkpoint 1 passed — GaussianNB trained and parameters inspected\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: 2D decision boundary (quadratic from Gaussians)
# ════════════════════════════════════════════════════════════════════════

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]

nb_2d = GaussianNB()
nb_2d.fit(X_train_2d, y_train)
xx, yy = decision_boundary_mesh(X_train_2d)
Z = nb_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

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
print(
    f"Decision mesh shape: {Z.shape} | "
    f"PCA variance captured: {pca_bundle['explained_variance'].sum():.2%}"
)
print("GaussianNB boundaries are smooth quadratics — a pair of elliptical regions.")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert Z.shape[0] > 0 and Z.shape[1] > 0, "Decision boundary mesh is empty"
print("[ok] Checkpoint 2 passed — GaussianNB 2D boundary computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: high-volume Singapore marketplace triage
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore marketplace wants to score every daily pageview
# with a churn-risk flag in near real time. With ~40 million events per
# day, inference latency and cost both matter.
#
# Why Naive Bayes fits:
#   - Training is O(n): re-train nightly from scratch on the full day's
#     data, no gradient descent loop.
#   - Prediction is a vectorised log-likelihood — microseconds per row.
#   - Baseline accuracy is usually within a few points of more
#     elaborate models, which is often "good enough" for front-line
#     triage.
#   - Transparent: priors and class means are human-readable audit
#     artefacts that compliance teams can sign off.
#
# LIMITATIONS:
#   - Conditional independence is false when features are correlated
#     (num_orders, total_spend, avg_basket). The model systematically
#     double-counts the shared signal.
#   - Gaussian assumption fails on long-tailed counts (orders, dollars).
#     A log-transform or discretisation helps but drifts the model
#     away from "fast and simple".

true_positives = int(((nb_result["pred"] == 1) & (y_test == 1)).sum())
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
  [x] Bayes theorem rewritten as a classifier
  [x] The independence assumption and its failure modes
  [x] Class priors and per-feature Gaussian parameters
  [x] Held-out accuracy: {nb_result['accuracy']:.4f}, F1: {nb_result['f1']:.4f}
  [x] Smooth, quadratic 2D decision boundary
  [x] High-volume triage business case — S${dollars_saved:,.0f} retained

  Next: 04_decision_tree.py — Gini impurity from scratch + sklearn trees.
"""
)
