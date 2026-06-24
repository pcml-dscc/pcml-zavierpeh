# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3.1: Support Vector Machines (Linear + RBF)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Understand margin maximisation and the soft-margin C parameter
#   - Train a linear SVM and an RBF-kernel SVM with sklearn
#   - Sweep C across orders of magnitude and pick the best via CV
#   - Visualise the RBF decision boundary in 2D PCA space
#   - Translate SVM accuracy into Singapore retail churn dollars saved
#
# PREREQUISITES: MLFP03 Exercise 2 (bias-variance, regularisation)
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — margin maximisation and the kernel trick
#   2. Build — linear + RBF SVM, C parameter sweep
#   3. Train — final RBF SVM on the full training set
#   4. Visualise — 2D decision boundary, support vector overlay
#   5. Apply — Singapore e-commerce churn cost-benefit
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from sklearn.svm import SVC

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
# THEORY — Margin Maximisation and the Kernel Trick
# ════════════════════════════════════════════════════════════════════════
# An SVM finds the hyperplane that separates two classes with the LARGEST
# possible gap (margin) between them. Intuition: a wider gap means any
# small perturbation in the data is less likely to flip a prediction.
#
#   Hard-margin primal:  minimise ||w||² / 2
#                        subject to y_i (w . x_i + b) >= 1 for all i
#
#   Soft-margin: add slack variables ξ_i that let the SVM misclassify
#   points at a cost C per unit of slack.
#     C -> infinity  : hard margin, no misclassification tolerated
#     C -> 0         : very wide margin, many misclassifications tolerated
#
# The KERNEL TRICK lets the SVM learn a nonlinear boundary without ever
# materialising the high-dimensional feature map φ(x). It computes inner
# products in that space via a kernel K(x, x'):
#     Linear : K(x, x') = x . x'
#     RBF    : K(x, x') = exp(-γ ||x - x'||²)
#
# RBF is the default choice for tabular data because it can bend the
# decision surface around arbitrary clusters.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Load data, sweep C for linear and RBF kernels
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 3.1 — Support Vector Machines")
print("=" * 70)

data = build_train_test_split()
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
cv = data["cv"]
feature_names = data["feature_names"]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Churn rate (train): {data['churn_rate']:.2%}")

C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


def sweep_c(kernel: str) -> dict[float, dict[str, float]]:
    """Sweep C for a given kernel and return {C: {accuracy, f1}}."""
    results: dict[float, dict[str, float]] = {}
    print(f"\n--- {kernel.upper()} SVM: C parameter sweep ---")
    print(f"{'C':>10} {'CV Accuracy':>14} {'CV F1':>10}")
    print("-" * 38)
    for c_val in C_VALUES:
        acc, f1 = cv_accuracy_f1(
            SVC(kernel=kernel, C=c_val, random_state=RANDOM_SEED),
            X_train,
            y_train,
            cv,
        )
        results[c_val] = {"accuracy": acc, "f1": f1}
        print(f"{c_val:>10.2f} {acc:>14.4f} {f1:>10.4f}")
    return results


linear_results = sweep_c("linear")
rbf_results = sweep_c("rbf")

best_c_linear = max(linear_results, key=lambda c: linear_results[c]["f1"])
best_c_rbf = max(rbf_results, key=lambda c: rbf_results[c]["f1"])
print(f"\nBest C — linear: {best_c_linear}, RBF: {best_c_rbf}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: final RBF SVM on the full training set
# ════════════════════════════════════════════════════════════════════════

svm_result = fit_and_evaluate(
    SVC(kernel="rbf", C=best_c_rbf, random_state=RANDOM_SEED, probability=True),
    X_train,
    y_train,
    X_test,
    y_test,
    name=f"SVM (RBF, C={best_c_rbf})",
)

print(
    f"\n{svm_result['name']}: trained in {svm_result['train_time']:.2f}s | "
    f"accuracy={svm_result['accuracy']:.4f} | "
    f"F1={svm_result['f1']:.4f} | AUC={svm_result['auc_roc']:.4f}"
)
print_classification_report(y_test, svm_result["pred"])

svm_model = svm_result["model"]
n_sv = int(svm_model.support_vectors_.shape[0])
sv_pct = n_sv / len(y_train)
print(
    f"Support vectors: {n_sv} ({sv_pct:.1%} of training). "
    f"A healthy SVM uses 10-30% of training points."
)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert svm_result["accuracy"] > 0.5, "SVM must beat random"
assert best_c_rbf in C_VALUES, "Best C must come from the sweep"
print("\n[ok] Checkpoint 1 passed — SVM trained and evaluated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: 2D decision boundary + C-sweep curve
# ════════════════════════════════════════════════════════════════════════

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]

svm_2d = SVC(kernel="rbf", C=best_c_rbf, random_state=RANDOM_SEED)
svm_2d.fit(X_train_2d, y_train)

xx, yy = decision_boundary_mesh(X_train_2d)
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

viz = get_visualizer()
# C-sweep as a training-history style plot (accuracy and F1 vs log C)
history = {
    "linear accuracy": [linear_results[c]["accuracy"] for c in C_VALUES],
    "linear F1": [linear_results[c]["f1"] for c in C_VALUES],
    "rbf accuracy": [rbf_results[c]["accuracy"] for c in C_VALUES],
    "rbf F1": [rbf_results[c]["f1"] for c in C_VALUES],
}
fig_sweep = viz.training_history(history, x_label="C value (index)")
fig_sweep.update_layout(title="SVM: linear vs RBF C sweep (CV accuracy / F1)")
sweep_out = OUTPUT_DIR / "ex3_01_svm_c_sweep.html"
fig_sweep.write_html(str(sweep_out))
print(f"Saved: {sweep_out}")

# Decision boundary: export the mesh + decision so the notebook can
# render either via ModelVisualizer or a static matplotlib preview.
print(
    f"Decision mesh shape: {Z.shape} | "
    f"PCA variance captured: {pca_bundle['explained_variance'].sum():.2%}"
)

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert Z.shape[0] > 0 and Z.shape[1] > 0, "Decision boundary mesh is empty"
print("[ok] Checkpoint 2 passed — 2D decision boundary computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore e-commerce churn cost-benefit
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A mid-market Singapore e-commerce marketplace (comparable to
# the SG-listed Shopee/Lazada footprint) has ~250K active customers per
# month. Annual churn baseline is ~22%. The retention team runs a
# targeted promo campaign (S$18 offer) against customers the model
# flags as likely churners.
#
# Why SVM is a reasonable candidate:
#   - Tabular feature space has around 20-30 behavioural features.
#     RBF SVM handles mid-dimensional data very well.
#   - The margin-based decision rule gives a calibrated "how close to
#     the boundary" signal that can prioritise the highest-risk flags.
#   - Training cost is a one-off nightly batch job — O(n²) is tolerable
#     at the subsampled 5K training size.
#
# LIMITATIONS:
#   - RBF SVM is a black box — the retention team cannot explain WHY a
#     customer was flagged. Compliance teams (PDPA, MAS) prefer
#     interpretable models for automated decisions.
#   - Scaling beyond ~50K training samples makes the O(n²) kernel matrix
#     impractical. For the full 250K customer base, move to Random
#     Forest or gradient boosting.

true_positives = int(((svm_result["pred"] == 1) & (y_test == 1)).sum())
dollars_saved = churn_saved_dollars(true_positives)
print(f"\nBusiness impact on held-out test set ({len(y_test)} customers):")
print(f"  True positives (churners caught): {true_positives}")
print(f"  Net retention value at 40% offer acceptance: S${dollars_saved:,.2f}")
print(
    f"  Extrapolated to the 250K monthly active base "
    f"(identical churn rate): "
    f"S${dollars_saved * (250_000 / len(y_test)):,.0f} / month"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Margin maximisation and the C parameter trade-off
  [x] Linear vs RBF kernels and when each is appropriate
  [x] CV-driven C selection across five orders of magnitude
  [x] Held-out accuracy: {svm_result['accuracy']:.4f}, F1: {svm_result['f1']:.4f}
  [x] 2D PCA decision boundary for visual intuition
  [x] Translated classifier output into S${dollars_saved:,.0f} of retained
      customer value on the held-out test fold

  Next: 02_knn.py — instance-based learning with no training phase.
"""
)
