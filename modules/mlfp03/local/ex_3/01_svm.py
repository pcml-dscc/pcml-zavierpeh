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
#   4. Visualise — 2D decision boundary
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
# SVM finds the hyperplane with the largest margin between classes.
# Soft-margin: allow misclassification with cost C.
#     C large -> hard margin   C small -> wide margin, tolerant of noise
# Kernel trick: K(x, x') computes inner products in a high-dimensional
# feature space WITHOUT materialising it. RBF is a good default.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Load data, sweep C
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

C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


def sweep_c(kernel: str) -> dict[float, dict[str, float]]:
    """Sweep C for one kernel and return {C: {accuracy, f1}}."""
    results: dict[float, dict[str, float]] = {}
    print(f"\n--- {kernel.upper()} SVM: C parameter sweep ---")
    print(f"{'C':>10} {'CV Accuracy':>14} {'CV F1':>10}")
    print("-" * 38)
    for c_val in C_VALUES:
        # TODO: build an SVC(kernel=kernel, C=c_val, random_state=RANDOM_SEED)
        # and call cv_accuracy_f1(estimator, X_train, y_train, cv).
        # Store {"accuracy": acc, "f1": f1} into results[c_val].
        acc, f1 = ____
        results[c_val] = ____
        print(f"{c_val:>10.2f} {acc:>14.4f} {f1:>10.4f}")
    return results


linear_results = sweep_c("linear")
rbf_results = sweep_c("rbf")

# TODO: pick the C with the highest F1 for each kernel (hint: use max with
# a key=lambda lookup into the dict).
best_c_linear = ____
best_c_rbf = ____
print(f"\nBest C — linear: {best_c_linear}, RBF: {best_c_rbf}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: final RBF SVM on the full training set
# ════════════════════════════════════════════════════════════════════════

# TODO: Fit SVC(kernel="rbf", C=best_c_rbf, random_state=RANDOM_SEED,
# probability=True) via fit_and_evaluate(...).
svm_result = ____

print(
    f"\n{svm_result['name']}: trained in {svm_result['train_time']:.2f}s | "
    f"accuracy={svm_result['accuracy']:.4f} | "
    f"F1={svm_result['f1']:.4f} | AUC={svm_result['auc_roc']:.4f}"
)
print_classification_report(y_test, svm_result["pred"])

svm_model = svm_result["model"]
n_sv = int(svm_model.support_vectors_.shape[0])
print(f"Support vectors: {n_sv} ({n_sv / len(y_train):.1%} of training)")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert svm_result["accuracy"] > 0.5, "SVM must beat random"
assert best_c_rbf in C_VALUES, "Best C must come from the sweep"
print("\n[ok] Checkpoint 1 passed — SVM trained and evaluated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: 2D decision boundary + C sweep curve
# ════════════════════════════════════════════════════════════════════════

pca_bundle = project_2d(X_train, X_test)
X_train_2d = pca_bundle["X_train_2d"]

# TODO: Fit SVC(kernel="rbf", C=best_c_rbf, random_state=RANDOM_SEED)
# on X_train_2d (the PCA-projected data) and store in svm_2d.
svm_2d = ____

xx, yy = decision_boundary_mesh(X_train_2d)
# TODO: predict over the mesh (np.c_[xx.ravel(), yy.ravel()]) and reshape
# into xx.shape. Store in Z.
Z = ____

viz = get_visualizer()
history = {
    "linear accuracy": [linear_results[c]["accuracy"] for c in C_VALUES],
    "linear F1": [linear_results[c]["f1"] for c in C_VALUES],
    "rbf accuracy": [rbf_results[c]["accuracy"] for c in C_VALUES],
    "rbf F1": [rbf_results[c]["f1"] for c in C_VALUES],
}
fig = viz.training_history(history, x_label="C value (index)")
fig.update_layout(title="SVM: linear vs RBF C sweep (CV accuracy / F1)")
out = OUTPUT_DIR / "ex3_01_svm_c_sweep.html"
fig.write_html(str(out))
print(f"Saved: {out}")
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
# SCENARIO: 250K MAU Singapore marketplace, retention team runs a S$18
# offer against flagged churners. Targeted promo acceptance rate ~40%.
# Why SVM: margin-based signal, mid-dimensional feature space, nightly
# batch job is acceptable. Limitations: black-box, O(n²) at train time.

# TODO: count true positives (pred==1 AND y_test==1) and compute
# churn_saved_dollars(true_positives).
true_positives = ____
dollars_saved = ____
print(f"\nBusiness impact on held-out test set ({len(y_test)} customers):")
print(f"  True positives: {true_positives}")
print(f"  Net retention value: S${dollars_saved:,.2f}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Margin maximisation and the C parameter trade-off
  [x] Linear vs RBF kernels and CV-driven C selection
  [x] Held-out accuracy: {svm_result['accuracy']:.4f}, F1: {svm_result['f1']:.4f}
  [x] 2D PCA decision boundary for visual intuition
  [x] Translated classifier output into S${dollars_saved:,.0f} of retained value

  Next: 02_knn.py — instance-based learning with no training phase.
"""
)
