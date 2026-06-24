# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2.5: Learning Curves and Diagnostic Playbook
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Read a learning curve to decide "more data" vs "better model"
#   - Compare OLS, Ridge, and Lasso learning curves on one dataset
#   - Use learning curves to justify (or reject) data-collection spend
#   - Tie the entire exercise together with a Singapore decision playbook
#
# PREREQUISITES:
#   - 01 through 04 in this exercise
#
# ESTIMATED TIME: ~35 minutes
#
# TASKS (5-phase R10):
#   1. Theory — three learning-curve shapes and what they mean
#   2. Build — three models to compare
#   3. Train — sklearn.learning_curve for each
#   4. Visualise — HTML plots per model
#   5. Apply — StarHub churn-scoring data-acquisition decision
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import learning_curve

from kailash_ml import ModelVisualizer

from shared.mlfp03.ex_2 import (
    load_credit_data,
    print_header,
    save_html_plot,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Reading a Learning Curve
# ════════════════════════════════════════════════════════════════════════
# Three canonical shapes:
#   1. CONVERGED-FAR APART — high bias, more data WON'T help
#   2. CONVERGED-CLOSE     — you're done, marginal returns from data
#   3. NOT-YET-CONVERGED   — high variance, more data WILL help


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD comparison models
# ════════════════════════════════════════════════════════════════════════

print_header("Learning Curves — OLS vs Ridge vs Lasso")
X_train, y_train, X_test, y_test, feature_names = load_credit_data()
print(f"Train: {X_train.shape}")

# TODO: Build a dict mapping a friendly name to a fresh model
# instance. Include unregularised LinearRegression, Ridge(alpha=1.0),
# and Lasso(alpha=0.1, max_iter=10_000).
models = {
    "OLS (unregularised)": ____,
    "Ridge (α=1)": ____,
    "Lasso (α=0.1)": ____,
}

train_sizes_frac = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN each model across growing training-set fractions
# ════════════════════════════════════════════════════════════════════════

all_curves: dict[str, dict[str, np.ndarray]] = {}
for name, model in models.items():
    # TODO: Call sklearn.model_selection.learning_curve with the model,
    # X_train, y_train, train_sizes=train_sizes_frac, cv=5, scoring="r2",
    # n_jobs=-1. It returns (train_sizes, train_scores, test_scores).
    train_sizes, tr_scores, te_scores = ____

    all_curves[name] = {
        "sizes": train_sizes,
        "train_mean": tr_scores.mean(axis=1),
        "test_mean": te_scores.mean(axis=1),
        "train_std": tr_scores.std(axis=1),
        "test_std": te_scores.std(axis=1),
    }

    print_header(name)
    print(f"{'N':>8} {'Train R²':>10} {'Test R²':>10} {'Gap':>10}")
    print("-" * 40)
    for n_, tr, te in zip(train_sizes, tr_scores.mean(axis=1), te_scores.mean(axis=1)):
        print(f"{n_:>8} {tr:>10.4f} {te:>10.4f} {(tr - te):>10.4f}")


# ── Checkpoint 1 ───────────────────────────────────────────────────────
assert len(train_sizes) == len(
    train_sizes_frac
), "Should have one entry per training-size fraction"
assert all(
    "test_mean" in c for c in all_curves.values()
), "Every model should record a test-mean curve"
print("\n[ok] Checkpoint 1 passed — learning curves computed for all models")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the learning curves
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
saved_paths: list[str] = []
for name, curves in all_curves.items():
    # TODO: Build a training_history figure with two series:
    # "{name} — train": curves["train_mean"].tolist()
    # "{name} — test":  curves["test_mean"].tolist()
    # and x_label="Training set size (samples)".
    fig = ____
    fig.update_layout(title=f"Learning Curve — {name}")
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    safe = safe.replace("=", "_").replace("α", "a")
    path = save_html_plot(fig, f"learning_curve_{safe}.html")
    saved_paths.append(path.name)

print("\nSaved learning-curve plots:")
for p in saved_paths:
    print(f"  {p}")


# ── Checkpoint 2 ───────────────────────────────────────────────────────
assert len(saved_paths) == len(models), "One plot per model should be saved"
print("\n[ok] Checkpoint 2 passed — all learning-curve plots written")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: StarHub Mobile Churn Scoring Data Decision
# ════════════════════════════════════════════════════════════════════════
# StarHub must decide whether to buy 18 months of extra CDR data for
# S$1.4M. A learning curve on the current model tells the CTO whether
# more data will pay back:
#   - FLAT curve (converged) → PASS; spend on features instead
#   - RISING curve          → BUY; ~S$1.39M/year retained revenue
#   - HIGH-BIAS curve       → PASS; investigate richer models
#
# Without a learning curve, the data science team either over-claims
# and wastes S$1.4M, or under-claims and foregoes ~S$1.39M/year.

print_header("StarHub Churn Scoring — Learning Curve Data Decision")
print(
    """
Learning curve shape       | Decision                       | S$ impact
---------------------------|--------------------------------|-----------
Converged-close (flat)     | PASS on extra data             |  +S$1.4M
Not-yet-converged (rising) | BUY the extra data             |  +S$1.39M/yr
Converged-far (high bias)  | PASS on data; try richer models|  +S$1.4M
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print(
    """
======================================================================
  WHAT YOU'VE MASTERED (ENTIRE EXERCISE)
======================================================================

  01. Bias-Variance     — why "more complex" is not always "better"
  02. Ridge (L2)        — smooth shrinkage, Gaussian prior, stability
  03. Lasso + ElasticNet — sparsity, L1 diamond, feature selection
  04. Cross-validation  — nested / time-series / group
  05. Learning curves   — diagnose data hunger vs model weakness

  NEXT: Exercise 3 — full supervised model zoo (SVM, KNN, Naive Bayes,
  Trees, Random Forests) on Singapore e-commerce churn data.
"""
)
