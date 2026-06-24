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
#   - Recognise the three canonical shapes: converged-far, converged-close,
#     and not-yet-converged
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
#   2. Build — three models to compare (OLS, Ridge, Lasso)
#   3. Train — sklearn.learning_curve for each
#   4. Visualise — HTML plots for each model's train-vs-test curve
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
# A learning curve plots model performance (train and validation) as a
# function of the training-set size. Three canonical shapes exist:
#
#   1. CONVERGED-FAR APART  — train much higher than test, both flat
#      Diagnosis: HIGH BIAS. More data WON'T help; you need a richer
#      model or more features. This is classic underfitting.
#
#   2. CONVERGED-CLOSE       — train and test curves meet at a good score
#      Diagnosis: You're done. Additional data gives marginal returns.
#
#   3. NOT-YET-CONVERGED    — test still rising as training size grows
#      Diagnosis: HIGH VARIANCE. More data WILL help. Invest in data
#      collection OR regularise harder.
#
# The gap between train and test curves is the VARIANCE component; the
# absolute level of the train curve upper-bounds the achievable bias.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the comparison models
# ════════════════════════════════════════════════════════════════════════

print_header("Learning Curves — OLS vs Ridge vs Lasso")
X_train, y_train, X_test, y_test, feature_names = load_credit_data()
print(f"Train: {X_train.shape}")

models = {
    "OLS (unregularised)": LinearRegression(),
    "Ridge (α=1)": Ridge(alpha=1.0),
    "Lasso (α=0.1)": Lasso(alpha=0.1, max_iter=10_000),
}

train_sizes_frac = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN each model across growing training-set fractions
# ════════════════════════════════════════════════════════════════════════

all_curves: dict[str, dict[str, np.ndarray]] = {}
for name, model in models.items():
    train_sizes, tr_scores, te_scores = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=train_sizes_frac,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
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
# TASK 4 — VISUALISE the three learning curves
# ════════════════════════════════════════════════════════════════════════
# Save one HTML per model: train vs test R² as a function of training-
# set size. These plots are the single best diagnostic you can show a
# credit committee when asking for a data-collection budget.

viz = ModelVisualizer()
saved_paths: list[str] = []
for name, curves in all_curves.items():
    fig = viz.training_history(
        {
            f"{name} — train": curves["train_mean"].tolist(),
            f"{name} — test": curves["test_mean"].tolist(),
        },
        x_label="Training set size (samples)",
    )
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
# INTERPRETATION:
#   - If OLS's test curve is still RISING at full size, more data will
#     improve OLS. Regularised models should converge earlier.
#   - If Ridge's test curve is FLAT and the gap is tiny, Ridge has
#     reached the ceiling; more data won't help, but a richer model
#     (kernel, tree) might.
#   - If Lasso's test curve is above Ridge at small N, Lasso's feature
#     selection is helping when data is scarce.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: StarHub Mobile Churn Scoring Data Decision
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: StarHub (Singapore telco, ~2.2M mobile subscribers) is
# deciding whether to buy an additional 18 months of historical CDR
# (call-detail-record) data from a partner carrier. The data would
# expand the churn-model training set from ~180K labelled subscribers
# to ~420K, at a cost of S$1.4M for the licence + integration.
#
# WHY LEARNING CURVES ARE THE RIGHT TOOL:
#   - The CTO needs a defensible answer: "will another S$1.4M of data
#     actually reduce churn?" A learning curve gives a quantitative
#     yes/no BEFORE the cheque is written.
#   - If the current churn model's test curve is FLAT (converged-close),
#     the S$1.4M won't pay back — the model has already extracted all
#     the signal its feature set can express. Spend the money on
#     BETTER features instead.
#   - If the curve is STILL RISING (not-yet-converged), the incremental
#     data will lift test R² by an estimable amount, which translates
#     directly into retained-revenue projections.
#
# CALCULATION (illustrative numbers):
#   - Current monthly churn: 1.4% × 2.2M subscribers × S$42 ARPU
#     = S$1.29M/month of recurring revenue lost to churn.
#   - Learning curve extrapolation shows adding 240K labelled rows lifts
#     test AUC by +0.028, which the retention team translates into a
#     relative churn reduction of ~9% in the top-decile risk cohort.
#   - Revenue saved: 9% × S$1.29M × 12 months = ~S$1.39M/year.
#   - Payback: S$1.4M data cost / S$1.39M annual savings ≈ 12 months.
#     Green-light the purchase.
#
# COUNTERFACTUAL: Without the learning curve, the data science team
# would either (a) over-claim and buy the data even if the model was
# already converged, wasting S$1.4M, or (b) under-claim and pass on
# the data, foregoing the S$1.39M/year saving. Either way, the wrong
# decision costs ~S$1.4M+. The learning curve is the S$0 diagnostic
# that anchors the ~S$1.4M decision.

print_header("StarHub Churn Scoring — Learning Curve Data Decision")
print(
    """
Learning curve shape       | Decision                       | S$ impact
---------------------------|--------------------------------|-----------
Converged-close (flat)     | PASS on extra data; buy        |  +S$1.4M
                           | features instead               |  saved
---------------------------|--------------------------------|-----------
Not-yet-converged (rising) | BUY the extra data             |  +S$1.39M/yr
                           |                                |  recurring
---------------------------|--------------------------------|-----------
Converged-far (high bias)  | PASS on data; investigate      |  +S$1.4M
                           | non-linear models or           |  saved +
                           | cross-product features         |  redirected
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION — The Full Exercise in One Frame
# ════════════════════════════════════════════════════════════════════════
print(
    """
======================================================================
  WHAT YOU'VE MASTERED (ENTIRE EXERCISE)
======================================================================

  01. Bias-Variance     — why "more complex" is not always "better"
  02. Ridge (L2)        — smooth shrinkage, Gaussian prior, stability
  03. Lasso + ElasticNet — sparsity, L1 diamond, feature selection
  04. Cross-validation  — nested / time-series / group, match deployment
  05. Learning curves   — diagnose data hunger vs model weakness

  DECISION PLAYBOOK FOR A SINGAPORE ML TEAM:
    Step 1. Start with a learning curve on a Ridge baseline.
    Step 2. If the curve is still rising, invest in more data AND use
            nested CV to pick α honestly.
    Step 3. If the curve is flat but the gap is large, switch to a
            richer model class (not just more data).
    Step 4. If the data has temporal or group structure, DO NOT use
            shuffled k-fold — use TimeSeriesSplit or GroupKFold.
    Step 5. For governance-critical work, prefer Lasso/ElasticNet so
            feature selection is auditable.

  KEY INSIGHT: Regularisation is how you encode your prior belief about
  the world. Cross-validation is how you audit that belief against
  reality. Learning curves tell you whether reality will change if you
  throw more data at it.

  NEXT: Exercise 3 — full supervised model zoo (SVM, KNN, Naive Bayes,
  Trees, Random Forests) on Singapore e-commerce churn data.
"""
)
