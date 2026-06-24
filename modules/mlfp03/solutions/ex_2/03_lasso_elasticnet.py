# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2.3: Lasso (L1) and ElasticNet
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Fit Lasso at many α values and read the sparsity pattern
#   - Explain L1 geometry: diamond constraint, corner solutions, exact zeros
#   - Use Lasso as built-in feature selection
#   - Blend L1 and L2 with ElasticNet for correlated features
#   - Draw the regularisation path and spot the "kinks" where L1 drops
#     features one by one
#   - Apply sparse selection to a Singapore insurance fraud scorecard
#
# PREREQUISITES:
#   - 02_ridge_regression.py (L2 geometry and Bayesian view)
#
# ESTIMATED TIME: ~40 minutes
#
# TASKS (5-phase R10):
#   1. Theory — the L1 diamond and why it produces zeros
#   2. Build — Lasso + ElasticNet fits across α and l1_ratio
#   3. Train — sparsity trajectory + ElasticNet sweep
#   4. Visualise — regularisation path (||β||₁ and non-zero count vs α)
#   5. Apply — AIA Singapore insurance fraud scorecard feature selection
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error

from kailash_ml import ModelVisualizer

from shared.mlfp03.ex_2 import (
    ALPHAS,
    load_credit_data,
    print_header,
    save_html_plot,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — L1 Regularisation
# ════════════════════════════════════════════════════════════════════════
# Lasso's objective replaces Ridge's squared-norm penalty with an
# absolute-value penalty:
#
#     min_β  ||y - Xβ||²  +  α · ||β||₁
#                             ─────────
#                             Sum of |β_i|
#
# GEOMETRY: ||β||₁ ≤ c traces a DIAMOND (a square rotated 45°). The
# MSE level sets are ellipses — they tend to first touch the diamond
# at a CORNER, and the corners of the diamond are precisely the points
# where some coordinate is zero. Result: Lasso zeroes out features
# exactly, performing built-in feature selection.
#
# NO CLOSED FORM: Because |x| is not differentiable at zero, there's no
# (X'X + αI)⁻¹ formula. Instead sklearn uses coordinate descent, which
# is why you'll see max_iter=10000 in production.
#
# BAYESIAN VIEW: L1 ⇔ Laplace prior P(β) ∝ exp(-|β|/b). The sharp peak
# at zero is what causes coordinates to snap to exactly zero in the MAP
# estimate.
#
# ELASTICNET: Sometimes you want "some sparsity" AND "group correlated
# features together". ElasticNet's objective is
#
#     min_β  ||y - Xβ||²  +  α · ( r·||β||₁ + (1-r)·||β||² )
#
# where r = l1_ratio ∈ [0,1]. r=1 is Lasso, r=0 is Ridge. Most real
# production scorecards use r≈0.5 to get robust sparsity on correlated
# features.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD Lasso + ElasticNet fits
# ════════════════════════════════════════════════════════════════════════

print_header("Lasso & ElasticNet on Singapore Credit Data")

X_train, y_train, X_test, y_test, feature_names = load_credit_data()
print(f"Train: {X_train.shape}  Features: {len(feature_names)}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN Lasso across the α sweep
# ════════════════════════════════════════════════════════════════════════
# Watch the sparsity column: as α grows, more coefficients are driven
# to exactly zero. That's L1 doing feature selection for you.

lasso_results: dict[float, dict[str, float]] = {}
for alpha in ALPHAS:
    lasso = Lasso(alpha=alpha, max_iter=10_000)
    lasso.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, lasso.predict(X_train))
    test_mse = mean_squared_error(y_test, lasso.predict(X_test))
    l1_norm = float(np.sum(np.abs(lasso.coef_)))
    n_zero = int(np.sum(np.abs(lasso.coef_) < 1e-6))
    lasso_results[alpha] = {
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "l1_norm": l1_norm,
        "n_zero": n_zero,
        "coef": lasso.coef_.copy(),
    }

print(
    f"\n{'alpha':>10} {'train MSE':>12} {'test MSE':>12} "
    f"{'||β||₁':>10} {'zeros':>8}  sparsity"
)
print("-" * 62)
for alpha, r in lasso_results.items():
    pct = r["n_zero"] / len(feature_names) * 100
    print(
        f"{alpha:>10.3f} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} "
        f"{r['l1_norm']:>10.4f} {r['n_zero']:>8}  {pct:>5.0f}%"
    )

best_alpha_lasso, best_lasso_row = min(
    lasso_results.items(), key=lambda x: x[1]["test_mse"]
)
print(
    f"\nBest Lasso α = {best_alpha_lasso}  "
    f"(test MSE = {best_lasso_row['test_mse']:.4f}, "
    f"zeros = {best_lasso_row['n_zero']})"
)

# Show the surviving features at the best α
best_lasso_model = Lasso(alpha=best_alpha_lasso, max_iter=10_000).fit(X_train, y_train)
kept = sorted(
    [
        (name, coef)
        for name, coef in zip(feature_names, best_lasso_model.coef_)
        if abs(coef) > 1e-6
    ],
    key=lambda x: abs(x[1]),
    reverse=True,
)

print(f"\nLasso-selected features ({len(kept)}):")
for name, coef in kept[:10]:
    sign = "+" if coef > 0 else "-"
    print(f"  {sign} {name:<34} β = {coef:+.4f}")


# ── Checkpoint 1 ───────────────────────────────────────────────────────
assert (
    lasso_results[100.0]["n_zero"] > lasso_results[0.001]["n_zero"]
), "Higher-α Lasso must zero out MORE coefficients"
print("\n[ok] Checkpoint 1 passed — Lasso sparsity increases with α")
# INTERPRETATION: Compare Lasso's surviving feature list against what
# a domain expert would pick. Lasso often agrees with expert intuition
# because strong signal overwhelms the L1 penalty, while weak /
# redundant features get zeroed first.


# ════════════════════════════════════════════════════════════════════════
# TASK 3b — TRAIN ElasticNet across l1_ratio
# ════════════════════════════════════════════════════════════════════════
# At a fixed α we vary the mix between L1 and L2. At l1_ratio=0 we
# recover Ridge behaviour (no exact zeros). At l1_ratio=1 we recover
# Lasso (maximum sparsity). 0.5 is a common "all-round" choice.

print_header("ElasticNet — Mixing L1 and L2 (α fixed at 0.1)")
en_results: dict[float, dict[str, float]] = {}
for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
    en = ElasticNet(alpha=0.1, l1_ratio=l1_ratio, max_iter=10_000)
    en.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, en.predict(X_train))
    test_mse = mean_squared_error(y_test, en.predict(X_test))
    n_zero = int(np.sum(np.abs(en.coef_) < 1e-6))
    en_results[l1_ratio] = {
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "n_zero": n_zero,
    }

print(f"\n{'l1_ratio':>10} {'train MSE':>12} {'test MSE':>12} {'zeros':>8}")
print("-" * 46)
for l1_ratio, r in en_results.items():
    print(
        f"{l1_ratio:>10.1f} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} "
        f"{r['n_zero']:>8}"
    )


# ── Checkpoint 2 ───────────────────────────────────────────────────────
assert (
    en_results[0.9]["n_zero"] >= en_results[0.1]["n_zero"]
), "l1_ratio=0.9 should zero at least as many coefficients as 0.1"
print("\n[ok] Checkpoint 2 passed — l1_ratio drives sparsity monotonically")
# INTERPRETATION: With CORRELATED features, pure Lasso arbitrarily picks
# one and drops the others — which is unstable. ElasticNet keeps
# correlated groups together with smaller shared coefficients.


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the regularisation path
# ════════════════════════════════════════════════════════════════════════
# The reg-path plot is the single most informative visual in the whole
# exercise. Two curves:
#   1. Lasso ||β||₁ vs α — steps down, with visible "kinks" each time a
#      feature is dropped to zero
#   2. Lasso non-zero count vs α — monotonic staircase toward zero
#
# Save the figures to outputs/mlfp03_ex2_regularisation_cv/.

print_header("Regularisation Path Visualisation")
viz = ModelVisualizer()

fig_l1 = viz.training_history(
    {"||β||₁ (Lasso)": [lasso_results[a]["l1_norm"] for a in ALPHAS]},
    x_label="Regularisation Strength (α)",
)
fig_l1.update_layout(
    title="Lasso: L1 Norm of Coefficients vs α",
    xaxis_type="log",
)
path_l1 = save_html_plot(fig_l1, "lasso_l1_norm_path.html")

fig_sparse = viz.training_history(
    {
        "Non-zero coefs": [
            len(feature_names) - lasso_results[a]["n_zero"] for a in ALPHAS
        ]
    },
    x_label="Regularisation Strength (α)",
)
fig_sparse.update_layout(
    title="Lasso: Sparsity vs α — features dropping to zero",
    xaxis_type="log",
)
path_sparse = save_html_plot(fig_sparse, "lasso_sparsity_path.html")

print(f"\nSaved: {path_l1.name}")
print(f"Saved: {path_sparse.name}")


# ── Checkpoint 3 ───────────────────────────────────────────────────────
coef_matrix = np.array([lasso_results[a]["coef"] for a in ALPHAS])
assert coef_matrix.shape == (
    len(ALPHAS),
    len(feature_names),
), "Coefficient matrix should be (n_alphas, n_features)"
print("\n[ok] Checkpoint 3 passed — regularisation path matrix shape correct")
# INTERPRETATION: Each kink in the path corresponds to a feature being
# zeroed out. This is the L1 diamond geometry in action — corners of the
# diamond are points where some coordinate is exactly zero.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: AIA Singapore Insurance Fraud Scorecard
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: AIA Singapore processes ~1.2M claims/year across life,
# health, and general insurance. The fraud team maintains a claim
# scoring model with 240 candidate features. Regulators (MAS + the
# General Insurance Association) require every production feature to
# be EXPLAINABLE — every β in the scorecard must have a sign-off memo.
#
# WHY LASSO IS THE RIGHT TOOL:
#   - 240 candidate features but only ~40 are "memo-able" without huge
#     documentation overhead. Lasso picks that subset automatically
#     instead of the fraud team running 240 univariate tests.
#   - Lasso's exact zeros mean the production scorecard can DROP
#     features entirely from the data pipeline — fewer upstream ETL
#     dependencies, fewer broken-ingest incidents.
#   - Sign-off memos scale linearly with feature count. 40 features at
#     1 day/memo = 40 days of compliance work; 240 features = 240 days.
#     Lasso saves ~200 compliance-days per annual model refresh.
#
# WHY ELASTICNET FOR A PRODUCTION REFRESH:
#   - Some features are highly correlated (e.g. "claim amount" vs "claim
#     amount relative to policy sum"). Pure Lasso randomly drops one,
#     creating unstable memos between refreshes. ElasticNet with r=0.5
#     keeps correlated groups together with shared smaller weights, so
#     the memo set is stable year on year.
#
# BUSINESS IMPACT (2026 AIA Singapore claims book, S$4.2B annual paid):
#   - Feature-selection compliance savings: ~200 analyst-days × S$1,800
#     = S$360K/year in reduced model governance cost.
#   - Fraud detection uplift from a focused 40-feature model: ~1.8
#     percentage points over a hand-picked 12-feature legacy scorecard,
#     worth ~S$9.1M/year in recovered/prevented fraudulent payouts
#     (assuming a 12% baseline fraud rate on the ~S$420M suspected
#     fraud pool).
#   - Operational savings from dropped ETL: each removed feature saves
#     ~S$4K/year in pipeline cost and one avoided incident per year.
#     200 dropped features ≈ S$800K in annual ETL cost avoided.

print_header("AIA Singapore Fraud Scorecard — Lasso Feature Selection")
print(
    """
Model                      | Features | Annual fraud savings | Compliance cost
---------------------------|----------|----------------------|-----------------
Legacy (hand-picked, 12)   |    12    |         (baseline)   |     S$72K
Lasso (best α, ~40)        |    40    |         +S$9.1M      |    S$240K
Unregularised OLS (240)    |   240    |         +S$9.4M      |    S$1.44M

Net: Lasso delivers ~97% of the fraud-detection uplift at ~17% of
the compliance cost of the unregularised model.
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print(
    """
======================================================================
  WHAT YOU'VE MASTERED
======================================================================

  [x] Lasso objective: ||y-Xβ||² + α·||β||₁
  [x] Diamond geometry → corner solutions → exact zeros
  [x] Laplace prior as the Bayesian twin of L1
  [x] ElasticNet (α, l1_ratio) for correlated-feature stability
  [x] Regularisation path as a diagnostic visual
  [x] Using Lasso for governance-friendly feature selection at AIA

  KEY INSIGHT: Use Lasso when you believe "only a handful of features
  matter". Use Ridge when you believe "everything matters a little".
  Use ElasticNet when you're not sure AND correlated groups exist.

  NEXT: 04_cross_validation.py — we've been picking α by eyeball. Now
  we formalise it with nested CV, time-series CV, and GroupKFold.
"""
)
