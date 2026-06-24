# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.1: Boosting Theory (From-Scratch + XGBoost Split Gain)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain why boosting reduces BIAS while bagging reduces VARIANCE
#   - Implement gradient boosting from scratch with shallow decision trees
#   - Derive the XGBoost split-gain formula from a 2nd-order Taylor
#     expansion of the log-loss
#   - Interpret λ (leaf-weight L2) and γ (min split loss) as structural
#     regularisers on trees
#   - Explain when a bank's credit team would refuse a split even if it
#     "looks" informative (the pruning condition: Gain < γ)
#
# PREREQUISITES: Exercise 3 (decision trees, Random Forest).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — bias vs variance, sequential residual fitting
#   2. Build — from-scratch gradient booster on 1D logistic data
#   3. Train — 10 rounds, watch residuals shrink
#   4. Visualise — decision surface after each round
#   5. Apply — Singapore SME credit committee: when to refuse a split
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.tree import DecisionTreeRegressor

from shared.mlfp03.ex_4 import (
    OUTPUT_DIR,
    SEED,
    make_1d_demo,
    xgb_split_gain,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Boosting Works
# ════════════════════════════════════════════════════════════════════════
# Bagging (Random Forest) reduces VARIANCE by averaging independent trees.
# Boosting reduces BIAS by training each new tree on the residuals
# (negative gradient of the loss) of the previous ensemble.
#
#     F_m(x) = F_{m-1}(x) + η · h_m(x)
#
# where h_m is a shallow tree fit to the pseudo-residuals:
#
#     r_i = y_i - sigmoid(F_{m-1}(x_i))
#
# η (learning rate) keeps each correction small. Smaller η = slower,
# more robust. This is the recipe behind XGBoost/LightGBM/CatBoost.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD a from-scratch gradient booster
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  From-Scratch Gradient Boosting on 1D Logistic Demo")
print("=" * 70)

x_demo, y_demo = make_1d_demo(n=200)
n_demo = len(y_demo)

learning_rate = 0.3
n_rounds = 10

# TODO: Initialise F_0 as the log-odds of the positive rate, repeated n_demo times.
# Hint: pos_rate = y_demo.mean(); use np.full + np.log(pos_rate / (1 - pos_rate))
pos_rate = ____
F = ____

print(f"\n  Initial F_0 = log-odds = {F[0]:.4f}")
print(f"\n  {'Round':>6} {'MSE(resid)':>14} {'Mean|resid|':>14} {'Accuracy':>10}")
print("  " + "─" * 48)

history = []
for m in range(1, n_rounds + 1):
    # TODO: Compute current probabilities p from F (use the sigmoid 1/(1+exp(-F)))
    p = ____

    # TODO: Compute pseudo-residuals = negative gradient of log-loss.
    # For log-loss this is y_demo - p.
    residuals = ____

    # TODO: Fit a DecisionTreeRegressor(max_depth=3, random_state=SEED) to
    # (x_demo, residuals) and predict h = tree.predict(x_demo).
    tree = ____
    tree.fit(x_demo, residuals)
    h = ____

    # TODO: Additive update: F = F + learning_rate * h
    F = ____

    p_new = 1 / (1 + np.exp(-F))
    preds = (p_new >= 0.5).astype(int)
    acc = float((preds == y_demo).mean())
    mse_resid = float(np.mean(residuals**2))
    mean_abs_resid = float(np.mean(np.abs(residuals)))
    history.append((m, mse_resid, mean_abs_resid, acc))
    print(f"  {m:>6} {mse_resid:>14.6f} {mean_abs_resid:>14.6f} {acc:>10.4f}")

final_acc = history[-1][3]


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert final_acc > 0.6, "From-scratch boosting should converge above 60% accuracy"
assert history[-1][1] < history[0][1], "MSE of residuals must shrink across rounds"
print("\n[ok] Checkpoint 1 passed — from-scratch gradient boosting converged\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — XGBoost Split-Gain Formula
# ════════════════════════════════════════════════════════════════════════
# XGBoost uses a 2nd-order Taylor expansion of log-loss. For binary:
#
#     g_i = p_i - y_i           (first derivative)
#     h_i = p_i * (1 - p_i)     (second derivative)
#
# Split gain:
#
#     Gain = ½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
#
# λ = L2 on leaf weights, γ = minimum split loss (pruning threshold).

print("\n" + "=" * 70)
print("  XGBoost Split-Gain Derivation — Worked Example")
print("=" * 70)

# Node: 100 defaults + 800 non-defaults, current prediction p=0.12
p_pred = 0.12

# TODO: Compute the per-sample gradients and Hessian.
# g_default = p_pred - 1  (defaulters)
# g_no_default = p_pred - 0  (non-defaulters)
# h_per_sample = p_pred * (1 - p_pred)
g_default = ____
g_no_default = ____
h_per_sample = ____

# Candidate split: left = 80 defaults + 50 non-defaults, right = rest
g_left = 80 * g_default + 50 * g_no_default
h_left = 130 * h_per_sample
g_right = 20 * g_default + 750 * g_no_default
h_right = 770 * h_per_sample

# TODO: Use the shared xgb_split_gain helper with lambda_reg=1.0, gamma=0.0
gain = ____
print(f"\n  Split gain (λ=1, γ=0): {gain:.4f}")

print("\n  --- Regularisation Effect on Split Gain ---")
print(f"  {'λ':>6} {'γ':>6} {'Gain':>10}  {'Decision':<15}")
print("  " + "─" * 44)
for lam in [0.0, 1.0, 10.0, 100.0]:
    for gam in [0.0, 1.0, 5.0]:
        # TODO: call xgb_split_gain(g_left, h_left, g_right, h_right, lam, gam)
        g = ____
        decision = "accept" if g > 0 else "PRUNE (Gain<0)"
        print(f"  {lam:>6.1f} {gam:>6.1f} {g:>10.4f}  {decision:<15}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert gain > 0, "A well-separated split should have positive gain"
assert (
    xgb_split_gain(g_left, h_left, g_right, h_right, 100.0, 5.0) < gain
), "Heavy regularisation should reduce the gain"
print("\n[ok] Checkpoint 2 passed — XGBoost split gain formula verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the residual shrinkage
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
rounds = [h[0] for h in history]
mean_abs = [h[2] for h in history]
acc_series = [h[3] for h in history]

# TODO: Add two line traces to fig — one for mean_abs (yaxis y1) and one
# for acc_series (yaxis y2). Use mode="lines+markers".
____
____

fig.update_layout(
    title="From-Scratch Gradient Boosting — Residual Shrinkage per Round",
    xaxis_title="Boosting round",
    yaxis=dict(title="Mean |residual|", side="left"),
    yaxis2=dict(title="Accuracy", overlaying="y", side="right", range=[0, 1]),
)
viz_path = OUTPUT_DIR / "ex4_01_residual_shrinkage.html"
fig.write_html(viz_path)
print(f"  Saved: {viz_path}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore SME Credit Committee
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore mid-tier bank's SME committee asks why the model
# refused to split on 'company age ≤ 2 years' even though it separates
# the classes. Answer: the minority leaf (40 apps) has small H, so the
# leaf weight G/(H+λ) gets shrunk toward zero — Gain ≈ 0.18, just under
# γ=0.2, so the tree prunes the split.
#
# BUSINESS IMPACT: Singapore SMEs default at ~4-6% on average. A model
# that splits aggressively on 40-row segments memorises noise; on a
# S$400M SME portfolio, a 2-point AUC collapse costs S$3-5M/year in
# under-priced loans. γ is literally a monetary policy dial — higher γ,
# more conservative portfolio.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Boosting = sequential residual fitting (bias reduction)
  [x] From-scratch booster with shallow trees; MSE shrinks round by round
  [x] XGBoost split-gain formula from 2nd-order Taylor expansion
  [x] λ and γ as structural regularisers that prevent noise memorisation
  [x] γ mapped to a Singapore SME committee's "refuse to split" decision

  Next: 02_xgboost.py — full XGBoost on real Singapore credit data.
"""
)
