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
# PREREQUISITES: Exercise 3 (decision trees, Random Forest). Boosting
# extends the same decision-tree primitive into a sequential ensemble.
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — bias vs variance, sequential residual fitting
#   2. Build — from-scratch gradient booster on 1D logistic data
#   3. Train — 10 rounds, watch residuals shrink
#   4. Visualise — decision surface after each round (saved HTML)
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
# Bagging (Random Forest) trains many independent trees on bootstrap
# samples and averages them. This reduces VARIANCE because each tree sees
# a different slice of the data, but every tree makes the SAME kind of
# systematic mistakes as the others (high variance, same bias).
#
# Boosting is the opposite idea: train one tree, look at where it is
# WRONG, train the next tree to correct THAT mistake, repeat. This
# attacks BIAS directly — each new tree is fitted to the residuals
# (the negative gradient of the loss), so the ensemble's systematic
# error shrinks round by round.
#
# The additive-model form:
#     F_0(x) = log-odds of the positive class
#     F_m(x) = F_{m-1}(x) + η · h_m(x)
#
# where h_m is a shallow tree fitted to the pseudo-residuals
#     r_i = y_i - sigmoid(F_{m-1}(x_i))
#
# and η (learning rate) is a step-size that keeps each correction small
# enough that the ensemble does not overshoot. Smaller η → slower but
# more robust convergence. This is the fundamental mechanism behind
# XGBoost, LightGBM, and CatBoost — all three use the same recipe, they
# just differ in how they find the best split at each round.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD a from-scratch gradient booster
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  From-Scratch Gradient Boosting on 1D Logistic Demo")
print("=" * 70)

x_demo, y_demo = make_1d_demo(n=200)
n_demo = len(y_demo)

# Hyperparameters for the demo run
learning_rate = 0.3
n_rounds = 10

# F_0 = log-odds of the positive rate (baseline score)
pos_rate = y_demo.mean()
F = np.full(n_demo, np.log(pos_rate / (1 - pos_rate)))

print(f"\n  Initial F_0 = log-odds = {F[0]:.4f}")
print(f"  Initial sigmoid(F_0) = {1 / (1 + np.exp(-F[0])):.4f}")
print(f"\n  {'Round':>6} {'MSE(resid)':>14} {'Mean|resid|':>14} {'Accuracy':>10}")
print("  " + "─" * 48)

history = []  # collect per-round (round, mse, mean_abs, acc)
for m in range(1, n_rounds + 1):
    # Current probabilities
    p = 1 / (1 + np.exp(-F))

    # Pseudo-residuals = negative gradient of log-loss
    residuals = y_demo - p

    # Fit a shallow tree to the residuals (this is the boosting primitive)
    tree = DecisionTreeRegressor(max_depth=3, random_state=SEED)
    tree.fit(x_demo, residuals)
    h = tree.predict(x_demo)

    # Additive update with learning rate
    F = F + learning_rate * h

    # Metrics
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
# INTERPRETATION: Every round, the MSE of the residuals decreases — the
# ensemble is learning the systematic error and subtracting it out. The
# learning rate 0.3 is aggressive; production boosters use 0.01-0.05 and
# compensate with more rounds.
print("\n[ok] Checkpoint 1 passed — from-scratch gradient boosting converged\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — XGBoost Split-Gain Formula
# ════════════════════════════════════════════════════════════════════════
# XGBoost takes the additive model above and asks a sharper question: for
# a given split, how much does the loss decrease? It uses a second-order
# Taylor expansion:
#
#     L ≈ Σ [g_i · f(x_i) + ½ h_i · f(x_i)²] + Ω(f)
#
# where g_i = ∂L/∂ŷ (first derivative) and h_i = ∂²L/∂ŷ² (second). For
# log-loss on a binary classification target:
#
#     g_i = p_i - y_i           (predicted minus actual)
#     h_i = p_i · (1 - p_i)     (prediction variance)
#
# The split-gain formula falls out algebraically:
#
#     Gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ) ] - γ
#
# where G = Σ g_i, H = Σ h_i over each side. λ penalises large leaf
# weights (L2 regularisation on the tree's output), γ is a fixed cost of
# adding a leaf (pruning threshold). If Gain < 0 the split is refused.

print("\n" + "=" * 70)
print("  XGBoost Split-Gain Derivation — Worked Example")
print("=" * 70)

# Numerical example: a node with 100 defaults + 800 non-defaults
p_pred = 0.12  # current ensemble prediction (≈ population default rate)
g_default = p_pred - 1  # −0.88 per defaulter
g_no_default = p_pred - 0  # +0.12 per non-defaulter
h_per_sample = p_pred * (1 - p_pred)  # 0.1056

# Candidate split: left = 80 defaults + 50 non-defaults, right = the rest
g_left = 80 * g_default + 50 * g_no_default
h_left = 130 * h_per_sample
g_right = 20 * g_default + 750 * g_no_default
h_right = 770 * h_per_sample

gain = xgb_split_gain(g_left, h_left, g_right, h_right, lambda_reg=1.0, gamma=0.0)
print(f"\n  Split gain (λ=1, γ=0): {gain:.4f}")
print(f"    Left:  G_L={g_left:>8.2f}  H_L={h_left:>8.2f}")
print(f"    Right: G_R={g_right:>8.2f}  H_R={h_right:>8.2f}")

# Sensitivity: how λ and γ change the pruning decision
print("\n  --- Regularisation Effect on Split Gain ---")
print(f"  {'λ':>6} {'γ':>6} {'Gain':>10}  {'Decision':<15}")
print("  " + "─" * 44)
for lam in [0.0, 1.0, 10.0, 100.0]:
    for gam in [0.0, 1.0, 5.0]:
        g = xgb_split_gain(g_left, h_left, g_right, h_right, lam, gam)
        decision = "accept" if g > 0 else "PRUNE (Gain<0)"
        print(f"  {lam:>6.1f} {gam:>6.1f} {g:>10.4f}  {decision:<15}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert gain > 0, "A well-separated split should have positive gain"
assert (
    xgb_split_gain(g_left, h_left, g_right, h_right, 100.0, 5.0) < gain
), "Heavy regularisation (λ=100, γ=5) should reduce the gain"
# INTERPRETATION: The gain formula is what makes XGBoost different from
# vanilla gradient boosting. λ limits how much any single leaf can swing
# the prediction; γ introduces a fixed cost for every new leaf. Together
# they force the tree to justify every split against the cost of
# complexity — this is structural regularisation, not just early stopping.
print("\n[ok] Checkpoint 2 passed — XGBoost split gain formula verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the residual shrinkage
# ════════════════════════════════════════════════════════════════════════
# The headline proof of boosting is that the mean absolute residual
# decreases every round. We plot it against the round index; a smooth
# curve shows the additive model is absorbing the signal.

fig = go.Figure()
rounds = [h[0] for h in history]
mean_abs = [h[2] for h in history]
acc_series = [h[3] for h in history]

fig.add_trace(
    go.Scatter(
        x=rounds,
        y=mean_abs,
        mode="lines+markers",
        name="Mean |residual|",
        yaxis="y1",
    )
)
fig.add_trace(
    go.Scatter(
        x=rounds,
        y=acc_series,
        mode="lines+markers",
        name="Accuracy",
        yaxis="y2",
    )
)
fig.update_layout(
    title="From-Scratch Gradient Boosting — Residual Shrinkage per Round",
    xaxis_title="Boosting round",
    yaxis=dict(title="Mean |residual|", side="left"),
    yaxis2=dict(title="Accuracy", overlaying="y", side="right", range=[0, 1]),
    legend=dict(x=0.02, y=0.98),
)
viz_path = OUTPUT_DIR / "ex4_01_residual_shrinkage.html"
fig.write_html(viz_path)
print(f"  Saved: {viz_path}")

# Decision-surface plot: probability vs x after final round
p_final = 1 / (1 + np.exp(-F))
order = np.argsort(x_demo.ravel())
surface = go.Figure()
surface.add_trace(
    go.Scatter(
        x=x_demo.ravel()[order],
        y=p_final[order],
        mode="lines",
        name="Booster P(y=1|x)",
    )
)
surface.add_trace(
    go.Scatter(
        x=x_demo.ravel(),
        y=y_demo,
        mode="markers",
        name="True labels",
        marker=dict(size=6, opacity=0.5),
    )
)
surface.update_layout(
    title="Boosted Probability Surface After 10 Rounds",
    xaxis_title="x",
    yaxis_title="P(default)",
)
surface_path = OUTPUT_DIR / "ex4_01_decision_surface.html"
surface.write_html(surface_path)
print(f"  Saved: {surface_path}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore SME Credit Committee
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore mid-tier bank runs an SME credit committee that
# scores loan applications up to S$500K. The underwriters use an XGBoost
# model to pre-rank applications; the committee then approves or refuses
# the top-scoring 40% each week.
#
# The question that keeps recurring at committee: "The model split on
# 'months since last bounced cheque ≤ 3' and put most of the defaults
# on one side. Why did it refuse to split on 'company age ≤ 2 years'
# even though that also separates the classes?"
#
# ANSWER — the XGBoost gain formula:
#   - 'months since bounced cheque' produces a clean G_L / G_R split with
#     large |G| on each side → high Gain → accepted.
#   - 'company age ≤ 2y' DOES separate the classes, but with only 40
#     applications in the minority leaf. With λ=1, the leaf weight
#     G/(H+λ) gets shrunk toward zero because H is small. The gain
#     formula computes Gain ≈ 0.18, just under γ=0.2 → PRUNED.
#
#   The committee's intuition ("young companies are risky") is correct,
#   but the MODEL refuses to split because the leaf is too small to be
#   reliably informative. This is how structural regularisation prevents
#   overfitting on minority segments.
#
# BUSINESS IMPACT: Singapore SMEs default at ~4-6% on average but the
# specific segment 'company age ≤ 2y AND no audited financials' defaults
# at ~18%. A bank that splits aggressively on this segment will see the
# training AUC climb but the out-of-sample AUC collapse — because the
# 40-row leaf memorises noise. At typical SME portfolio sizes (S$400M
# loans at risk), a 2-point AUC collapse is worth S$3-5M in under-priced
# loans per year.
#
# The γ parameter, tuned on a hold-out set, is literally a monetary
# policy decision: the higher γ, the more the bank refuses to split on
# small segments, the lower the training AUC, the more robust the
# portfolio. This is why γ is tuned with the risk team in the loop.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Explained boosting as sequential residual fitting (bias reduction)
  [x] Implemented a from-scratch gradient booster with shallow trees
  [x] Watched MSE of residuals shrink round by round (the proof)
  [x] Derived the XGBoost split-gain formula from the 2nd-order Taylor
      expansion of log-loss
  [x] Interpreted λ and γ as structural regularisers that prevent the
      tree from memorising small, high-variance segments
  [x] Connected γ to a Singapore SME credit committee's decision to
      refuse splits on minority segments

  KEY INSIGHT: Boosting is just gradient descent in function space. Every
  round, a new tree is the negative gradient direction in a space of
  shallow trees. The gain formula is what makes XGBoost competitive — it
  lets the tree justify every split against a complexity cost, turning
  "build a big tree" into "build only the splits that pay for themselves".

  Next: 02_xgboost.py trains the full XGBoost classifier on real Singapore
  credit data, measures feature importance, and compares it against a
  naive Random Forest baseline.
"""
)
