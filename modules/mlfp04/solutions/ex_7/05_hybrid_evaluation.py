# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 7.5: Hybrid Recommender + Full Evaluation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Blend four recommenders into a hybrid via MAP-weighted averaging
#   - Compare every method on RMSE, coverage, precision@k, and MAP
#   - Understand why ranking metrics matter more than RMSE in production
#   - Explain implicit vs explicit feedback and when each applies
#   - See how the Netflix Prize winner used 107 blended models
#
# PREREQUISITES: Exercises 7.1, 7.2, 7.3, 7.4
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why blends beat any single model
#   2. Build — re-run all four base recommenders and the hybrid blender
#   3. Train — no training; this is evaluation + blending
#   4. Visualise — side-by-side method comparison (RMSE + MAP)
#   5. Apply — Singapore news aggregator front-page personalisation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_7 import (
    N_ITEMS,
    N_USERS,
    build_rating_dataset,
    holdout_rmse,
    mean_average_precision,
    precision_at_k,
    save_html,
)

K_LATENT = 10
LAMBDA_REG = 0.1
N_ITERATIONS = 50
K_NEIGHBOURS = 20


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why hybrid beats any single method
# ════════════════════════════════════════════════════════════════════════
# Every recommender in this exercise has a failure mode:
#   - Content-based: cold-start items, filter bubble
#   - User-CF: O(N^2) compute, cold-start users
#   - Item-CF: niche-item sparsity
#   - ALS MF: popularity bias, cold-start everything
#
# No single method is best for every user. Some users have rich histories
# (CF wins). Some items are brand new (content-based wins). Some taste
# clusters are well-captured in the latent space (MF wins).
#
# A hybrid recommender averages the predictions from multiple methods.
# The blending weights can be static (equal), learned (stacking), or
# performance-weighted (by MAP, as we do here). The Netflix Prize winner
# used a stacked ensemble of 107 different models.
#
# RMSE vs ranking metrics:
#   RMSE measures rating prediction accuracy. Precision@k and MAP measure
#   whether the TOP recommendations are actually good. For production
#   systems, ranking matters more — users never see predictions 10-50.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: rerun the four base models and the blender
# ════════════════════════════════════════════════════════════════════════
# We import the four base algorithms from the per-technique files via
# the shared module OR inline the logic. To keep this file standalone we
# inline the algorithms here (they're short) so the file runs top-to-
# bottom without cross-file state.


def content_based_predict(R, item_feats, obs_mask):
    n_users, n_items = R.shape
    preds = np.full((n_users, n_items), np.nan)
    for u in range(n_users):
        rated = np.where(obs_mask[u])[0]
        if len(rated) == 0:
            continue
        ratings_u = np.nan_to_num(R[u, rated], nan=0.0)
        profile = (ratings_u[:, None] * item_feats[rated]).sum(axis=0)
        pn = np.linalg.norm(profile)
        if pn < 1e-10:
            continue
        profile /= pn
        for j in range(n_items):
            fn = np.linalg.norm(item_feats[j])
            if fn < 1e-10:
                continue
            sim = profile @ item_feats[j] / fn
            preds[u, j] = 1.0 + (sim + 1.0) * 2.0
    return preds


def user_cf_predict(R, obs_mask, k=K_NEIGHBOURS):
    n_users, n_items = R.shape
    user_means = np.array(
        [
            float(np.nanmean(R[u, obs_mask[u]])) if obs_mask[u].any() else 0.0
            for u in range(n_users)
        ]
    )
    Rc = R.copy()
    for u in range(n_users):
        Rc[u, obs_mask[u]] -= user_means[u]
    Rc[~obs_mask] = 0.0
    sim = np.zeros((n_users, n_users))
    for u in range(n_users):
        for v in range(u, n_users):
            both = obs_mask[u] & obs_mask[v]
            if not both.any():
                continue
            ru, rv = Rc[u, both], Rc[v, both]
            d = np.linalg.norm(ru) * np.linalg.norm(rv)
            if d < 1e-10:
                continue
            s = float(ru @ rv / d)
            sim[u, v] = s
            sim[v, u] = s
    preds = np.full((n_users, n_items), np.nan)
    for u in range(n_users):
        s = sim[u].copy()
        s[u] = -np.inf
        top = np.argsort(s)[-k:]
        top = top[s[top] > 0]
        if len(top) == 0:
            continue
        for j in range(n_items):
            rated_n = top[obs_mask[top, j]]
            if len(rated_n) == 0:
                continue
            w = sim[u, rated_n]
            denom = np.abs(w).sum()
            if denom < 1e-10:
                continue
            preds[u, j] = (
                user_means[u] + (w @ (R[rated_n, j] - user_means[rated_n])) / denom
            )
    return np.clip(preds, 1.0, 5.0)


def item_cf_predict(R, obs_mask, k=K_NEIGHBOURS):
    n_users, n_items = R.shape
    item_means = np.array(
        [
            float(np.nanmean(R[obs_mask[:, j], j])) if obs_mask[:, j].any() else 0.0
            for j in range(n_items)
        ]
    )
    Rc = R.copy()
    for j in range(n_items):
        Rc[obs_mask[:, j], j] -= item_means[j]
    Rc[~obs_mask] = 0.0
    sim = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(i, n_items):
            both = obs_mask[:, i] & obs_mask[:, j]
            if not both.any():
                continue
            ri, rj = Rc[both, i], Rc[both, j]
            d = np.linalg.norm(ri) * np.linalg.norm(rj)
            if d < 1e-10:
                continue
            s = float(ri @ rj / d)
            sim[i, j] = s
            sim[j, i] = s
    preds = np.full((n_users, n_items), np.nan)
    for u in range(n_users):
        rated = np.where(obs_mask[u])[0]
        if len(rated) == 0:
            continue
        for j in range(n_items):
            sims = sim[j, rated]
            top_idx = np.argsort(sims)[-k:] if len(sims) > k else np.arange(len(sims))
            pos = top_idx[sims[top_idx] > 0]
            if len(pos) == 0:
                continue
            w = sims[pos]
            denom = np.abs(w).sum()
            if denom < 1e-10:
                continue
            preds[u, j] = (w @ R[u, rated[pos]]) / denom
    return np.clip(preds, 1.0, 5.0)


def als_predict(R, obs_mask, k, lam, n_iter, rng):
    n_users, n_items = R.shape
    U = rng.normal(0, 0.1, size=(n_users, k))
    V = rng.normal(0, 0.1, size=(n_items, k))
    R_safe = np.nan_to_num(R, nan=0.0)
    identity = lam * np.eye(k)
    for _ in range(n_iter):
        for u in range(n_users):
            rated = np.where(obs_mask[u])[0]
            if len(rated) == 0:
                continue
            V_u = V[rated]
            A = V_u.T @ V_u + identity
            b = V_u.T @ R_safe[u, rated]
            U[u] = np.linalg.solve(A, b)
        for j in range(n_items):
            raters = np.where(obs_mask[:, j])[0]
            if len(raters) == 0:
                continue
            U_j = U[raters]
            A = U_j.T @ U_j + identity
            b = U_j.T @ R_safe[raters, j]
            V[j] = np.linalg.solve(A, b)
    return np.clip(U @ V.T, 1.0, 5.0)


def blend_hybrid(all_preds: dict, eval_results: dict) -> np.ndarray:
    """MAP-weighted blend of multiple prediction matrices."""
    map_scores = {name: max(r["MAP"], 0.01) for name, r in eval_results.items()}
    total = sum(map_scores.values())
    weights = {name: m / total for name, m in map_scores.items()}

    print("\nBlending weights (MAP-based):")
    for name, w in weights.items():
        print(f"  {name:<18} {w:.3f}")

    def normalise(p):
        valid = ~np.isnan(p)
        out = p.copy()
        if valid.sum() > 0:
            pmin, pmax = np.nanmin(p), np.nanmax(p)
            if pmax > pmin:
                out[valid] = (p[valid] - pmin) / (pmax - pmin)
        return out

    hybrid = np.zeros((N_USERS, N_ITEMS))
    for name, preds in all_preds.items():
        norm = np.nan_to_num(normalise(preds), nan=0.5)
        hybrid += weights[name] * norm

    hybrid = hybrid * 4 + 1  # rescale [0,1] back to [1,5]
    return np.clip(hybrid, 1.0, 5.0)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Evaluation run
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Hybrid Recommender + Full Evaluation")
print("=" * 70)

data = build_rating_dataset()
R_train = data["R_train"]
R_observed = data["R_observed"]
train_mask = data["train_mask"]
holdout_mask = data["holdout_mask"]
item_features = data["item_features"]

print("\nRunning base recommenders...")
cb = content_based_predict(R_train, item_features, train_mask)
ucf = user_cf_predict(R_train, train_mask)
icf = item_cf_predict(R_train, train_mask)
als = als_predict(R_train, train_mask, K_LATENT, LAMBDA_REG, N_ITERATIONS, data["rng"])

all_predictions = {
    "Content-Based": cb,
    "User-CF": ucf,
    "Item-CF": icf,
    "ALS MF": als,
}

print(f"\n{'Method':<18} {'RMSE':>8} {'Coverage':>10} {'P@5':>8} {'MAP':>8}")
print("─" * 54)
eval_results: dict = {}
for name, preds in all_predictions.items():
    rmse, cov = holdout_rmse(preds, R_observed, holdout_mask)
    p5 = precision_at_k(preds, R_observed, holdout_mask, k=5)
    m = mean_average_precision(preds, R_observed, holdout_mask)
    eval_results[name] = {"RMSE": rmse, "Coverage": cov, "P@5": p5, "MAP": m}
    print(f"{name:<18} {rmse:>8.4f} {cov:>9.1%} {p5:>8.4f} {m:>8.4f}")

hybrid_preds = blend_hybrid(all_predictions, eval_results)
hybrid_rmse, hybrid_cov = holdout_rmse(hybrid_preds, R_observed, holdout_mask)
hybrid_p5 = precision_at_k(hybrid_preds, R_observed, holdout_mask, k=5)
hybrid_map = mean_average_precision(hybrid_preds, R_observed, holdout_mask)
eval_results["Hybrid"] = {
    "RMSE": hybrid_rmse,
    "Coverage": hybrid_cov,
    "P@5": hybrid_p5,
    "MAP": hybrid_map,
}
print(
    f"{'Hybrid':<18} {hybrid_rmse:>8.4f} {hybrid_cov:>9.1%} "
    f"{hybrid_p5:>8.4f} {hybrid_map:>8.4f}"
)

best_single_map = max(r["MAP"] for n, r in eval_results.items() if n != "Hybrid")
lift = hybrid_map - best_single_map
print(f"\nBest single-method MAP: {best_single_map:.4f}")
print(f"Hybrid MAP lift:        {lift:+.4f}")
print(
    "\nNote: on this 100x50 synthetic matrix the simple MAP-weighted blend "
    "may underperform the single best method — the blend dilutes the winner "
    "with weaker models. In production with millions of users, uneven "
    "per-user performance makes blends reliably win. The Netflix Prize "
    "winner used a LEARNED stacking model (not static weights) over 107 "
    "base recommenders."
)


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(eval_results) == 5, "Should have evaluated 4 base methods + hybrid"
assert hybrid_rmse > 0, "Hybrid RMSE should be positive"
assert 0.0 <= hybrid_p5 <= 1.0, "Precision@k must be in [0, 1]"
assert 0.0 <= hybrid_map <= 1.0, "MAP must be in [0, 1]"
print("\n[ok] Checkpoint passed — all five methods evaluated on RMSE, P@5, MAP\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: side-by-side comparison
# ════════════════════════════════════════════════════════════════════════
# A metric comparison chart makes trade-offs obvious — ALS may win RMSE
# while Item-CF wins MAP, and the Hybrid sits above everyone on at least
# one dimension.

viz = ModelVisualizer()
comparison_metrics = {
    name: {"RMSE": r["RMSE"], "MAP": r["MAP"]} for name, r in eval_results.items()
}
fig_cmp = viz.metric_comparison(comparison_metrics)
fig_cmp.update_layout(title="Recommender Method Comparison (RMSE vs MAP)")
save_html(fig_cmp, "05_method_comparison.html")


# ════════════════════════════════════════════════════════════════════════
# IMPLICIT vs EXPLICIT FEEDBACK — conceptual summary
# ════════════════════════════════════════════════════════════════════════
print(
    """
Explicit feedback: users provide ratings (1-5 stars).
  + clear signal
  - sparse (most users rate few items)
  - absence of rating is ambiguous

Implicit feedback: clicks, views, purchases, time spent.
  + abundant
  - noisy (click != like)
  - no NEGATIVE signal (no click could mean "never seen")

ALS for implicit data (Hu et al. 2008):
  - Treat ALL pairs as observed
  - p[u,i] = 1 if user interacted, 0 otherwise
  - c[u,i] = 1 + alpha * count  (confidence)
  - Loss: sum_all c(u,i) * (p(u,i) - U[u] V[i])^2 + lambda * (||U||^2 + ||V||^2)

Both methods dominate production systems: Spotify, Netflix, YouTube.
"""
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore News Aggregator Front-Page Personalisation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore news aggregator (think CNA, Mothership, or a
# pan-ASEAN equivalent) runs personalised front pages for ~3M daily
# active users. The front page has 8 slots. Every refresh must rank
# hundreds of articles by predicted relevance in <50ms.
#
# Why a HYBRID is the right tool:
#   - BREAKING NEWS = cold-start items (0 reads). Content-based kicks in
#     using tags, section, author, entity mentions
#   - REGULAR CONTENT = CF wins via co-read patterns
#   - DEEP TASTE = ALS captures "reads Opinion + Tech + Weekend" clusters
#     no single feature could describe
#   - A weighted blend picks the right lever per (user, article) pair
#
# BUSINESS IMPACT: Industry data from SEA news publishers shows a 15-25%
# lift in clicks-per-session from personalised front pages. On a 3M DAU
# aggregator with S$35M annual ad revenue, a 20% click lift translates to
# roughly S$7M/year in incremental ad impressions — vs ~S$400K/year in
# ML infra + engineering. 17x ROI, and the hybrid is the reason: every
# single-method alternative underperforms on at least one user segment.
#
# THE NETFLIX PRIZE: The winning entry was a blend of 107 models. The
# Netflix team famously said "no single model comes close." The same is
# true in production ML generally: ensemble over one great model wins.
#
# LIMITATIONS of simple MAP-weighted blending:
#   - Weights are global; a per-user gating network (learn which method
#     to trust for WHICH user) is strictly better
#   - Training a meta-model (stacking) can recover another 5-10% MAP
#   - At scale, even this 4-method blend costs 4x inference — in practice
#     teams distil the ensemble into a single neural model (Ex 8 preview)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Evaluated four recommenders on RMSE, coverage, P@5, and MAP
  [x] Blended them via MAP-weighted averaging into a hybrid
  [x] Measured the hybrid lift vs best single method ({lift:+.4f} MAP)
  [x] Explained implicit vs explicit feedback and ALS-implicit
  [x] Identified a S$7M/year SEA news scenario where hybrids dominate

  KEY INSIGHT: No single recommender wins on every user. Hybrids lift
  ranking quality by routing each prediction through the method best
  suited to that user-item pair.

  Exercise 7 complete — you now understand the full recommender stack.

  Next module — MLFP05 deep learning: neural networks generalise matrix
  factorisation by adding non-linear activations. The hidden layer IS
  an embedding, learned by the same principle.
"""
)
