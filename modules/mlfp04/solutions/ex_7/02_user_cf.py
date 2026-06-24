# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 7.2: User-Based Collaborative Filtering
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compute pairwise user similarity on mean-centred ratings
#   - Borrow preferences from the top-k most similar users
#   - Understand why mean-centring fixes "generous rater" bias
#   - See the failure modes: scale, sparsity, and cold-start users
#
# PREREQUISITES: Exercise 7.1 (content-based baseline)
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — "users who agreed with me before will agree with me again"
#   2. Build — user similarity matrix + top-k CF prediction
#   3. Train — no training loop; neighbourhoods are looked up at inference
#   4. Visualise — user similarity heatmap + neighbour quality
#   5. Apply — Singapore streaming platform watchlist expansion
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.express as px

from shared.mlfp04.ex_7 import (
    N_USERS,
    build_rating_dataset,
    holdout_rmse,
    print_method_scores,
    save_html,
)

K_NEIGHBOURS = 20


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why user-based CF works
# ════════════════════════════════════════════════════════════════════════
# Core intuition: if Alice and Bob agreed on 50 previous movies, Bob's
# opinion on a new movie is a strong signal for Alice's taste. Find Alice's
# nearest neighbours in rating space, then predict the new movie by a
# weighted average of those neighbours' ratings.
#
# Why mean-centring matters:
#   Generous raters give everything 4-5 stars; tough raters give 2-3.
#   Without centring, they look dissimilar even when they rank items the
#   same way. Subtracting each user's mean removes the bias and compares
#   RANKINGS, not absolute scores.
#
# Why top-k:
#   - Averaging over all users drowns out signal with noise
#   - Top-k focuses on the most reliable neighbours (20-50 typical)
#
# STRENGTHS: captures community taste that content features miss
# WEAKNESSES: O(N^2) similarity compute, cold-start users still broken


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the user similarity matrix + CF predictor
# ════════════════════════════════════════════════════════════════════════


def user_similarity_matrix(
    R: np.ndarray, obs_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise cosine similarity between users on mean-centred ratings.

    Returns (sim_matrix, user_means). user_means is kept so the predictor
    can add each user's mean back when forming predictions.
    """
    n_users = R.shape[0]
    sim = np.zeros((n_users, n_users))

    user_means = np.array(
        [
            float(np.nanmean(R[u, obs_mask[u]])) if obs_mask[u].any() else 0.0
            for u in range(n_users)
        ]
    )
    R_centred = R.copy()
    for u in range(n_users):
        R_centred[u, obs_mask[u]] -= user_means[u]
    R_centred[~obs_mask] = 0.0

    for u in range(n_users):
        for v in range(u, n_users):
            both = obs_mask[u] & obs_mask[v]
            if not both.any():
                continue
            ru, rv = R_centred[u, both], R_centred[v, both]
            denom = np.linalg.norm(ru) * np.linalg.norm(rv)
            if denom < 1e-10:
                continue
            s = float(ru @ rv / denom)
            sim[u, v] = s
            sim[v, u] = s

    return sim, user_means


def user_based_cf_predict(
    R: np.ndarray,
    obs_mask: np.ndarray,
    sim: np.ndarray,
    user_means: np.ndarray,
    k: int = K_NEIGHBOURS,
) -> np.ndarray:
    """Predict ratings using the top-k most similar users.

    prediction(u, j) = mean_u + sum(sim(u, v) * (r(v, j) - mean_v))
                                 / sum(|sim(u, v)|)
    """
    n_users, n_items = R.shape
    predictions = np.full((n_users, n_items), np.nan)

    for u in range(n_users):
        similarities = sim[u].copy()
        similarities[u] = -np.inf
        top_k = np.argsort(similarities)[-k:]
        top_k = top_k[similarities[top_k] > 0]

        if len(top_k) == 0:
            continue

        for j in range(n_items):
            rated_neighbours = top_k[obs_mask[top_k, j]]
            if len(rated_neighbours) == 0:
                continue
            weights = sim[u, rated_neighbours]
            denom = np.abs(weights).sum()
            if denom < 1e-10:
                continue
            weighted_dev = weights @ (
                R[rated_neighbours, j] - user_means[rated_neighbours]
            )
            predictions[u, j] = user_means[u] + weighted_dev / denom

    return np.clip(predictions, 1.0, 5.0)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — "TRAIN" (precompute similarity, then inference is a lookup)
# ════════════════════════════════════════════════════════════════════════
# There is no iterative training. We precompute the N x N similarity
# matrix once, then every prediction is a top-k lookup + weighted average.
# In production this precompute is rerun nightly (or incrementally when a
# user rates a new item).

print("\n" + "=" * 70)
print("  User-Based CF on SG E-commerce Ratings")
print("=" * 70)

data = build_rating_dataset()
R_train = data["R_train"]
R_observed = data["R_observed"]
train_mask = data["train_mask"]
holdout_mask = data["holdout_mask"]

user_sim, user_means = user_similarity_matrix(R_train, train_mask)
ubcf_predictions = user_based_cf_predict(
    R_train, train_mask, user_sim, user_means, k=K_NEIGHBOURS
)


# ── Checkpoint ──────────────────────────────────────────────────────────
ubcf_rmse, ubcf_cov = holdout_rmse(ubcf_predictions, R_observed, holdout_mask)
assert user_sim.shape == (N_USERS, N_USERS), "User similarity must be N x N"
assert np.allclose(user_sim, user_sim.T), "Similarity must be symmetric"
assert ubcf_rmse > 0, "User-CF RMSE should be positive"
print(
    f"\n[ok] Checkpoint passed — User-CF holdout RMSE={ubcf_rmse:.4f}, "
    f"coverage={ubcf_cov:.1%}\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the similarity structure
# ════════════════════════════════════════════════════════════════════════
# The similarity matrix is the model. Visualising it reveals whether the
# population has distinct taste clusters (block structure) or is a single
# blob (no useful neighbourhoods).

order = np.argsort(user_means)
sim_sorted = user_sim[np.ix_(order, order)]

fig = px.imshow(
    sim_sorted,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    title="User-User Similarity Heatmap (sorted by mean rating)",
    labels={"x": "user j", "y": "user i", "color": "cosine sim"},
)
save_html(fig, "02_user_similarity_heatmap.html")

neighbour_counts = [(user_sim[u] > 0).sum() - 1 for u in range(N_USERS)]
print(
    f"Neighbourhood stats: mean={np.mean(neighbour_counts):.1f}, "
    f"min={np.min(neighbour_counts)}, max={np.max(neighbour_counts)}"
)

print_method_scores("User-CF", ubcf_predictions, R_observed, holdout_mask)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore Streaming Platform Watchlist Expansion
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore-based streaming service (think mewatch / Viu) runs
# a "because users like you enjoyed..." row on its homepage. The item
# catalogue is 30K shows, ratings are explicit thumbs-up/down, and the
# platform has ~2M monthly active users across SG, MY, and ID.
#
# Why user-CF fits:
#   - Community-driven taste: "viewers who loved 'Ah Boys to Men' also
#     rated 'Money No Enough 2' highly" is a signal pure content features
#     would miss (both are SG comedies but the feature-level overlap is low)
#   - The platform has years of rating history — similarity is stable
#   - Row labels are explicit and explainable: "because u_042 liked this"
#
# BUSINESS IMPACT: Industry data from SEA streaming shows that a good
# "users like you" row lifts watch-through rate by ~18%. On a 2M MAU
# platform with S$12 ARPU, an 18% engagement lift translates to roughly
# S$4.3M/year in retained subscription revenue (churn prevention + upsell
# to annual plans).
#
# LIMITATIONS:
#   - O(N^2) similarity: at 2M users that's 4 x 10^12 pairs; production
#     systems approximate with locality-sensitive hashing or ANN libraries
#   - Cold-start users (signed up today): still zero neighbours
#   - Popularity bias: heavily-rated shows dominate every recommendation
#
# The next technique (03_item_cf.py) solves the scale problem by flipping
# the matrix and computing item-item similarity instead of user-user — the
# item catalogue grows much slower than the user base.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Computed pairwise user similarity on mean-centred ratings
  [x] Borrowed preferences from top-k most similar neighbours
  [x] Understood why mean-centring beats raw cosine for generous raters
  [x] Measured holdout RMSE + ranking metrics on a real CF model
  [x] Identified a SEA streaming scenario with S$4M/year in impact

  KEY INSIGHT: User-CF's strength is community taste that features can't
  capture. Its weakness is O(N^2) similarity compute that breaks at
  internet scale without approximation.

  Next: 03_item_cf.py — flip the similarity direction and discover why
  Amazon, Netflix, and Spotify all converged on item-based CF at scale.
"""
)
