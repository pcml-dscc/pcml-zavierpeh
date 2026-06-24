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
#   3. Train — no training; neighbourhoods are looked up at inference
#   4. Visualise — user similarity heatmap + neighbour quality
#   5. Apply — Singapore streaming watchlist expansion
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
# If Alice and Bob agreed on 50 past items, Bob's opinion on a new item
# is strong evidence for Alice. Find each user's nearest neighbours in
# rating space and predict by a weighted average of their ratings.
#
# Mean-centring per user removes "generous rater" bias so the similarity
# compares rankings, not absolute scores.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD similarity + predictor
# ════════════════════════════════════════════════════════════════════════


def user_similarity_matrix(
    R: np.ndarray, obs_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise cosine similarity between users on mean-centred ratings."""
    n_users = R.shape[0]
    sim = np.zeros((n_users, n_users))

    # TODO: Compute each user's mean rating (only over observed entries).
    # Use np.nanmean on R[u, obs_mask[u]] and default to 0.0 when empty.
    user_means = np.array([____ for u in range(n_users)])

    # TODO: Build a mean-centred copy of R. Subtract user_means[u] from
    # each user's observed entries and zero out the unobserved entries.
    R_centred = R.copy()
    for u in range(n_users):
        ____
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
            # TODO: cosine similarity of centred vectors ru, rv
            s = ____
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
    """Predict ratings using the top-k most similar users."""
    n_users, n_items = R.shape
    predictions = np.full((n_users, n_items), np.nan)

    for u in range(n_users):
        similarities = sim[u].copy()
        similarities[u] = -np.inf
        # TODO: Pick the top-k indices by similarity. Hint: np.argsort(...)[-k:]
        top_k = ____
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
            # TODO: weighted deviation of neighbours' ratings from their means
            weighted_dev = ____
            predictions[u, j] = user_means[u] + weighted_dev / denom

    return np.clip(predictions, 1.0, 5.0)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Precompute similarity, then inference is a lookup
# ════════════════════════════════════════════════════════════════════════

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
# TASK 4 — VISUALISE similarity structure
# ════════════════════════════════════════════════════════════════════════

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
# TASK 5 — APPLY: Singapore Streaming Watchlist Expansion
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A SG/SEA streaming service (~2M MAU) runs a "because users
# like you enjoyed..." row on its homepage. User-CF captures community
# taste that content features miss.
#
# BUSINESS IMPACT: 18% watch-through lift ~ S$4.3M/year retained
# subscription revenue on a 2M MAU S$12 ARPU platform.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Computed pairwise user similarity on mean-centred ratings
  [x] Borrowed preferences from top-k nearest neighbours
  [x] Understood why mean-centring beats raw cosine
  [x] Identified a SEA streaming scenario with S$4M/year in impact

  Next: 03_item_cf.py — flip the direction and scale the algorithm.
"""
)
