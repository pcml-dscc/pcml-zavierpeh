# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 7.3: Item-Based Collaborative Filtering
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Flip CF from user-similarity to item-similarity
#   - Understand why item similarity is more stable than user similarity
#   - Implement item-item cosine similarity with mean-centring per item
#   - See why Amazon/Netflix/Spotify converged on item-CF at scale
#
# PREREQUISITES: Exercise 7.2 (user-based CF)
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — "items co-rated by the same people are similar"
#   2. Build — item similarity + weighted sum predictor
#   3. Train — precompute the item x item matrix once
#   4. Visualise — item similarity heatmap + nearest neighbours
#   5. Apply — Amazon-style "customers also bought"
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.express as px

from shared.mlfp04.ex_7 import (
    N_ITEMS,
    build_rating_dataset,
    holdout_rmse,
    print_method_scores,
    save_html,
)

K_NEIGHBOURS = 20


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why item-based CF dominates at scale
# ════════════════════════════════════════════════════════════════════════
# User-CF asks "who rates like me?" and struggles with scale. Item-CF
# asks "which items were rated similarly?" — the item set is smaller and
# more stable. Mean-centre PER ITEM (not per user) to remove the
# "everyone loves this item" bias.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD item similarity + predictor
# ════════════════════════════════════════════════════════════════════════


def item_similarity_matrix(
    R: np.ndarray, obs_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise cosine similarity between items on mean-centred ratings."""
    n_items = R.shape[1]
    sim = np.zeros((n_items, n_items))

    # TODO: Compute the mean rating per item using np.nanmean over the
    # observed raters for each column.
    item_means = np.array([____ for j in range(n_items)])

    # TODO: Build a mean-centred copy of R. Subtract item_means[j] from
    # each column's observed rows; zero out unobserved entries.
    R_centred = R.copy()
    for j in range(n_items):
        ____
    R_centred[~obs_mask] = 0.0

    for i in range(n_items):
        for j in range(i, n_items):
            both = obs_mask[:, i] & obs_mask[:, j]
            if not both.any():
                continue
            ri, rj = R_centred[both, i], R_centred[both, j]
            denom = np.linalg.norm(ri) * np.linalg.norm(rj)
            if denom < 1e-10:
                continue
            # TODO: cosine similarity between ri and rj
            s = ____
            sim[i, j] = s
            sim[j, i] = s
    return sim, item_means


def item_based_cf_predict(
    R: np.ndarray,
    obs_mask: np.ndarray,
    item_sim: np.ndarray,
    k: int = K_NEIGHBOURS,
) -> np.ndarray:
    """Weighted-sum predictor over the top-k most similar items."""
    n_users, n_items = R.shape
    predictions = np.full((n_users, n_items), np.nan)

    for u in range(n_users):
        rated_items = np.where(obs_mask[u])[0]
        if len(rated_items) == 0:
            continue
        for j in range(n_items):
            sims = item_sim[j, rated_items]
            if len(sims) > k:
                top_idx = np.argsort(sims)[-k:]
            else:
                top_idx = np.arange(len(sims))
            pos_idx = top_idx[sims[top_idx] > 0]
            if len(pos_idx) == 0:
                continue
            weights = sims[pos_idx]
            denom = np.abs(weights).sum()
            if denom < 1e-10:
                continue
            # TODO: weighted average of the user's ratings on rated_items[pos_idx]
            predictions[u, j] = ____

    return np.clip(predictions, 1.0, 5.0)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Precompute item x item once
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Item-Based CF on SG E-commerce Ratings")
print("=" * 70)

data = build_rating_dataset()
R_train = data["R_train"]
R_observed = data["R_observed"]
train_mask = data["train_mask"]
holdout_mask = data["holdout_mask"]
item_ids = data["item_ids"]

item_sim, item_means = item_similarity_matrix(R_train, train_mask)
ibcf_predictions = item_based_cf_predict(R_train, train_mask, item_sim, k=K_NEIGHBOURS)


# ── Checkpoint ──────────────────────────────────────────────────────────
ibcf_rmse, ibcf_cov = holdout_rmse(ibcf_predictions, R_observed, holdout_mask)
assert item_sim.shape == (N_ITEMS, N_ITEMS), "Item similarity must be M x M"
assert np.allclose(item_sim, item_sim.T), "Item similarity must be symmetric"
assert ibcf_rmse > 0, "Item-CF RMSE should be positive"
print(
    f"\n[ok] Checkpoint passed — Item-CF holdout RMSE={ibcf_rmse:.4f}, "
    f"coverage={ibcf_cov:.1%}\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE similarity structure
# ════════════════════════════════════════════════════════════════════════

order = np.argsort(item_means)
item_sim_sorted = item_sim[np.ix_(order, order)]
fig_heat = px.imshow(
    item_sim_sorted,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    title="Item-Item Similarity Heatmap (sorted by mean rating)",
    labels={"x": "item j", "y": "item i", "color": "cosine sim"},
)
save_html(fig_heat, "03_item_similarity_heatmap.html")

# TODO: Pick the most-rated item (argmax over train_mask.sum(axis=0))
# and print its top-5 most similar items as "customers also bought" candidates.
rated_counts = ____
anchor = ____
top5 = np.argsort(item_sim[anchor])[::-1][1:6]
print(f"\nAnchor item: {item_ids[anchor]} (rated by {rated_counts[anchor]} users)")
print("Top-5 most similar items ('customers also bought'):")
for j in top5:
    print(f"  {item_ids[j]}  sim={item_sim[anchor, j]:+.3f}")

print_method_scores("Item-CF", ibcf_predictions, R_observed, holdout_mask)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Amazon-Style "Customers Who Bought This Also Bought..."
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: SG cross-border e-commerce platform (1.8M users, 12M SKUs).
# "You may also like" drives ~22% of GMV. Item-CF fits because:
#   - O(M^2) scales with items, not users (M grows slower than N)
#   - Item relationships are stable and precomputable
#   - Redis top-50-neighbours cache = <5ms page-load recommendations
#
# BUSINESS IMPACT: 0.3% lift in cross-sell conversion on S$4.2B annual
# GMV = S$12.6M/year, vs ~S$250K engineering/infra cost. 50x ROI.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built item-item similarity on mean-centred ratings
  [x] Inspected top-5 neighbours for the most-rated item
  [x] Understood why item-CF scales better than user-CF

  Next: 04_matrix_factorisation.py — learn dense embeddings by optimisation.
"""
)
