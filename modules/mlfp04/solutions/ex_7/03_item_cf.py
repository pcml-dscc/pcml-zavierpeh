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
#   - See why Amazon/Netflix/Spotify all converged on item-CF at scale
#
# PREREQUISITES: Exercise 7.2 (user-based CF)
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — "items that were co-rated by the same people"
#   2. Build — item similarity + weighted sum predictor
#   3. Train — precompute the item x item matrix once
#   4. Visualise — item similarity heatmap + most-similar items
#   5. Apply — Amazon-style "customers who bought this also bought..."
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.express as px
from kailash_ml import ModelVisualizer  # noqa: F401

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
# User-based CF asks: "who rates like me?" and is unstable — users change
# tastes, new users have no history, and the N-user set grows with every
# signup (often into the millions).
#
# Item-based CF asks: "which items were rated the same way?" The item set
# is much smaller and far more stable — Amazon has ~500M products but the
# top-selling 50K account for 95% of impressions. Item-item relationships
# ("people who bought A also bought B") change slowly, so the precompute
# can run nightly and still be accurate.
#
# Key trick: mean-centre PER ITEM (not per user). This removes the
# "everyone loves this item" bias and compares how items RANK in each
# user's preference order.
#
# This is the algorithm Amazon published in 2003 and it still runs under
# "Customers who bought this also bought..." today.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD item similarity + predictor
# ════════════════════════════════════════════════════════════════════════


def item_similarity_matrix(
    R: np.ndarray, obs_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise cosine similarity between items on mean-centred ratings."""
    n_items = R.shape[1]
    sim = np.zeros((n_items, n_items))
    item_means = np.array(
        [
            float(np.nanmean(R[obs_mask[:, j], j])) if obs_mask[:, j].any() else 0.0
            for j in range(n_items)
        ]
    )
    R_centred = R.copy()
    for j in range(n_items):
        R_centred[obs_mask[:, j], j] -= item_means[j]
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
            s = float(ri @ rj / denom)
            sim[i, j] = s
            sim[j, i] = s
    return sim, item_means


def item_based_cf_predict(
    R: np.ndarray,
    obs_mask: np.ndarray,
    item_sim: np.ndarray,
    k: int = K_NEIGHBOURS,
) -> np.ndarray:
    """Weighted-sum predictor over the top-k most similar items.

    For user u and target item j:
      prediction = sum(sim(j, i) * r(u, i)) / sum(|sim(j, i)|)
    where i ranges over the top-k items user u already rated.
    """
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
            predictions[u, j] = weights @ R[u, rated_items[pos_idx]] / denom

    return np.clip(predictions, 1.0, 5.0)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — "TRAIN" (precompute item-item matrix once)
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
# TASK 4 — VISUALISE item similarity structure
# ════════════════════════════════════════════════════════════════════════
# Two visual signals matter:
#   1. The similarity matrix — does it show block structure? (= categories)
#   2. For a chosen "anchor" item, which items are most similar?
#      These are the "customers who bought this also bought..." candidates.

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

# Pick the most-rated item and show its top-5 neighbours
rated_counts = train_mask.sum(axis=0)
anchor = int(np.argmax(rated_counts))
top5 = np.argsort(item_sim[anchor])[::-1][1:6]
print(f"\nAnchor item: {item_ids[anchor]} (rated by {rated_counts[anchor]} users)")
print("Top-5 most similar items ('customers also bought'):")
for j in top5:
    print(f"  {item_ids[j]}  sim={item_sim[anchor, j]:+.3f}")

print_method_scores("Item-CF", ibcf_predictions, R_observed, holdout_mask)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Amazon-Style "Customers Who Bought This Also Bought..."
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Singapore cross-border e-commerce platform (think Qoo10 /
# Shopee SG) serves 1.8M active users and a 12M-item catalogue. The "you
# may also like" carousel on every product page is the platform's
# highest-converting surface — it drives ~22% of gross merchandise value.
#
# Why item-CF is the industry default:
#   - The similarity matrix is O(M^2) not O(N^2). For Shopee, M (items)
#     grows ~10% per year while N (users) grows ~35% — item-CF scales
#     with the smaller dimension
#   - Item relationships are stable: "phone + phone case" stays true for
#     years, while user taste shifts monthly
#   - Precompute once, cache forever: the sparse top-50 neighbours per
#     item fits in a single Redis key of ~2KB, enabling <5ms lookups at
#     page-load time
#
# BUSINESS IMPACT: A 1% lift in cross-sell conversion on a S$4.2B annual
# GMV platform = S$42M in incremental revenue. Even a conservative 0.3%
# lift from tuning the item-CF model is S$12.6M/year — vs roughly
# S$250K/year in engineering + infra cost. 50x ROI, reason Amazon has run
# this algorithm for 20+ years.
#
# LIMITATIONS:
#   - Niche items (long tail) have sparse similarity rows
#   - Items with wildly different rating distributions still leak through
#   - Cold-start NEW items still need content features (back to Ex 7.1)
#
# The next technique (04_matrix_factorisation.py) takes a completely
# different approach: learn dense user and item embeddings by minimising
# a loss — the bridge from recommenders to deep learning.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Flipped CF from user-similarity to item-similarity
  [x] Understood why items are more stable than users at scale
  [x] Built the Amazon "customers also bought" predictor
  [x] Inspected top-5 neighbours for the most-rated item
  [x] Identified a S$12-42M/year impact scenario for SG e-commerce

  KEY INSIGHT: Amazon, Netflix, Spotify all converged on item-CF because
  the item catalogue is smaller, more stable, and precomputable.

  Next: 04_matrix_factorisation.py — abandon similarity entirely and
  learn dense embeddings by optimisation. This is the bridge from
  classical recommenders to neural networks.
"""
)
