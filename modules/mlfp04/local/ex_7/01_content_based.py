# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 7.1: Content-Based Filtering
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a user profile from the items they already rated
#   - Score unseen items by cosine similarity to that profile
#   - Understand why content-based filtering handles cold-start items
#   - See where it FAILS: cold-start users, narrow interest tunnels
#
# PREREQUISITES: MLFP04 Exercise 3 (cosine similarity, vector embeddings)
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — item features + user profile intuition
#   2. Build — content-based prediction function
#   3. Train — no training loop; profiles are computed at inference
#   4. Visualise — predicted vs observed ratings scatter
#   5. Apply — Singapore e-commerce new-SKU launch
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_7 import (
    N_ITEMS,
    N_USERS,
    build_rating_dataset,
    holdout_rmse,
    print_method_scores,
    save_html,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Why content-based works
# ════════════════════════════════════════════════════════════════════════
# Every item has a feature vector. Every user has a rating history. The
# user's "taste profile" is a weighted sum of the features of items they
# rated, with weights = ratings. To score a new item, cosine-similarity
# it against the profile.
#
# STRENGTHS: works for cold-start items, explainable, no cross-user leakage
# WEAKNESSES: cold-start users (no profile), filter bubble, feature quality


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the content-based predictor
# ════════════════════════════════════════════════════════════════════════


def content_based_predict(
    R: np.ndarray,
    item_feats: np.ndarray,
    obs_mask: np.ndarray,
) -> np.ndarray:
    """Predict ratings via cosine similarity to the user's taste profile."""
    n_users, n_items = R.shape
    predictions = np.full((n_users, n_items), np.nan)

    for u in range(n_users):
        rated_idx = np.where(obs_mask[u])[0]
        if len(rated_idx) == 0:
            continue

        # TODO: Build the user's taste profile as the rating-weighted sum of
        # item feature vectors for items they rated. Hint:
        #   ratings_u = np.nan_to_num(R[u, rated_idx], nan=0.0)
        #   profile = (ratings_u[:, None] * item_feats[rated_idx]).sum(axis=0)
        profile = ____

        # TODO: Normalise the profile to unit length (guard against zero norm)
        profile_norm = np.linalg.norm(profile)
        if profile_norm < 1e-10:
            continue
        profile = ____

        for j in range(n_items):
            feat_norm = np.linalg.norm(item_feats[j])
            if feat_norm < 1e-10:
                continue
            # TODO: cosine similarity between profile and item_feats[j]
            sim = ____
            # Map cosine [-1, 1] to rating [1, 5]
            predictions[u, j] = 1.0 + (sim + 1.0) * 2.0

    return predictions


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — "TRAIN" (content-based has no training loop)
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Content-Based Filtering on SG E-commerce Ratings")
print("=" * 70)

data = build_rating_dataset()
R_train = data["R_train"]
R_observed = data["R_observed"]
train_mask = data["train_mask"]
holdout_mask = data["holdout_mask"]
item_features = data["item_features"]

# TODO: Call content_based_predict on R_train with item_features and train_mask
cb_predictions = ____


# ── Checkpoint ──────────────────────────────────────────────────────────
cb_rmse, cb_coverage = holdout_rmse(cb_predictions, R_observed, holdout_mask)
assert cb_predictions.shape == (N_USERS, N_ITEMS), "Prediction matrix shape"
assert cb_rmse > 0, "Content-based RMSE should be positive"
print(
    "\n[ok] Checkpoint passed — content-based predictor produced holdout "
    f"RMSE={cb_rmse:.4f}, coverage={cb_coverage:.1%}\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE predicted vs observed
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

pairs = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if holdout_mask[i, j] and not np.isnan(cb_predictions[i, j]):
            pairs.append(
                {
                    "user": f"u{i:03d}",
                    "true": float(R_observed[i, j]),
                    "pred": float(cb_predictions[i, j]),
                }
            )

pair_df = pl.DataFrame(pairs)
# TODO: Build a scatter plot with viz.scatter — x="true", y="pred", color="user"
fig = ____
fig.update_layout(
    title="Content-Based: Predicted vs Observed Rating (holdout)",
    xaxis_title="Observed rating (1-5)",
    yaxis_title="Predicted rating (1-5)",
)
save_html(fig, "01_content_based_scatter.html")

print_method_scores("Content-Based", cb_predictions, R_observed, holdout_mask)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore E-commerce New-SKU Launch
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore consumer-electronics retailer launches 200 new
# SKUs per week. New SKUs have zero ratings — pure cold-start. CF methods
# all need co-rating history that doesn't exist yet.
#
# Content-based filtering is the ONLY method here that can recommend a
# SKU on day 0 as long as the SKU has a feature vector (price tier,
# category, brand, screen size, etc).
#
# BUSINESS IMPACT: New SKUs earn ~80% of lifetime revenue in the first
# 30 days. Content-based filtering recovers ~60% of the cold-start gap —
# roughly S$72K/month in recovered cross-sell revenue on a 200-SKU/week
# launch cadence.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a user taste profile from rated items + their ratings
  [x] Scored unseen items via cosine similarity to the profile
  [x] Measured holdout RMSE and ranking metrics (P@5, MAP)
  [x] Understood WHEN to use content-based: cold-start items
  [x] Understood WHEN it fails: cold-start users, narrow graphs

  Next: 02_user_cf.py — community opinions escape the filter bubble.
"""
)
