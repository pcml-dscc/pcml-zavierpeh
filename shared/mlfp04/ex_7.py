# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 7 — Recommender Systems.

Scenario: Singapore e-commerce marketplace (similar to Shopee/Lazada).
  - 100 users x 50 SKUs from a fictional SG electronics retailer
  - 30% ratings observed (explicit 1-5 stars)
  - Holdout 20% of observed ratings for evaluation
  - Item features = 8 latent product attributes (price tier, category, etc.)

Contains: synthetic rating matrix generation, evaluation metrics (RMSE,
precision@k, MAP), shared constants, and output-dir setup.

Technique-specific code (content-based profile, user/item CF, ALS, hybrid
blending) lives in the per-technique files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════

setup_environment()

OUTPUT_DIR = Path("outputs") / "ex7_recommenders"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# CONSTANTS — synthetic SG e-commerce dataset
# ════════════════════════════════════════════════════════════════════════

N_USERS = 100
N_ITEMS = 50
N_LATENT_TRUE = 5
N_ITEM_FEATURES = 8
SPARSITY = 0.30
HOLDOUT_FRAC = 0.20
RNG_SEED = 42

RATING_MIN = 1.0
RATING_MAX = 5.0
RELEVANCE_THRESHOLD = 3.5  # ratings >= 3.5 are "relevant" for ranking metrics


# ════════════════════════════════════════════════════════════════════════
# SYNTHETIC RATING MATRIX — shared across every technique
# ════════════════════════════════════════════════════════════════════════


def build_rating_dataset(
    seed: int = RNG_SEED,
) -> dict:
    """Generate the shared synthetic e-commerce rating matrix.

    Returns a dict with:
      R_observed       — (N_USERS, N_ITEMS) with NaN where not rated
      R_train          — R_observed with holdout entries set to NaN
      mask             — boolean observed mask
      train_mask       — boolean training-only mask
      holdout_mask     — boolean holdout mask
      U_true, V_true   — ground-truth latent factors (for subspace recovery)
      item_features    — (N_ITEMS, N_ITEM_FEATURES) content-feature matrix
      user_ids, item_ids — stable string IDs
      ratings_df       — long-format polars DataFrame of observed ratings
    """
    rng = np.random.default_rng(seed=seed)

    U_true = rng.normal(0, 1, size=(N_USERS, N_LATENT_TRUE))
    V_true = rng.normal(0, 1, size=(N_ITEMS, N_LATENT_TRUE))

    R_full = U_true @ V_true.T
    R_full = (R_full - R_full.min()) / (R_full.max() - R_full.min()) * 4 + 1
    R_full += rng.normal(0, 0.3, size=R_full.shape)
    R_full = np.clip(R_full, RATING_MIN, RATING_MAX)

    mask = rng.random(size=(N_USERS, N_ITEMS)) < SPARSITY
    R_observed = np.where(mask, R_full, np.nan)

    holdout_mask = mask & (rng.random(size=(N_USERS, N_ITEMS)) < HOLDOUT_FRAC)
    train_mask = mask & ~holdout_mask
    R_train = np.where(train_mask, R_observed, np.nan)

    item_features = rng.random(size=(N_ITEMS, N_ITEM_FEATURES))

    user_ids = [f"sg_user_{i:03d}" for i in range(N_USERS)]
    item_ids = [f"sku_{j:02d}" for j in range(N_ITEMS)]

    rows = []
    for i in range(N_USERS):
        for j in range(N_ITEMS):
            if mask[i, j]:
                rows.append(
                    {
                        "user_id": user_ids[i],
                        "item_id": item_ids[j],
                        "rating": round(float(R_observed[i, j]), 1),
                        "in_holdout": bool(holdout_mask[i, j]),
                    }
                )
    ratings_df = pl.DataFrame(rows)

    return {
        "R_observed": R_observed,
        "R_train": R_train,
        "mask": mask,
        "train_mask": train_mask,
        "holdout_mask": holdout_mask,
        "U_true": U_true,
        "V_true": V_true,
        "item_features": item_features,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "ratings_df": ratings_df,
        "rng": rng,
    }


# ════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS — shared across every technique
# ════════════════════════════════════════════════════════════════════════


def holdout_rmse(
    predictions: np.ndarray,
    R_true: np.ndarray,
    holdout_mask: np.ndarray,
) -> tuple[float, float]:
    """Return (RMSE, coverage) on the holdout mask.

    Coverage = fraction of holdout pairs where the method produced a
    non-NaN prediction. Methods that can't predict cold pairs have
    coverage < 1.0.
    """
    errors = []
    n_holdout = int(holdout_mask.sum())
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if holdout_mask[i, j] and not np.isnan(predictions[i, j]):
                errors.append((R_true[i, j] - predictions[i, j]) ** 2)
    if not errors:
        return float("inf"), 0.0
    rmse = float(np.sqrt(np.mean(errors)))
    coverage = len(errors) / max(n_holdout, 1)
    return rmse, coverage


def precision_at_k(
    predictions: np.ndarray,
    R_true: np.ndarray,
    holdout_mask: np.ndarray,
    k: int = 5,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Precision@k averaged across users.

    For each user, rank holdout items by predicted score, take the top-k,
    and compute what fraction of those have true rating >= threshold.
    """
    precisions = []
    for u in range(predictions.shape[0]):
        holdout_items = np.where(holdout_mask[u])[0]
        if len(holdout_items) == 0:
            continue
        relevant = {j for j in holdout_items if R_true[u, j] >= threshold}
        if not relevant:
            continue
        scored = [
            (j, predictions[u, j])
            for j in holdout_items
            if not np.isnan(predictions[u, j])
        ]
        scored.sort(key=lambda x: -x[1])
        top = [j for j, _ in scored[:k]]
        if not top:
            continue
        precisions.append(len(set(top) & relevant) / len(top))
    return float(np.mean(precisions)) if precisions else 0.0


def mean_average_precision(
    predictions: np.ndarray,
    R_true: np.ndarray,
    holdout_mask: np.ndarray,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Mean Average Precision across users on the holdout set."""
    aps = []
    for u in range(predictions.shape[0]):
        holdout_items = np.where(holdout_mask[u])[0]
        if len(holdout_items) == 0:
            continue
        relevant = {j for j in holdout_items if R_true[u, j] >= threshold}
        if not relevant:
            continue
        scored = [
            (j, predictions[u, j])
            for j in holdout_items
            if not np.isnan(predictions[u, j])
        ]
        scored.sort(key=lambda x: -x[1])

        hits = 0
        sum_precision = 0.0
        for rank, (j, _) in enumerate(scored, 1):
            if j in relevant:
                hits += 1
                sum_precision += hits / rank
        if hits > 0:
            aps.append(sum_precision / len(relevant))
    return float(np.mean(aps)) if aps else 0.0


def print_method_scores(
    name: str, preds: np.ndarray, R_true: np.ndarray, holdout_mask: np.ndarray
) -> dict:
    """Print a single-line scorecard and return the metrics dict."""
    rmse, cov = holdout_rmse(preds, R_true, holdout_mask)
    p5 = precision_at_k(preds, R_true, holdout_mask, k=5)
    m = mean_average_precision(preds, R_true, holdout_mask)
    print(
        f"  {name:<22} RMSE={rmse:6.4f}  coverage={cov:6.1%}  "
        f"P@5={p5:6.4f}  MAP={m:6.4f}"
    )
    return {"RMSE": rmse, "Coverage": cov, "P@5": p5, "MAP": m}


# ════════════════════════════════════════════════════════════════════════
# VISUAL HELPERS
# ════════════════════════════════════════════════════════════════════════


def save_html(fig, filename: str) -> Path:
    """Write a plotly fig into OUTPUT_DIR and return the path."""
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    print(f"  saved: {path}")
    return path
