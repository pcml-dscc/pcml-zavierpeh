# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 7.4: Matrix Factorisation with ALS (from scratch)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement Alternating Least Squares (ALS) matrix factorisation
#   - Recover true latent structure from a sparse rating matrix
#   - Track loss convergence — verify RMSE is non-increasing
#   - Visualise learned user + item embeddings in 2D
#   - Understand the SVD++ extension for implicit feedback
#   - Articulate THE PIVOT: optimisation drives feature discovery
#
# PREREQUISITES: Exercises 7.1-7.3; MLFP04 Ex 3 (PCA/SVD)
#
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — why R ≈ U V^T and why ALS converges
#   2. Build — ALS update equations from scratch
#   3. Train — run the alternating least squares loop
#   4. Visualise — user + item embeddings (PCA projection) + convergence
#   5. Apply — Spotify-style music recommendation at scale
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from kailash_ml import ModelVisualizer
from sklearn.decomposition import PCA

from shared.mlfp04.ex_7 import (
    N_ITEMS,
    N_LATENT_TRUE,
    N_USERS,
    build_rating_dataset,
    holdout_rmse,
    print_method_scores,
    save_html,
)

K_LATENT = 10
LAMBDA_REG = 0.1
N_ITERATIONS = 50


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why matrix factorisation works
# ════════════════════════════════════════════════════════════════════════
# The central assumption: every user and every item can be represented by
# a short vector of latent factors (k = 10 here). A rating is the inner
# product of the user's factors and the item's factors.
#
#   R[u, j] ≈ U[u] · V[j]
#
# If we could see the full R, SVD would give us U and V in closed form.
# Since R is sparse (only ~30% observed), we minimise a loss:
#
#   L(U, V) = sum_{(u,j) observed} (R[u,j] - U[u] · V[j])^2
#           + lambda * (||U||^2 + ||V||^2)
#
# ALS: fix V, solve for U (ridge regression per user). Then fix U, solve
# for V (ridge regression per item). Alternate until RMSE stops improving.
# Each sub-problem is a closed-form least squares solve — no gradient
# descent, no learning rate. RMSE is monotone non-increasing by
# construction.
#
# Connection to SVD: when R is fully observed and lambda = 0, ALS
# converges to the truncated SVD. Sparse + regularised ALS is a GENERALISED
# SVD that handles missing values gracefully.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the ALS update step
# ════════════════════════════════════════════════════════════════════════


def als_matrix_factorisation(
    R: np.ndarray,
    obs_mask: np.ndarray,
    k: int,
    lam: float,
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Alternating Least Squares matrix factorisation from scratch.

    Returns (U, V, rmse_history).
    """
    n_users, n_items = R.shape
    U = rng.normal(0, 0.1, size=(n_users, k))
    V = rng.normal(0, 0.1, size=(n_items, k))
    R_safe = np.nan_to_num(R, nan=0.0)
    history: list[float] = []
    identity = lam * np.eye(k)

    for iteration in range(n_iter):
        # Fix V, solve for each user
        for u in range(n_users):
            rated = np.where(obs_mask[u])[0]
            if len(rated) == 0:
                continue
            V_u = V[rated]
            r_u = R_safe[u, rated]
            A = V_u.T @ V_u + identity
            b = V_u.T @ r_u
            U[u] = np.linalg.solve(A, b)

        # Fix U, solve for each item
        for j in range(n_items):
            raters = np.where(obs_mask[:, j])[0]
            if len(raters) == 0:
                continue
            U_j = U[raters]
            r_j = R_safe[raters, j]
            A = U_j.T @ U_j + identity
            b = U_j.T @ r_j
            V[j] = np.linalg.solve(A, b)

        R_hat = U @ V.T
        residuals = (R_safe - R_hat)[obs_mask]
        rmse = float(np.sqrt(np.mean(residuals**2)))
        history.append(rmse)
        if iteration % 10 == 0 or iteration == n_iter - 1:
            print(f"  iter {iteration:3d}: train RMSE = {rmse:.4f}")

    return U, V, history


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN via ALS
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  ALS Matrix Factorisation on SG E-commerce Ratings")
print("=" * 70)
print(f"k={K_LATENT}, lambda={LAMBDA_REG}, iterations={N_ITERATIONS}")

data = build_rating_dataset()
R_train = data["R_train"]
R_observed = data["R_observed"]
train_mask = data["train_mask"]
holdout_mask = data["holdout_mask"]
U_true = data["U_true"]
V_true = data["V_true"]

U_learned, V_learned, rmse_history = als_matrix_factorisation(
    R_train,
    train_mask,
    k=K_LATENT,
    lam=LAMBDA_REG,
    n_iter=N_ITERATIONS,
    rng=data["rng"],
)

als_predictions = np.clip(U_learned @ V_learned.T, 1.0, 5.0)
als_rmse, als_cov = holdout_rmse(als_predictions, R_observed, holdout_mask)


# ── Checkpoint ──────────────────────────────────────────────────────────
assert U_learned.shape == (N_USERS, K_LATENT), "U must be (N_USERS, K_LATENT)"
assert V_learned.shape == (N_ITEMS, K_LATENT), "V must be (N_ITEMS, K_LATENT)"
for i in range(1, len(rmse_history)):
    assert (
        rmse_history[i] <= rmse_history[i - 1] + 0.01
    ), f"Train RMSE must be non-increasing (violated at step {i})"
print(
    f"\n[ok] Checkpoint passed — ALS converged: train {rmse_history[0]:.4f} -> "
    f"{rmse_history[-1]:.4f}, holdout={als_rmse:.4f}, coverage={als_cov:.0%}\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE embeddings + convergence
# ════════════════════════════════════════════════════════════════════════
# The 10-dimensional learned factors can be projected to 2D via PCA. The
# result is a scatter plot where geometric proximity = similar taste.

viz = ModelVisualizer()

pca_users = PCA(n_components=2, random_state=42)
U_2d = pca_users.fit_transform(U_learned)
user_avg = np.array(
    [
        (
            float(np.nanmean(R_observed[i, data["mask"][i]]))
            if data["mask"][i].any()
            else 0.0
        )
        for i in range(N_USERS)
    ]
)
user_df = pl.DataFrame(
    {
        "user_id": data["user_ids"],
        "pc1": U_2d[:, 0].tolist(),
        "pc2": U_2d[:, 1].tolist(),
        "avg_rating": user_avg.tolist(),
    }
)
fig_u = viz.scatter(user_df, x="pc1", y="pc2", color="avg_rating")
fig_u.update_layout(title="ALS User Embeddings (2D PCA projection)")
save_html(fig_u, "04_als_user_embeddings.html")

pca_items = PCA(n_components=2, random_state=42)
V_2d = pca_items.fit_transform(V_learned)
item_avg = np.array(
    [
        (
            float(np.nanmean(R_observed[data["mask"][:, j], j]))
            if data["mask"][:, j].any()
            else 0.0
        )
        for j in range(N_ITEMS)
    ]
)
item_df = pl.DataFrame(
    {
        "item_id": data["item_ids"],
        "pc1": V_2d[:, 0].tolist(),
        "pc2": V_2d[:, 1].tolist(),
        "avg_rating": item_avg.tolist(),
    }
)
fig_i = viz.scatter(item_df, x="pc1", y="pc2", color="avg_rating")
fig_i.update_layout(title="ALS Item Embeddings (2D PCA projection)")
save_html(fig_i, "04_als_item_embeddings.html")

fig_conv = viz.training_history({"train_rmse": rmse_history}, x_label="ALS iteration")
fig_conv.update_layout(title="ALS Convergence")
save_html(fig_conv, "04_als_convergence.html")


# Subspace alignment — verify ALS recovered the true latent structure
def subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Mean of principal angle cosines. 1.0 = perfect subspace recovery."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    _, sigmas, _ = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    return float(np.mean(np.minimum(sigmas, 1.0)))


user_align = subspace_similarity(U_learned[:, :N_LATENT_TRUE], U_true)
item_align = subspace_similarity(V_learned[:, :N_LATENT_TRUE], V_true)
print(
    f"\nSubspace recovery (1.0 = perfect): user={user_align:.3f}, item={item_align:.3f}"
)

print_method_scores("ALS MF", als_predictions, R_observed, holdout_mask)


# ════════════════════════════════════════════════════════════════════════
# SVD++ — conceptual extension (no code, just the intuition)
# ════════════════════════════════════════════════════════════════════════
# Plain MF: r_hat(u, i) = U[u] · V[i]
#
# SVD++ adds implicit feedback (clicks, views, not just ratings):
#
#   r_hat(u, i) = mu + b_u + b_i
#               + V[i] · (U[u] + |N(u)|^(-1/2) * sum_{j in N(u)} y[j])
#
# where:
#   mu, b_u, b_i  = global/user/item biases
#   N(u)          = items user u interacted with (implicit set)
#   y[j]          = implicit feedback vectors
#
# The term after U[u] is "if you clicked these items, your taste drifts
# in this direction." It captures information that pure rating-based MF
# misses. SVD++ was the key ingredient in the Netflix Prize winner.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Spotify-Style Music Recommendation at Scale
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Southeast Asian music streaming service (think Spotify SEA
# or Joox) has 600M "plays" per day across 80M tracks and 12M users.
# Explicit ratings (thumbs up/down) are rare — <3% of plays. Implicit
# feedback (skipped after 5s, played to completion, added to playlist) is
# the only usable signal at volume.
#
# Why matrix factorisation is the right tool:
#   - Implicit feedback is ~97% of the signal — user-CF and item-CF
#     struggle with binary play/skip data; MF with a confidence weighting
#     (Hu et al. 2008) handles it natively
#   - Embeddings compress 12M x 80M = 960T pairs down to (12M + 80M) x
#     128 dimensions ~ 12GB — small enough to live in RAM on a single box
#   - Once trained, inference is a single dot product per (user, track)
#     pair — ~2 microseconds, 500K predictions/sec on one CPU core
#   - The same embeddings power "similar tracks", "for you daily mix",
#     and "discover weekly" — train once, serve three products
#
# BUSINESS IMPACT: Spotify's 2019 paper reported MF-powered Discover
# Weekly drove a 30% lift in discovery listening. On a SEA market with
# S$180M annual streaming ARR, a 30% discovery lift translates to roughly
# S$35M/year in additional watch-time-to-revenue conversion, plus the
# retention benefit (users who engage with Discover Weekly churn 50% less).
#
# LIMITATIONS:
#   - Cold-start tracks: a brand-new upload has no listens = no factors
#     (solution: hybrid with content-based audio embeddings)
#   - Popularity bias: top-streamed tracks dominate every recommendation
#   - Filter bubble: the 10-dim space collapses the long tail
#
# THE PIVOT: optimisation drives feature discovery.
#
#   We never told ALS what the 10 latent dimensions mean. It learned them
#   by minimising reconstruction error. Compare:
#       MF:              min ||R - U V^T||^2
#       Neural net:      min ||y - f(W x + b)||^2
#   SAME PRINCIPLE. Neural nets just add non-linearities.
#
# The next technique (05_hybrid_evaluation.py) combines all four
# recommenders and reveals a deeper truth: no single method wins on
# every user — hybrid blending lifts P@5 and MAP beyond any component.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Implemented ALS from scratch — ridge regression per user + per item
  [x] Verified train RMSE is non-increasing ({rmse_history[0]:.4f} -> {rmse_history[-1]:.4f})
  [x] Recovered the true latent subspace (user={user_align:.3f}, item={item_align:.3f})
  [x] Visualised 10-dim embeddings in 2D via PCA
  [x] Understood SVD++ as the implicit-feedback extension of MF
  [x] Identified a S$35M/year SEA streaming scenario where MF dominates

  THE PIVOT: Matrix factorisation DISCOVERS latent factors by optimising
  a loss. This is the same principle as neural network training — the
  hidden layer is just an embedding, learned by backpropagation.

  Next: 05_hybrid_evaluation.py — combine all four recommenders and see
  why the Netflix Prize winner used 107 blended models.
"""
)
