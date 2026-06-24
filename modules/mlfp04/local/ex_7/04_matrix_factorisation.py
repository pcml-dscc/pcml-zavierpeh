# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 7.4: Matrix Factorisation with ALS
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement Alternating Least Squares (ALS) from scratch
#   - Recover true latent structure from a sparse rating matrix
#   - Track loss convergence — verify RMSE is non-increasing
#   - Visualise learned user + item embeddings in 2D
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
#   4. Visualise — user + item embeddings + convergence
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
# R[u,j] ≈ U[u] · V[j]. When R is sparse, solve:
#   min_{U,V} sum_{(u,j) observed} (R[u,j] - U[u]·V[j])^2
#                                   + lambda (||U||^2 + ||V||^2)
#
# ALS: fix V, solve for U (ridge regression per user). Fix U, solve for V.
# Alternate. Each sub-problem is closed-form — no learning rate needed.
# Monotone non-increasing RMSE by construction.


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
    """Alternating Least Squares matrix factorisation from scratch."""
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
            # TODO: Build A = V_u^T V_u + lambda*I and b = V_u^T r_u, then
            # solve A x = b and store into U[u]. Hint: np.linalg.solve.
            A = ____
            b = ____
            U[u] = ____

        # Fix U, solve for each item
        for j in range(n_items):
            raters = np.where(obs_mask[:, j])[0]
            if len(raters) == 0:
                continue
            U_j = U[raters]
            r_j = R_safe[raters, j]
            # TODO: Mirror the user update for the item side.
            A = ____
            b = ____
            V[j] = ____

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
    f"{rmse_history[-1]:.4f}, holdout={als_rmse:.4f}\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE embeddings + convergence
# ════════════════════════════════════════════════════════════════════════

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
# TODO: viz.scatter(user_df, x="pc1", y="pc2", color="avg_rating")
fig_u = ____
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
# TODO: same scatter for items
fig_i = ____
fig_i.update_layout(title="ALS Item Embeddings (2D PCA projection)")
save_html(fig_i, "04_als_item_embeddings.html")

# TODO: viz.training_history({"train_rmse": rmse_history}, x_label="ALS iteration")
fig_conv = ____
fig_conv.update_layout(title="ALS Convergence")
save_html(fig_conv, "04_als_convergence.html")


def subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    _, sigmas, _ = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    return float(np.mean(np.minimum(sigmas, 1.0)))


user_align = subspace_similarity(U_learned[:, :N_LATENT_TRUE], U_true)
item_align = subspace_similarity(V_learned[:, :N_LATENT_TRUE], V_true)
print(f"\nSubspace recovery: user={user_align:.3f}, item={item_align:.3f}")

print_method_scores("ALS MF", als_predictions, R_observed, holdout_mask)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Spotify-Style Music Recommendation at Scale
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: SEA streaming (12M users, 80M tracks, 600M daily plays).
# Matrix factorisation compresses an intractable user-item matrix into
# small dense embeddings; one dot product = one prediction.
#
# BUSINESS IMPACT: MF-powered Discover Weekly drives 30% discovery lift =
# S$35M/year incremental revenue on a S$180M ARR market.
#
# THE PIVOT: ALS learned the 10 latent dimensions by minimising a loss.
# Neural networks do the same thing — hidden activations are embeddings
# learned by optimisation. Matrix factorisation is a linear special case.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Implemented ALS from scratch
  [x] Verified non-increasing train RMSE
  [x] Recovered the true latent subspace
  [x] Understood THE PIVOT: optimisation drives feature discovery

  Next: 05_hybrid_evaluation.py — blend all four methods.
"""
)
