# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 2 — EM and Gaussian Mixture Models.

Contains: synthetic GMM data generation, Singapore e-commerce loader, scaler,
scoring helpers, and OUTPUT_DIR management.

Technique-specific code (the EM E-step / M-step from scratch, BIC/AIC sweeps,
covariance-type comparison, mixture-of-experts gating) does NOT live here.
It lives in the per-technique files under modules/mlfp04/solutions/ex_2/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

OUTPUT_DIR = Path("outputs") / "ex2_gmm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE: int = 42


# ════════════════════════════════════════════════════════════════════════
# SYNTHETIC 2D DATA — 3 well-separated Gaussians
# ════════════════════════════════════════════════════════════════════════
#
# Used to validate the from-scratch EM implementation. Three well-separated
# components make convergence obvious and let students verify the recovered
# parameters against the ground truth.

TRUE_MEANS: np.ndarray = np.array([[0.0, 0.0], [5.0, 2.0], [2.0, 6.0]])
TRUE_COVS: np.ndarray = np.array(
    [
        [[1.0, 0.3], [0.3, 0.8]],
        [[0.8, -0.2], [-0.2, 1.2]],
        [[1.5, 0.0], [0.0, 0.5]],
    ]
)
TRUE_WEIGHTS: np.ndarray = np.array([0.4, 0.35, 0.25])
N_SYNTH: int = 600


def make_synthetic_gmm(
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw N_SYNTH points from the 3-component GMM defined by TRUE_*.

    Returns (X, z_true) where z_true is the ground-truth component index.
    """
    rng = np.random.default_rng(seed)
    n_per = (TRUE_WEIGHTS * N_SYNTH).astype(int)
    n_per[-1] = N_SYNTH - n_per[:-1].sum()

    parts: list[np.ndarray] = []
    labels: list[int] = []
    for k, (mean, cov, n) in enumerate(zip(TRUE_MEANS, TRUE_COVS, n_per)):
        parts.append(rng.multivariate_normal(mean, cov, n))
        labels.extend([k] * n)

    X = np.vstack(parts)
    z = np.array(labels)

    # Shuffle so the order does not leak the label
    idx = rng.permutation(N_SYNTH)
    return X[idx], z[idx]


# ════════════════════════════════════════════════════════════════════════
# REAL DATA — Singapore e-commerce customers
# ════════════════════════════════════════════════════════════════════════
#
# We reuse the MLFP03 e-commerce customer dataset (~6K rows, Singapore)
# for every real-data task in this exercise. Segmentation is the business
# frame: the GMM will propose soft customer segments that marketing can
# score on expected value.


def load_customers_scaled() -> (
    tuple[np.ndarray, pl.DataFrame, list[str], StandardScaler]
):
    """Return (X_scaled, customers_df, feature_cols, scaler).

    The DataFrame and feature_cols are returned so technique files can
    join cluster labels back onto the original rows for segment profiling.
    """
    loader = MLFPDataLoader()
    customers = loader.load("mlfp03", "ecommerce_customers.parquet")

    numeric_types = (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    feature_cols = [
        c
        for c, d in zip(customers.columns, customers.dtypes)
        if d in numeric_types and c not in ("customer_id",)
    ]
    customers = customers.drop_nulls(subset=feature_cols)
    X, _, _ = to_sklearn_input(customers, feature_columns=feature_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, customers, feature_cols, scaler


# ════════════════════════════════════════════════════════════════════════
# PARAMETER COUNTING — for BIC/AIC interpretation
# ════════════════════════════════════════════════════════════════════════


def count_gmm_params(n_components: int, n_features: int, cov_type: str) -> int:
    """Number of free parameters in a GMM given components, features, cov type.

    Used to explain the BIC/AIC ranking across covariance types.
    """
    d = n_features
    k = n_components
    if cov_type == "full":
        return k * (d * (d + 1) // 2 + d + 1) - 1
    if cov_type == "tied":
        return d * (d + 1) // 2 + k * (d + 1) - 1
    if cov_type == "diag":
        return k * (2 * d + 1) - 1
    if cov_type == "spherical":
        return k * (d + 2) - 1
    raise ValueError(f"Unknown cov_type: {cov_type}")


# ════════════════════════════════════════════════════════════════════════
# SCORING HELPERS
# ════════════════════════════════════════════════════════════════════════


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette with a graceful fallback when only one cluster is present."""
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


def out_path(filename: str) -> Path:
    """Return a path under OUTPUT_DIR for a visualisation artefact."""
    return OUTPUT_DIR / filename
