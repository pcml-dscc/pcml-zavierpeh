# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Shared infrastructure for MLFP02 Exercise 6 — Logistic Regression and
Classification Foundations.

Contains: HDB data loading, binary target construction, design matrix
preparation, sigmoid implementation, the neg-log-likelihood + gradient used
by scipy.optimize, calibration-curve binning, and output-directory setup.

Technique-specific code (odds ratios, threshold sweeps, confusion matrices,
ROC/PR/calibration plots, ANOVA + Tukey HSD) does NOT belong here — it
lives in the per-technique files in solutions/ex_6/ and local/ex_6/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

OUTPUT_DIR = Path("outputs") / "mlfp02_ex6_logistic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS: list[str] = ["floor_area_sqm", "storey_mid", "remaining_lease"]


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — HDB resale transactions (Singapore)
# ════════════════════════════════════════════════════════════════════════


def load_hdb_recent() -> pl.DataFrame:
    """Load HDB resale transactions filtered to 2020+.

    Returns a polars DataFrame with the raw columns from the dataset
    (resale_price, flat_type, floor_area_sqm, storey_range, lease_commence_date,
    month, etc.). No target construction — see build_classification_frame().
    """
    loader = MLFPDataLoader()
    hdb = loader.load("mlfp01", "hdb_resale.parquet")
    return hdb.filter(pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1))


def build_classification_frame(hdb_recent: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """Add the binary target and engineered features used by logistic regression.

    The target ``high_price`` = 1 if resale_price > median, 0 otherwise — this
    gives a balanced ~50/50 split so the baseline accuracy is 50%.

    Returns (dataframe, median_price).
    """
    median_price = hdb_recent["resale_price"].median()
    frame = hdb_recent.with_columns(
        (pl.col("resale_price") > median_price).cast(pl.Int32).alias("high_price"),
        (
            (
                pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
                + pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
            )
            / 2.0
        ).alias("storey_mid"),
        (
            99
            - (
                pl.col("month").str.to_date("%Y-%m").dt.year()
                - pl.col("lease_commence_date")
            )
        )
        .cast(pl.Float64)
        .alias("remaining_lease"),
    ).drop_nulls(
        subset=["floor_area_sqm", "storey_mid", "high_price", "remaining_lease"]
    )
    return frame, float(median_price)


def build_design_matrix(
    frame: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Standardise features and append an intercept column.

    Returns (X_with_intercept, y, X_mean, X_std, feature_names_with_intercept).
    """
    X_raw = frame.select(FEATURE_COLS).to_numpy().astype(np.float64)
    y = frame["high_price"].to_numpy().astype(np.float64)

    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_scaled = (X_raw - X_mean) / X_std

    n_obs = X_scaled.shape[0]
    X = np.column_stack([np.ones(n_obs), X_scaled])
    feature_names = ["intercept", *FEATURE_COLS]
    return X, y, X_mean, X_std, feature_names


# ════════════════════════════════════════════════════════════════════════
# SIGMOID + BERNOULLI LOG-LIKELIHOOD
# ════════════════════════════════════════════════════════════════════════


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid σ(z) = 1 / (1 + exp(-z)).

    For z ≥ 0 use 1/(1+exp(-z)); for z < 0 use exp(z)/(1+exp(z)). Both
    branches avoid overflow at the extremes.
    """
    z = np.asarray(z, dtype=np.float64)
    result = np.zeros_like(z)
    pos = z >= 0
    neg = ~pos
    result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    result[neg] = exp_z / (1.0 + exp_z)
    return result


def neg_log_likelihood_logistic(
    beta: np.ndarray, X: np.ndarray, y: np.ndarray
) -> float:
    """Negative Bernoulli log-likelihood: -Σ[y log p + (1-y) log(1-p)]."""
    z = X @ beta
    p = sigmoid(z)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(-ll)


def neg_ll_gradient(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of the negative log-likelihood: -X^T (y - p)."""
    z = X @ beta
    p = sigmoid(z)
    return -X.T @ (y - p)


# ════════════════════════════════════════════════════════════════════════
# ORIGINAL-SCALE COEFFICIENT CONVERSION
# ════════════════════════════════════════════════════════════════════════


def unscale_coefficients(
    beta_scaled: np.ndarray, X_mean: np.ndarray, X_std: np.ndarray
) -> np.ndarray:
    """Convert coefficients fit on standardised features to the original scale.

    β_original[j]   = β_scaled[j] / σ_j  for j ≥ 1
    β_original[0]   = β_scaled[0] - Σ β_scaled[j] * μ_j / σ_j
    """
    beta_original = np.zeros_like(beta_scaled)
    beta_original[0] = beta_scaled[0] - float(np.sum(beta_scaled[1:] * X_mean / X_std))
    beta_original[1:] = beta_scaled[1:] / X_std
    return beta_original


# ════════════════════════════════════════════════════════════════════════
# CALIBRATION BINNING
# ════════════════════════════════════════════════════════════════════════


def calibration_bins(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> tuple[list[float], list[float], list[int]]:
    """Equal-width binning over predicted probabilities.

    Returns (mean_predicted, mean_observed, counts) for non-empty bins only.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred: list[float] = []
    mean_obs: list[float] = []
    counts: list[int] = []
    for i in range(n_bins):
        hi = edges[i + 1] + (1e-12 if i == n_bins - 1 else 0.0)
        mask = (p >= edges[i]) & (p < hi)
        if mask.sum() > 0:
            mean_pred.append(float(p[mask].mean()))
            mean_obs.append(float(y[mask].mean()))
            counts.append(int(mask.sum()))
    return mean_pred, mean_obs, counts
