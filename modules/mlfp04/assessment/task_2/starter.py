# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 2: Dimensionality Reduction & Anomaly Detection

Complete the `solve()` function. Read problem.md for the full specification.
Framework-first: use DimReductionEngine (PCA) and AnomalyDetectionEngine.

    python grader.py starter.py     # grade your attempt
"""
from __future__ import annotations

import numpy as np
import polars as pl

from kailash_ml.engines.anomaly_detection import AnomalyDetectionEngine
from kailash_ml.engines.dim_reduction import DimReductionEngine

SEED = 20260402
N_NORMAL = 975
N_ANOM = 25
D = 24
K_LATENT = 3
CONTAMINATION = 0.025


def make_sensor_matrix() -> pl.DataFrame:
    """Deterministic 24-channel telemetry on a 3-factor manifold + 25 outliers.

    Do NOT change the seed or sizes — the grader regenerates this exact table
    (and the hidden anomaly flags it never gave you).
    """
    rng = np.random.default_rng(SEED)
    Z = rng.normal(0, 1, (N_NORMAL, K_LATENT))
    W = rng.normal(0, 1, (K_LATENT, D)) * 3.5
    X_normal = Z @ W + rng.normal(0, 0.5, (N_NORMAL, D))
    X_anom = rng.normal(12.0, 4.0, (N_ANOM, D)) * rng.choice([-1, 1], (N_ANOM, D))
    X = np.vstack([X_normal, X_anom])
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    cols = [f"f{i:02d}" for i in range(D)]
    return pl.DataFrame({c: X[:, j] for j, c in enumerate(cols)})


def solve() -> dict:
    """Compress with PCA and flag off-manifold rows — kailash-ml engines."""
    df = make_sensor_matrix()

    # TODO 1: DimReductionEngine().reduce(df, algorithm="pca", n_components=df.width);
    #         read explained_variance_ratio, take the cumulative sum, and find the
    #         smallest count of components reaching >= 0.90. Call it n_components_90.
    # TODO 2: reduce(df, algorithm="pca", n_components=n_components_90); read
    #         reconstruction_error off the DimReductionResult.
    # TODO 3: AnomalyDetectionEngine().detect(df, algorithm="isolation_forest",
    #         contamination=CONTAMINATION); read scores and labels.
    # TODO 4: Build anomaly_labels (1 = anomaly, 0 = normal). The engine flags
    #         anomalies with label == -1.
    # TODO 5: Return the dict described in problem.md (5 keys).

    return {
        "n_components_90": df.width,
        "reconstruction_error": 0.0,
        "anomaly_scores": [0.0] * df.height,
        "anomaly_labels": [0] * df.height,
        "n_anomalies": 0,
    }


if __name__ == "__main__":
    print(solve())
