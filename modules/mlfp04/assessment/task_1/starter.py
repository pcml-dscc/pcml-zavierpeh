# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 1: Customer Segmentation by Clustering

Complete the `solve()` function. Read problem.md for the full specification.
Framework-first: clustering MUST run through kailash-ml ClusteringEngine.

    python grader.py starter.py     # grade your attempt
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

SEED = 20260401
FEATURES = [
    "recency_days",
    "frequency",
    "monetary_sgd",
    "tenure_months",
    "avg_basket_sgd",
]


def make_customers() -> pl.DataFrame:
    """Deterministic loyalty cohort with four planted spending personas.

    Do NOT change the seed or sizes — the grader regenerates this exact table.
    """
    rng = np.random.default_rng(SEED)
    centers = np.array(
        [
            [5.0, 50.0, 2000.0, 60.0, 180.0],
            [40.0, 10.0, 400.0, 12.0, 60.0],
            [90.0, 3.0, 150.0, 48.0, 45.0],
            [15.0, 25.0, 1200.0, 36.0, 220.0],
        ]
    )
    spreads = np.array([3.0, 3.0, 120.0, 5.0, 18.0])
    sizes = [320, 300, 280, 300]
    blocks = []
    for c, n in zip(centers, sizes):
        blocks.append(c + rng.normal(0, 1, (n, 5)) * spreads)
    X = np.vstack(blocks)
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    return pl.DataFrame({col: X[:, j] for j, col in enumerate(FEATURES)})


def solve() -> dict:
    """Recover the planted personas with the kailash-ml ClusteringEngine."""
    df = make_customers()

    zdf = df.select(
        [
            ((pl.col(feature) - pl.col(feature).mean()) / pl.col(feature).std()).alias(
                feature
            )
            for feature in FEATURES
        ]
    )
    engine = ClusteringEngine()
    sweep = engine.sweep_k(
        zdf,
        range(2, 9),
        algorithm="kmeans",
        criterion="silhouette",
    )
    result = engine.fit(zdf, algorithm="kmeans", n_clusters=sweep.optimal_k)

    return {
        "labels": [int(label) for label in result.labels],
        "n_clusters": int(result.n_clusters),
        "silhouette": float(result.silhouette_score),
    }


if __name__ == "__main__":
    print(solve())
