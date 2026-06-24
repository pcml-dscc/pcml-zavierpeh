# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 3 — Dimensionality Reduction.

Contains: data loading, scaling, common output directory, KMeans-based
silhouette evaluation in the embedding space. Technique-specific code
(PCA/KPCA/t-SNE/UMAP algorithms and their plots) lives in the per-
technique files, NOT here.

    from shared.mlfp04.ex_3 import (
        OUTPUT_DIR, load_customer_matrix, evaluate_embedding_silhouette,
    )
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from kailash_ml import ExperimentTracker
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# OUTPUT + REPRODUCIBILITY
# ════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs") / "ex3_dimreduce"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 4

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — E-commerce customers (reused from MLFP03)
# ════════════════════════════════════════════════════════════════════════


def load_customer_matrix() -> tuple[np.ndarray, list[str], pl.DataFrame]:
    """Load e-commerce customers, standardise numeric features.

    Returns:
        X          : (n_samples, n_features) standardised float matrix
        feature_cols: list of feature column names in order
        df_raw     : the raw polars DataFrame before scaling
    """
    loader = MLFPDataLoader()
    customers = loader.load("mlfp03", "ecommerce_customers.parquet")

    feature_cols = [
        c
        for c, d in zip(customers.columns, customers.dtypes)
        if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        and c not in ("customer_id",)
    ]

    df_clean = customers.drop_nulls(subset=feature_cols)
    X_raw, _, _ = to_sklearn_input(df_clean, feature_columns=feature_cols)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    return X, feature_cols, df_clean


# ════════════════════════════════════════════════════════════════════════
# EMBEDDING-SPACE CLUSTER QUALITY
# ════════════════════════════════════════════════════════════════════════


def evaluate_embedding_silhouette(
    embedding: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> float:
    """Fit KMeans in the embedding space and return the silhouette score.

    This is the standard "does the reducer preserve structure?" probe used
    across all five technique files. Returns -1.0 when only one cluster is
    found (e.g. collapsed embedding).
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5)
    labels = km.fit_predict(embedding)
    if len(set(labels)) < 2:
        return -1.0
    return float(silhouette_score(embedding, labels))


# ════════════════════════════════════════════════════════════════════════
# SUBSAMPLING — used by KPCA / t-SNE / UMAP / Isomap for kernel-cost paths
# ════════════════════════════════════════════════════════════════════════


def subsample_indices(
    n_samples: int, n_target: int, random_state: int = RANDOM_STATE
) -> np.ndarray:
    """Deterministic subsample indices for expensive O(n^2) methods."""
    rng = np.random.default_rng(random_state)
    return rng.choice(n_samples, min(n_target, n_samples), replace=False)


# ════════════════════════════════════════════════════════════════════════
# KAILASH-ML EXPERIMENT TRACKER — shared by every dim-reduction technique
# ════════════════════════════════════════════════════════════════════════
# Every M4 ex_3 lesson logs its sweep + final-fit metrics to a single
# SQLite store so students can compare PCA / Kernel-PCA / t-SNE / UMAP
# embedding-quality runs after the lesson group ends. Mirrors the ex_1
# clustering-zoo pattern; separate DB so dim-reduction has its own
# leaderboard distinct from clustering.

DIMREDUCE_DB = "sqlite:///mlfp04_ex3_dimreduction.db"
EXPERIMENT_NAME = "m4_dimreduction_zoo"


async def _setup_engines_async() -> tuple[ExperimentTracker, str]:
    """Open the dim-reduction ExperimentTracker (kailash-ml 1.5.1)."""
    tracker = await ExperimentTracker.create(store_url=DIMREDUCE_DB)
    return tracker, EXPERIMENT_NAME


def setup_engines() -> tuple[ExperimentTracker, str]:
    """Sync wrapper. Returns (tracker, experiment_name)."""
    return asyncio.run(_setup_engines_async())


def teardown_engines(tracker: ExperimentTracker) -> None:
    """Drain the aiosqlite worker threads before the script returns.

    kailash's AsyncSQLitePool spawns NON-DAEMON aiosqlite worker threads on
    first pool use. Python 3.13's ``Py_FinalizeEx`` joins non-daemon threads
    BEFORE running ``atexit`` handlers, so an atexit-based close runs too
    late — the interpreter hangs forever in ``wait_for_thread_shutdown``
    waiting on workers stuck in ``queue.get()``.

    Solutions MUST call ``teardown_engines(tracker)`` after the REFLECTION
    block. See ``rules/patterns.md`` § "Async Resource Cleanup".
    """
    asyncio.run(tracker.close())


async def _track_run_async(
    tracker: ExperimentTracker,
    exp_name: str,
    run_name: str,
    params: dict[str, Any],
    scalar_metrics: dict[str, float],
    series_metrics: dict[str, list[float]] | None = None,
) -> None:
    """Log one lesson's run: scalar metrics + optional per-step series."""
    async with tracker.track(experiment=exp_name, run_name=run_name) as run:
        await run.log_params({k: str(v) for k, v in params.items()})
        for name, value in scalar_metrics.items():
            await run.log_metric(name, float(value))
        if series_metrics:
            for name, values in series_metrics.items():
                for step, value in enumerate(values, start=1):
                    await run.log_metric(name, float(value), step=step)


def track_run(
    tracker: ExperimentTracker,
    exp_name: str,
    run_name: str,
    params: dict[str, Any],
    scalar_metrics: dict[str, float],
    series_metrics: dict[str, list[float]] | None = None,
) -> None:
    """Sync wrapper for logging a single technique's run."""
    asyncio.run(
        _track_run_async(
            tracker, exp_name, run_name, params, scalar_metrics, series_metrics
        )
    )
