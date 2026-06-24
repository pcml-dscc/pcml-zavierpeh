# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 4: Production Pipeline — Registry, Drift, Deploy

Complete the `solve()` function. Read problem.md for the full specification.
Train a LightGBM model through the kailash-ml `TrainingPipeline`, register and
promote it staging -> production in the `ModelRegistry`, then arm a
`DriftMonitor` and check a clean slice (no alarm) and a shifted slice (drift).
Your submission is auto-graded against an independent re-derivation.

    python grader.py starter.py     # grade your attempt

IMPORTANT — TWO DATABASES: give the ModelRegistry and the DriftMonitor SEPARATE
SQLite files. A model registry and a monitoring store are distinct systems with
independent lifecycles, and using fresh, separate files avoids reusing a stale
database whose schema predates your installed kailash-ml version.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"
REFERENCE_ROWS = 7_500
PSI_THRESHOLD = 0.2
KS_THRESHOLD = 0.05
BASE_FEATURES = [
    "satisfaction_score",
    "avg_order_value",
    "num_returns",
    "order_count",
    "loyalty_int",
    "total_revenue",
    "days_since_last_order",
    "customer_tenure_days",
]


def _model_frame() -> pl.DataFrame:
    """Load N_ROWS, derive ``premium_response`` (given), return the model frame."""
    df = MLFPDataLoader().load("mlfp03", "ecommerce_customers.parquet")
    df = df.sort("customer_id").head(N_ROWS)
    rng = np.random.default_rng(SEED)

    def z(col: str) -> np.ndarray:
        a = df[col].to_numpy().astype(float)
        return (a - a.mean()) / (a.std() + 1e-9)

    loyal = df["loyalty_member"].cast(pl.Int64).to_numpy().astype(float)
    sat_high = (df["satisfaction_score"] >= 4).cast(pl.Int64).to_numpy().astype(float)
    logit = (
        1.0 * z("satisfaction_score")
        + 0.9 * loyal
        + 0.8 * z("avg_order_value")
        - 0.7 * z("num_returns")
        + 0.5 * z("order_count")
        + 1.4 * (loyal * sat_high)
        + rng.normal(0.0, 1.3, size=df.height)
    )
    df = df.with_columns(
        [
            pl.col("loyalty_member").cast(pl.Int64).alias("loyalty_int"),
            pl.Series(TARGET, (logit > 2.0).astype(np.int64)),
            pl.int_range(0, df.height, dtype=pl.Int64).alias("row_id"),
        ]
    )
    return df.select(BASE_FEATURES + ["row_id", TARGET])


def _shift_slice(clean: pl.DataFrame) -> pl.DataFrame:
    """Economic-downturn shift: spend collapses, recency stretches, mood drops.

    Given to you — keep it intact so your drift outcome matches the grader.
    """
    return clean.with_columns(
        [
            (pl.col("avg_order_value") * 0.6).alias("avg_order_value"),
            (pl.col("total_revenue") * 0.6).alias("total_revenue"),
            (pl.col("days_since_last_order") * 1.5 + 60).alias("days_since_last_order"),
            (pl.col("satisfaction_score") - 1).alias("satisfaction_score"),
        ]
    )


async def _run() -> dict:
    frame = _model_frame()
    reference = frame.select(BASE_FEATURES).head(REFERENCE_ROWS)
    clean = frame.select(BASE_FEATURES).tail(frame.height - REFERENCE_ROWS)
    shifted = _shift_slice(clean)

    # TODO 1: import ConnectionManager, DriftMonitor, ModelRegistry,
    #         TrainingPipeline, EvalSpec, ModelSpec, FeatureField, FeatureSchema.
    #         Build a FeatureSchema over BASE_FEATURES (entity_id_column="row_id").
    #
    # TODO 2: open TWO ConnectionManagers on SEPARATE temp SQLite files (one for
    #         the registry, one for the drift monitor). Use a unique filename per
    #         run, e.g. tempfile.gettempdir() + os.getpid() + uuid; clean up in a
    #         finally block.
    #
    # TODO 3: train LightGBM via TrainingPipeline on the registry connection:
    #         model_class="lightgbm.LGBMClassifier", framework="lightgbm",
    #         {n_estimators:200, random_state:42, verbose:-1}; EvalSpec metrics
    #         ["accuracy","f1","auc"], holdout, test_size=0.25.
    #
    # TODO 4: promote the registered version staging -> production with a reason,
    #         then get_model(name, stage="production") to confirm.
    #
    # TODO 5: arm a DriftMonitor on the DRIFT connection
    #         (tenant_id="_single", psi_threshold=PSI_THRESHOLD,
    #         ks_threshold=KS_THRESHOLD). set_reference_data(name, reference,
    #         BASE_FEATURES), then check_drift on `clean` and on `shifted`.
    #
    # TODO 6: return the dict described in the docstring.
    return {}


def solve() -> dict:
    """Train, register, promote to production, and run drift detection.

    Return a dict with: registered_version, production_stage, reference_auc,
    clean_drift_detected, shift_drift_detected, n_drifted_features_clean,
    n_drifted_features_shift, shift_severity.
    """
    return asyncio.run(_run())


if __name__ == "__main__":
    print(solve())
