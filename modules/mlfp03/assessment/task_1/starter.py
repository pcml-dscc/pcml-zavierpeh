# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 1: Feature Engineering & Leakage-Free Selection

Complete the `solve()` function. Read problem.md for the full specification:
the derived target, the six exact engineered-feature formulas, the candidate
pool, and the leakage-free selection contract. Your submission is auto-graded
against an independent reference — every wrong formula, leaked column, or
mis-ranked feature fails a check.

    python grader.py starter.py     # grade your attempt
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import polars as pl
from kailash_ml.engines.feature_engineer import (
    FeatureEngineer,
    GeneratedColumn,
    GeneratedFeatures,
)

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# ── Deterministic contract (do not change) ────────────────────────────────
N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"
TRAIN_FRACTION = 0.75
TOP_K = 8

BASE_FEATURES = [
    "total_revenue",
    "order_count",
    "avg_order_value",
    "days_since_last_order",
    "customer_tenure_days",
    "satisfaction_score",
    "num_returns",
    "loyalty_int",
]
ENGINEERED_FEATURES = [
    "revenue_per_order",
    "returns_per_order",
    "is_satisfied",
    "loyal_and_satisfied",
    "tenure_years",
    "spend_per_tenure_day",
]


def _load_base() -> pl.DataFrame:
    """Load the first N_ROWS (sorted by customer_id) and derive the target.

    The derived target ``premium_response`` is given to you in full — keep it
    EXACTLY as written so your output matches the grader's reference.
    """
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
    target = (logit > 2.0).astype(np.int64)

    return df.with_columns(
        [
            pl.col("loyalty_member").cast(pl.Int64).alias("loyalty_int"),
            pl.Series(TARGET, target),
        ]
    )


def solve() -> dict:
    """Engineer features then rank them leakage-free with FeatureEngineer.

    Return a dict with keys: feature_matrix (pl.DataFrame of the 14 candidate
    columns + target), engineered_columns (list of 6), selected_features
    (top-8 by importance), target_column. See problem.md for exact formulas.
    """
    df = _load_base()

    df = df.with_columns(
        [
            (pl.col("total_revenue") / pl.col("order_count")).alias("revenue_per_order"),
            (pl.col("num_returns") / pl.col("order_count")).alias("returns_per_order"),
            (pl.col("satisfaction_score") >= 4).cast(pl.Int64).alias("is_satisfied"),
            (
                pl.col("loyalty_int")
                * (pl.col("satisfaction_score") >= 4).cast(pl.Int64)
            ).alias("loyal_and_satisfied"),
            (pl.col("customer_tenure_days") / 365.0).alias("tenure_years"),
            (pl.col("total_revenue") / pl.col("customer_tenure_days")).alias(
                "spend_per_tenure_day"
            ),
        ]
    )

    feature_matrix = df.select(BASE_FEATURES + ENGINEERED_FEATURES + [TARGET])
    train_rows = int(feature_matrix.height * TRAIN_FRACTION)
    train = feature_matrix.head(train_rows)

    generated_columns = [
        GeneratedColumn(
            name="revenue_per_order",
            source_columns=["total_revenue", "order_count"],
            strategy="ratio",
            dtype="float",
        ),
        GeneratedColumn(
            name="returns_per_order",
            source_columns=["num_returns", "order_count"],
            strategy="ratio",
            dtype="float",
        ),
        GeneratedColumn(
            name="is_satisfied",
            source_columns=["satisfaction_score"],
            strategy="threshold",
            dtype="int",
        ),
        GeneratedColumn(
            name="loyal_and_satisfied",
            source_columns=["loyalty_int", "satisfaction_score"],
            strategy="interaction",
            dtype="int",
        ),
        GeneratedColumn(
            name="tenure_years",
            source_columns=["customer_tenure_days"],
            strategy="scaling",
            dtype="float",
        ),
        GeneratedColumn(
            name="spend_per_tenure_day",
            source_columns=["total_revenue", "customer_tenure_days"],
            strategy="ratio",
            dtype="float",
        ),
    ]
    candidates = GeneratedFeatures(
        original_columns=BASE_FEATURES,
        generated_columns=generated_columns,
        total_candidates=len(BASE_FEATURES) + len(generated_columns),
        data=train,
    )
    selected = FeatureEngineer(feature_store=None, max_features=TOP_K).select(
        train,
        candidates,
        target=TARGET,
        method="importance",
        top_k=TOP_K,
    )

    return {
        "feature_matrix": feature_matrix,
        "engineered_columns": ENGINEERED_FEATURES,
        "selected_features": selected.selected_columns,
        "target_column": TARGET,
    }


if __name__ == "__main__":
    out = solve()
    print(out["selected_features"])
