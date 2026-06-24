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

import warnings

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

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

    # TODO 1: Engineer the six features with the EXACT formulas in problem.md:
    #         revenue_per_order, returns_per_order, is_satisfied,
    #         loyal_and_satisfied, tenure_years, spend_per_tenure_day.
    #         Hint: df.with_columns([... .alias("revenue_per_order"), ...])

    # TODO 2: Build feature_matrix = the 14 candidate columns
    #         (BASE_FEATURES + ENGINEERED_FEATURES) plus the TARGET column.
    #         Do NOT include customer_id, review_text, ltv_tier, or churned.

    # TODO 3: Take the TRAIN split only (first TRAIN_FRACTION of the rows) so
    #         no test-set signal leaks into selection.

    # TODO 4: Rank features with kailash-ml FeatureEngineer. Build a
    #         GeneratedFeatures candidate set (original_columns=BASE_FEATURES,
    #         generated_columns=[GeneratedColumn(...) for each engineered
    #         feature]) then call FeatureEngineer(...).select(train, gen,
    #         target=TARGET, method="importance", top_k=TOP_K).
    #         Hint: from kailash_ml.engines.feature_engineer import (
    #                   FeatureEngineer, GeneratedColumn, GeneratedFeatures)

    # TODO 5: Return the required dict.
    return {
        "feature_matrix": df,  # <- replace with the 14-column candidate matrix
        "engineered_columns": [],
        "selected_features": [],
        "target_column": TARGET,
    }


if __name__ == "__main__":
    out = solve()
    print(out["selected_features"])
