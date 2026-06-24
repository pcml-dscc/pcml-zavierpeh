# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 2: The Model Zoo

Complete the `solve()` function. Read problem.md for the full specification:
the six required algorithms, the deterministic data contract, and the exact
comparison-table schema. Train every model through the kailash-ml
`TrainingPipeline` (no raw `.fit()`). Your submission is auto-graded — the
grader independently re-trains one model to verify your table is real.

    python grader.py starter.py     # grade your attempt
"""
from __future__ import annotations

import asyncio
import warnings

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"
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

# The six required algorithms. Keep these names as the "model" column values.
MODEL_ZOO: dict[str, tuple[str, str, dict]] = {
    "logistic_regression": (
        "sklearn.linear_model.LogisticRegression",
        "sklearn",
        {"max_iter": 2000, "random_state": SEED},
    ),
    "naive_bayes": ("sklearn.naive_bayes.GaussianNB", "sklearn", {}),
    "decision_tree": (
        "sklearn.tree.DecisionTreeClassifier",
        "sklearn",
        {"max_depth": 6, "random_state": SEED},
    ),
    "random_forest": (
        "sklearn.ensemble.RandomForestClassifier",
        "sklearn",
        {"n_estimators": 150, "random_state": SEED, "n_jobs": -1},
    ),
    "extra_trees": (
        "sklearn.ensemble.ExtraTreesClassifier",
        "sklearn",
        {"n_estimators": 150, "random_state": SEED, "n_jobs": -1},
    ),
    "lightgbm": (
        "lightgbm.LGBMClassifier",
        "lightgbm",
        {"n_estimators": 200, "random_state": SEED, "verbose": -1},
    ),
}


def _model_frame() -> pl.DataFrame:
    """Load N_ROWS, derive ``premium_response`` (given), return the model frame.

    The frame holds the 8 base features + ``row_id`` (entity id) + the target.
    Keep the derived-target block exactly as written.
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
    df = df.with_columns(
        [
            pl.col("loyalty_member").cast(pl.Int64).alias("loyalty_int"),
            pl.Series(TARGET, (logit > 2.0).astype(np.int64)),
            pl.int_range(0, df.height, dtype=pl.Int64).alias("row_id"),
        ]
    )
    return df.select(BASE_FEATURES + ["row_id", TARGET])


async def _run_zoo() -> list[dict]:
    # TODO 1: import ConnectionManager, ModelRegistry, TrainingPipeline,
    #         EvalSpec, ModelSpec, FeatureField, FeatureSchema.
    # TODO 2: build the model frame and a FeatureSchema over BASE_FEATURES with
    #         entity_id_column="row_id".
    # TODO 3: open an in-memory ConnectionManager, build a TrainingPipeline
    #         (feature_store=None, registry=ModelRegistry(conn)).
    # TODO 4: for each (name, (model_class, framework, hp)) in MODEL_ZOO, call
    #         await pipeline.train(...) with EvalSpec metrics
    #         ["accuracy","f1","auc"], split "holdout", test_size=0.25.
    #         Collect {"model","accuracy","f1","auc"} per model.
    #         Hint: always close the connection in a finally block.
    return []


def solve() -> pl.DataFrame:
    """Train the six-model zoo; return [model, accuracy, f1, auc] sorted by auc desc."""
    rows = asyncio.run(_run_zoo())
    if not rows:
        # placeholder so the grader runs (and fails) cleanly
        return pl.DataFrame({"model": [], "accuracy": [], "f1": [], "auc": []})
    return pl.DataFrame(rows).sort("auc", descending=True)


if __name__ == "__main__":
    print(solve())
