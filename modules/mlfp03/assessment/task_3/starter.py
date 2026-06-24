# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 3: Evaluation, Class Imbalance & Interpretability

Complete the `solve()` function. Read problem.md for the full specification.
Train a baseline and a class-balanced RandomForest through the kailash-ml
`TrainingPipeline`, evaluate per-class behaviour with `km.diagnose`, and explain
the balanced model with `ModelExplainer` (SHAP). Your submission is auto-graded
against an independent re-derivation.

    python grader.py starter.py     # grade your attempt
"""
from __future__ import annotations

import asyncio
import pickle
import warnings

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"
TOP_K = 6
SHAP_BACKGROUND = 64
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


def _holdout_test(frame: pl.DataFrame) -> pl.DataFrame:
    """Reproduce TrainingPipeline's deterministic holdout split (test portion).

    Use this same X_test / y_test for km.diagnose so per-class recall lines up
    with the engine's evaluation. Given to you — do not change.
    """
    n = frame.height
    idx = np.arange(n)
    np.random.RandomState(42).shuffle(idx)
    split_idx = int(n * 0.75)
    return frame[idx[split_idx:].tolist()]


async def _run() -> dict:
    # TODO 1: import ConnectionManager, ModelRegistry, TrainingPipeline,
    #         ModelExplainer, diagnose, EvalSpec, ModelSpec, FeatureField,
    #         FeatureSchema. Build the frame, schema (entity_id_column="row_id"),
    #         and EvalSpec(metrics=["accuracy","f1","auc"], holdout, 0.25).
    #
    # TODO 2: train TWO RandomForests via TrainingPipeline:
    #         - baseline: {n_estimators:150, random_state:42, n_jobs:-1}
    #         - balanced: same + class_weight="balanced"
    #
    # TODO 3: load both fitted models from the registry
    #         (pickle.loads(await registry.load_artifact(name, version))).
    #
    # TODO 4: evaluate per-class behaviour with
    #         diagnose(model, kind="classical_classifier", data=(X_test,y_test),
    #         show=False). The minority (positive) class is key "1.0":
    #         report.per_class["1.0"]["recall"]. Macro recall + accuracy live in
    #         report.metrics ("recall_macro", "accuracy").
    #
    # TODO 5: interpret the BALANCED model with ModelExplainer:
    #         ModelExplainer(model=bal_model,
    #                        X=frame.select(BASE_FEATURES).head(SHAP_BACKGROUND),
    #                        feature_names=BASE_FEATURES).explain_global(
    #                        max_display=TOP_K)["feature_importance"] -> dict
    #         ordered by importance; take the first TOP_K keys.
    #
    # TODO 6: return the dict described in the docstring (remember to close the
    #         connection in a finally block).
    return {}


def solve() -> dict:
    """Evaluate imbalance handling + interpret the balanced model.

    Return a dict with: baseline_minority_recall, balanced_minority_recall,
    baseline_recall_macro, balanced_recall_macro, baseline_accuracy,
    balanced_accuracy, roc_auc, top_features (top-6 by SHAP), n_features.
    """
    return asyncio.run(_run())


if __name__ == "__main__":
    print(solve())
