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
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

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

    from kailash.db import ConnectionManager
    from kailash_ml import ModelRegistry, TrainingPipeline
    from kailash_ml.engines.drift_monitor import DriftMonitor
    from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
    from kailash_ml.types import FeatureField, FeatureSchema

    schema = FeatureSchema(
        name="premium_model_input",
        features=[FeatureField(name=feature, dtype="float64") for feature in BASE_FEATURES],
        entity_id_column="row_id",
    )
    eval_spec = EvalSpec(
        metrics=["accuracy", "f1", "auc"],
        split_strategy="holdout",
        test_size=0.25,
    )

    run_id = f"{os.getpid()}_{uuid.uuid4().hex}"
    temp_dir = Path(tempfile.gettempdir())
    registry_path = temp_dir / f"mlfp03_task4_registry_{run_id}.db"
    drift_path = temp_dir / f"mlfp03_task4_drift_{run_id}.db"
    registry_conn = ConnectionManager(f"sqlite:///{registry_path.as_posix()}")
    drift_conn = ConnectionManager(f"sqlite:///{drift_path.as_posix()}")

    await registry_conn.initialize()
    await drift_conn.initialize()
    registry = ModelRegistry(registry_conn)
    pipeline = TrainingPipeline(feature_store=None, registry=registry)
    monitor = DriftMonitor(
        drift_conn,
        tenant_id="_single",
        psi_threshold=PSI_THRESHOLD,
        ks_threshold=KS_THRESHOLD,
    )
    model_name = "premium_lightgbm_production"
    try:
        result = await pipeline.train(
            data=frame,
            schema=schema,
            model_spec=ModelSpec(
                model_class="lightgbm.LGBMClassifier",
                framework="lightgbm",
                hyperparameters={
                    "n_estimators": 200,
                    "random_state": SEED,
                    "verbose": -1,
                },
            ),
            eval_spec=eval_spec,
            experiment_name=model_name,
        )
        registered_version = result.model_version.version
        await registry.promote_model(
            name=model_name,
            version=registered_version,
            target_stage="production",
            reason=(
                f"Promote premium response model after validation: "
                f"AUC={result.metrics.get('auc', 0.0):.4f}."
            ),
        )
        production_model = await registry.get_model(model_name, stage="production")

        await monitor.set_reference_data(model_name, reference, BASE_FEATURES)
        clean_report = await monitor.check_drift(model_name, clean)
        shifted_report = await monitor.check_drift(model_name, shifted)

        clean_drifted = [
            feature
            for feature in clean_report.feature_results
            if feature.drift_detected
        ]
        shifted_drifted = [
            feature
            for feature in shifted_report.feature_results
            if feature.drift_detected
        ]
        return {
            "registered_version": int(registered_version),
            "production_stage": production_model.stage,
            "reference_auc": float(result.metrics["auc"]),
            "clean_drift_detected": bool(clean_report.overall_drift_detected),
            "shift_drift_detected": bool(shifted_report.overall_drift_detected),
            "n_drifted_features_clean": len(clean_drifted),
            "n_drifted_features_shift": len(shifted_drifted),
            "shift_severity": shifted_report.overall_severity,
        }
    finally:
        await registry_conn.close()
        await drift_conn.close()
        for path in (registry_path, drift_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def solve() -> dict:
    """Train, register, promote to production, and run drift detection.

    Return a dict with: registered_version, production_stage, reference_auc,
    clean_drift_detected, shift_drift_detected, n_drifted_features_clean,
    n_drifted_features_shift, shift_severity.
    """
    return asyncio.run(_run())


if __name__ == "__main__":
    print(solve())
