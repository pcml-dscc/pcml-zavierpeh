# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.2: DataFlow Persistence with @db.model
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Declare a database table with `@db.model`
#   - Persist evaluation records with `db.express.create`
#   - Query persisted results back with `db.express.list`
#   - Understand why compliance ML needs a first-class result store
#
# PREREQUISITES: 01_workflow_builder.py
# ESTIMATED TIME: ~35 min
#
# 5-PHASE R10:
#   1. Theory     — why "write metrics to a database" is a regulatory must
#   2. Build      — @db.model classes for ModelEvaluation + ModelArtifact
#   3. Train      — train once and persist via db.express
#   4. Visualise  — list persisted records
#   5. Apply      — MAS Notice 635 evidence store
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json

import lightgbm as lgb

from dataflow import DataFlow

from shared.mlfp03.ex_7 import (
    DB_URL,
    RANDOM_SEED,
    SG_BANK_PORTFOLIO,
    compute_classification_metrics,
    headline_roi_text,
    prepare_credit_split,
    print_metric_block,
    scale_pos_weight_for,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why persist metrics?
# ════════════════════════════════════════════════════════════════════════
# `print()` is the cheapest, worst form of observability. Under MAS
# Notice 635, an auditor can ask for the metrics of any model that
# went into production six months ago. A database table is the only
# defensible answer.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD @db.model tables
# ════════════════════════════════════════════════════════════════════════

# TODO: instantiate a DataFlow with DB_URL
# Hint: db = DataFlow(DB_URL)
db = ____


@db.model
class ModelEvaluation:
    """Stores evaluation metrics for every trained model version.

    Note: the modern ``@db.model`` decorator AUTO-GENERATES the primary
    key (named ``id``) from a plain ``id: int`` annotation — no helper
    call is needed. Plain class annotations ARE the schema. Defaults
    are written as ordinary Python defaults (e.g. ``... = "{}"``).
    """

    # TODO: declare the primary-key column
    # Hint: a plain annotation `id: int` — @db.model auto-generates the PK
    id: ____
    model_name: str
    dataset: str
    accuracy: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    log_loss_val: float
    train_size: int
    test_size: int
    feature_count: int
    hyperparameters: str = "{}"


@db.model
class ModelArtifact:
    """Stores the artefact pointer + lifecycle stage for each model."""

    id: int
    model_name: str
    version: int
    artifact_path: str
    is_production: bool = False
    created_by: str = "mlfp03_ex7"


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN once then persist the result
# ════════════════════════════════════════════════════════════════════════

split = prepare_credit_split()
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=scale_pos_weight_for(split.y_train),
    random_state=RANDOM_SEED,
    verbose=-1,
)
model.fit(split.X_train, split.y_train)
y_pred = model.predict(split.X_test)
y_proba = model.predict_proba(split.X_test)[:, 1]
metrics = compute_classification_metrics(split.y_test, y_pred, y_proba)
print_metric_block("Trained LightGBM — metrics ready to persist", metrics)


async def persist_evaluation() -> tuple[dict, dict]:
    # TODO: initialise DataFlow before the first express call
    # Hint: await db.initialize()
    await ____

    # TODO: use db.express.create to write a ModelEvaluation row
    # Hint: await db.express.create("ModelEvaluation", {...})
    eval_row = await ____(
        "ModelEvaluation",
        {
            "model_name": "lgbm_credit_v1",
            "dataset": "sg_credit_scoring",
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1"],
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "log_loss_val": metrics["log_loss"],
            "train_size": split.train_size,
            "test_size": split.test_size,
            "feature_count": split.feature_count,
            "hyperparameters": json.dumps(model.get_params(), default=str),
        },
    )

    artifact_row = await db.express.create(
        "ModelArtifact",
        {
            "model_name": "lgbm_credit_v1",
            "version": 1,
            "artifact_path": "models/lgbm_credit_v1.pkl",
            "is_production": False,
            "created_by": "mlfp03_ex7",
        },
    )
    return eval_row, artifact_row


eval_record, artifact_record = asyncio.run(persist_evaluation())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    eval_record is not None and "id" in eval_record
), "Task 3: DataFlow should return an evaluation row with an auto-assigned id"
assert (
    artifact_record is not None and "id" in artifact_record
), "Task 3: DataFlow should return an artefact row with an auto-assigned id"
print(f"\n  ModelEvaluation.id = {eval_record['id']}")
print(f"  ModelArtifact.id   = {artifact_record['id']}")
print("\n[ok] Checkpoint passed — DataFlow persistence verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE persisted records
# ════════════════════════════════════════════════════════════════════════


async def list_persisted() -> tuple[list[dict], list[dict]]:
    # TODO: list the two tables you just wrote
    # Hint: await db.express.list("ModelEvaluation")
    evals = await ____("ModelEvaluation")
    artifacts = await db.express.list("ModelArtifact")
    return evals, artifacts


persisted_evals, persisted_artifacts = asyncio.run(list_persisted())

print(f"=== Persisted Evaluations ({len(persisted_evals)}) ===")
for row in persisted_evals:
    print(
        f"  {row['model_name']}: AUC-ROC={row['auc_roc']:.4f}"
        f"  AUC-PR={row['auc_pr']:.4f}"
        f"  train={row['train_size']}  test={row['test_size']}"
    )

print(f"\n=== Persisted Artifacts ({len(persisted_artifacts)}) ===")
for row in persisted_artifacts:
    stage = "PRODUCTION" if row.get("is_production") else "staging"
    print(f"  {row['model_name']} v{row['version']}: {stage}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS Notice 635 Evidence Store
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  APPLY: MAS Notice 635 Evidence Store")
print("=" * 70)
print(headline_roi_text())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Declared database tables with the @db.model decorator
  [x] Used db.express.create for fast single-record CRUD
  [x] Queried persisted evaluations with db.express.list
  [x] Tied persistence to a real MAS compliance scenario

  Next: 03_hyperparameter_search.py — improve the metrics you persist.
  DB URL: {DB_URL}
  Portfolio anchor: S${SG_BANK_PORTFOLIO['portfolio_sgd']/1e9:.0f}B
"""
)
