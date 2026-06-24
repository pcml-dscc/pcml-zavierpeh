# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.2: DataFlow Persistence with @db.model
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Declare a database table with `@db.model` (schema = Python class)
#   - Let DataFlow auto-migrate the schema — no hand-written DDL
#   - Persist evaluation records with `db.express.create` (~23x faster
#     than driving CreateNode through WorkflowBuilder for single records)
#   - Query persisted results back with `db.express.list`
#   - Understand why compliance-grade ML needs a first-class result store
#
# PREREQUISITES: 01_workflow_builder.py
# ESTIMATED TIME: ~35 min
#
# 5-PHASE R10:
#   1. Theory     — why "write metrics to a database" is a regulatory must
#   2. Build      — @db.model classes for ModelEvaluation + ModelArtifact
#   3. Train      — train once and persist the result through db.express
#   4. Visualise  — list persisted records and render them
#   5. Apply      — Singapore MAS Notice 635 evidence store
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
# The `print()` statement is the cheapest, worst form of observability
# for ML. In a regulated environment (MAS Notice 635, EBA, Fed SR 11-7)
# an auditor can walk in six months after a loan was declined and ask:
#
#   "Show me the model that was used for this decision, the version,
#    the exact metrics on the evaluation set at the time of promotion,
#    and the hyperparameters the training run used."
#
# If your answer is "let me grep some log files", you have already lost.
# The compliance answer is a database table — queryable, append-only,
# indexed by model_name + version — that records every evaluation, every
# artifact, and every stage transition. DataFlow turns that table from
# a week of SQL-migration work into a decorated Python class.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD @db.model tables
# ════════════════════════════════════════════════════════════════════════

db = DataFlow(DB_URL)


@db.model
class ModelEvaluation:
    """Stores evaluation metrics for every trained model version.

    Note: the modern ``@db.model`` decorator auto-generates the primary
    key (named ``id``) — no explicit ``field(primary_key=True)``
    declaration is required. Plain class annotations ARE the schema.
    Defaults (e.g. ``hyperparameters: str = "{}"``) are respected.
    """

    id: int
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
    """Initialise DataFlow, write one evaluation row and one artefact row."""
    await db.initialize()

    eval_row = await db.express.create(
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
# Modern db.express.create returns {<fields...>, "rows_affected": 1}
# — it does NOT echo the auto-generated primary key in the response
# payload. The authoritative contract is `rows_affected == 1`. The auto
# id is visible when we list the table below.
assert (
    eval_record is not None and eval_record.get("rows_affected") == 1
), "Task 3: DataFlow should insert exactly one evaluation row"
assert (
    artifact_record is not None and artifact_record.get("rows_affected") == 1
), "Task 3: DataFlow should insert exactly one artefact row"
print(f"\n  ModelEvaluation row written: {eval_record.get('model_name')}")
print(f"  ModelArtifact   row written: {artifact_record.get('model_name')}")
print("\n[ok] Checkpoint passed — DataFlow persistence verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE persisted records by reading them back
# ════════════════════════════════════════════════════════════════════════
# Any persistence code you cannot READ BACK is a black hole. The audit
# test is "I wrote it, I can list it, I can filter it."


async def list_persisted() -> tuple[list[dict], list[dict]]:
    evals = await db.express.list("ModelEvaluation")
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
# TASK 5 — APPLY: Singapore MAS Notice 635 Evidence Store
# ════════════════════════════════════════════════════════════════════════
# MAS Notice 635 (Credit Card and Unsecured Credit Rules) requires every
# lender using an ML model in a credit decision to show, for each
# decision, the model that produced the recommendation, the version, the
# evaluation metrics at the time of promotion, and the training data
# snapshot reference.
#
# Before this file, the bank's compliance team spent ~160 analyst hours
# per month reconciling ad-hoc notebooks into evidence packs. After this
# file, the evidence pack is a single SELECT over the ModelEvaluation
# and ModelArtifact tables.
print("\n" + "=" * 70)
print("  APPLY: MAS Notice 635 Evidence Store")
print("=" * 70)
print(headline_roi_text())
print(
    "\n  The `Audit prep savings` line above is UNLOCKED by this file."
    "\n  Without persisted @db.model records, the savings are S$0."
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Declared database tables with the @db.model decorator
  [x] Used db.express.create for 23x-faster single-record CRUD
  [x] Queried persisted evaluations with db.express.list
  [x] Tied the persistence layer to a real MAS compliance scenario

  KEY INSIGHT: If your metrics only exist in a print() call, they don't
  exist from an auditor's perspective. DataFlow turns "write it to a DB"
  from a week of schema work into a one-line decorator.

  Next: 03_hyperparameter_search.py — use Bayesian search to improve the
  metrics you just learned how to persist.

  DB URL: {DB_URL}
  Portfolio anchor: S${SG_BANK_PORTFOLIO['portfolio_sgd']/1e9:.0f}B
"""
)
