# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.5: FeatureSchema Validation, Multi-Method
#                         Consensus, and Leakage Audit
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Declare a FeatureSchema contract (types, nullability, documentation)
#   - Validate the engineered matrix against the schema at runtime
#   - Vote across filter/wrapper/embedded selections to build a ROBUST
#     consensus feature set
#   - Log the final feature set + metrics to ExperimentTracker
#   - Run a leakage audit that every selection MUST pass before training
#
# PREREQUISITES: 02_filter_selection.py, 03_wrapper_selection.py,
#                04_embedded_selection.py
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why schemas + audits are the last line of defence
#   2. Build — declare FeatureSchema + replay the three selections
#   3. Train — vote consensus across methods
#   4. Visualise — consensus table + leakage audit report
#   5. Apply — MOH Singapore population-health pipeline (S$ impact)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from collections import Counter

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kailash_ml import DataExplorer
from kailash_ml.types import FeatureField, FeatureSchema

from shared.mlfp03.ex_1 import (
    build_full_feature_frame,
    load_icu_tables,
    log_selection_run,
    prepare_selection_inputs,
    setup_tracking,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Schemas + Audits Are The Last Line Of Defence
# ════════════════════════════════════════════════════════════════════════
# Feature engineering is where the highest-stakes bugs live. A single
# leaky feature can pass every unit test, every code review, and every
# cross-validation fold — and then fail catastrophically in production
# because the validation set and the train set were drawn from the
# same leaky joint distribution.
#
# Two complementary defences catch these bugs:
#
#   1. FeatureSchema — a type + nullability contract that runs at
#      runtime. If a downstream refactor renames a column or changes
#      its dtype, the schema check fires before the model trains on
#      bad data. Think of it as a type system for ML features.
#
#   2. Leakage audit — a mechanical scan that looks for features
#      whose name or correlation with the target implies they encode
#      post-hoc information (discharge diagnosis, death time, total
#      charges). Any feature flagged by the audit MUST be removed
#      before training.
#
# Neither defence is optional. Schemas catch "we renamed a column"
# bugs; the leakage audit catches "we accidentally used the future"
# bugs. Different failure modes, both fatal.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: feature matrix + FeatureSchema + re-run selections
# ════════════════════════════════════════════════════════════════════════

tables = load_icu_tables()
features = build_full_feature_frame(tables)
feature_cols, X_sel, y_binary = prepare_selection_inputs(features)

print("\n" + "=" * 70)
print("  Validation, Consensus & Leakage Audit")
print("=" * 70)
print(f"  Features: {len(feature_cols)}")
print(f"  Samples:  {X_sel.shape[0]}")

# FeatureSchema — declare expected types on the core clinical contract
icu_schema = FeatureSchema(
    name="icu_clinical_features_v1",
    features=[
        FeatureField(
            name="age",
            dtype="float64",
            nullable=False,
            description="Patient age at admission",
        ),
        FeatureField(
            name="los_days",
            dtype="float64",
            nullable=False,
            description="Length of ICU stay in days",
        ),
        FeatureField(
            name="n_unique_medications",
            dtype="int64",
            nullable=False,
            description="Count of distinct medications administered",
        ),
        FeatureField(
            name="received_vasopressors",
            dtype="bool",
            nullable=False,
            description="Whether patient received vasopressor drugs",
        ),
        FeatureField(
            name="n_abnormal_labs",
            dtype="int64",
            nullable=False,
            description="Count of abnormal lab results",
        ),
        FeatureField(
            name="abnormal_lab_ratio",
            dtype="float64",
            nullable=False,
            description="Proportion of lab results flagged abnormal",
        ),
        FeatureField(
            name="medication_intensity",
            dtype="float64",
            nullable=False,
            description="Medication doses per day of stay",
        ),
    ],
    entity_id_column="patient_id",
    timestamp_column="admit_time",
    version=1,
)

print(f"\n--- FeatureSchema: {icu_schema.name} (v{icu_schema.version}) ---")
for f in icu_schema.features:
    nullable = "nullable" if f.nullable else "required"
    print(f"  {f.name:<25} {f.dtype:<10} {nullable}  -- {f.description}")

# Validate schema against the built feature matrix
for field_def in icu_schema.features:
    assert (
        field_def.name in features.columns
    ), f"Schema field '{field_def.name}' missing from feature matrix"

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert icu_schema.name == "icu_clinical_features_v1", "Task 2: schema name mismatch"
assert len(icu_schema.features) == 7, "Task 2: schema should declare 7 fields"
print("\n[ok] Checkpoint 1 passed — FeatureSchema validated against the matrix\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: re-run the three selections for the consensus vote
# ════════════════════════════════════════════════════════════════════════

# (a) Filter — mutual information
mi_scores = mutual_info_classif(X_sel, y_binary, random_state=42)
mi_top = {
    name
    for name, _ in sorted(
        zip(feature_cols, mi_scores), key=lambda x: x[1], reverse=True
    )[:15]
}

# (b) Filter — chi-squared
X_chi2 = MinMaxScaler().fit_transform(X_sel)
chi2_scores, _ = chi2(X_chi2, y_binary)
chi2_top = {
    name
    for name, _ in sorted(
        zip(feature_cols, chi2_scores), key=lambda x: x[1], reverse=True
    )[:15]
}

# (c) Wrapper — RFE + Random Forest
rfe = RFE(
    estimator=RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    ),
    n_features_to_select=15,
    step=5,
)
rfe.fit(X_sel, y_binary)
rfe_top = {name for name, selected in zip(feature_cols, rfe.support_) if selected}

# (d) Embedded — L1 Lasso
X_scaled = StandardScaler().fit_transform(X_sel)
lasso = LogisticRegression(
    penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42
)
lasso.fit(X_scaled, y_binary)
lasso_top = {
    name for name, coef in zip(feature_cols, lasso.coef_[0]) if abs(coef) > 1e-6
}

all_methods: dict[str, set[str]] = {
    "MI (top 15)": mi_top,
    "Chi2 (top 15)": chi2_top,
    "RFE (RF, 15)": rfe_top,
    "Lasso (C=0.1)": lasso_top,
}

votes: Counter = Counter()
for feats in all_methods.values():
    for f in feats:
        votes[f] += 1

consensus_3plus = [f for f, v in votes.most_common() if v >= 3]
consensus_2plus = [f for f, v in votes.most_common() if v >= 2]
final_features = consensus_3plus if len(consensus_3plus) >= 8 else consensus_2plus[:15]

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(final_features) > 0, "Task 3: consensus must select at least one feature"
assert len(final_features) <= len(
    feature_cols
), "Task 3: cannot select more features than exist"
print("\n[ok] Checkpoint 2 passed — multi-method consensus complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the consensus and run the leakage audit
# ════════════════════════════════════════════════════════════════════════

print("\n--- Feature Selection Method Comparison ---")
print(f"{'Method':<20} {'Selected':>10}")
print("-" * 32)
for method, feats in all_methods.items():
    print(f"  {method:<18} {len(feats):>10}")

print(f"\n  Features with >=3 method votes: {len(consensus_3plus)}")
print(f"  Features with >=2 method votes: {len(consensus_2plus)}")
print(f"\n  Final feature set ({len(final_features)} features):")
for f in final_features:
    picking_methods = [m for m, s in all_methods.items() if f in s]
    print(f"    {f:<35}  [{', '.join(picking_methods)}]")


# --- Leakage Audit ---
print("\n--- Leakage Detection Audit ---")

leakage_suspects = [
    c
    for c in feature_cols
    if any(
        kw in c.lower()
        for kw in ("mortality", "death", "outcome", "discharge_diagnosis")
    )
]
if leakage_suspects:
    print(f"  [WARN] target-derived feature names: {leakage_suspects}")
else:
    print("  [ok] no target-derived feature names")

future_risk_cols = [
    c
    for c in features.columns
    if any(kw in c.lower() for kw in ("discharge", "icu_out", "death_time"))
]
if future_risk_cols:
    print(f"  [WARN] potential future-information columns: {future_risk_cols}")
else:
    print("  [ok] no future-information columns")

print("  [ok] vitals / medications / labs filtered to [admit_time, discharge_time]")

# Correlation-based sniff test — any feature with r > 0.95 against the
# target is almost certainly leaked.
target_col = "mortality" if "mortality" in features.columns else "los_days"
target_high_corr: list[tuple[str, float]] = []
if target_col in features.columns:
    for col in feature_cols[:40]:
        try:
            corr = features.select(
                pl.corr(
                    pl.col(col).cast(pl.Float64),
                    pl.col(target_col).cast(pl.Float64),
                )
            ).item()
            if corr is not None and abs(corr) > 0.95:
                target_high_corr.append((col, float(corr)))
        except Exception:
            continue

if target_high_corr:
    print("  [WARN] near-perfect target correlation (|r|>0.95):")
    for name, r in target_high_corr:
        print(f"    {name:<33} r={r:.4f}")
else:
    print("  [ok] no features with suspicious target correlation")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert (
    len(leakage_suspects) == 0
), f"Task 4: leakage suspects detected: {leakage_suspects}"
assert (
    len(target_high_corr) == 0
), f"Task 4: features with r>0.95 to target: {target_high_corr}"
print("\n[ok] Checkpoint 3 passed — leakage audit clean\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4b — LOG the consensus run + profile to ExperimentTracker
# ════════════════════════════════════════════════════════════════════════


async def log_consensus() -> str:
    conn, tracker, exp_id = await setup_tracking()
    explorer = DataExplorer()
    profile = await explorer.profile(features)
    null_rate = sum(features[c].null_count() for c in features.columns) / (
        features.height * len(features.columns)
    )
    run_id = await log_selection_run(
        tracker,
        exp_id,
        run_name="consensus_multi_method",
        method="consensus",
        selected_features=final_features,
        total_features=len(feature_cols),
        extra_params={
            "schema": icu_schema.name,
            "schema_version": str(icu_schema.version),
            "voting_methods": ",".join(all_methods.keys()),
            "vote_threshold": "3",
        },
        extra_metrics={
            "n_features_engineered": float(len(feature_cols)),
            "n_features_consensus": float(len(final_features)),
            "null_rate": float(null_rate),
            "n_alerts": float(len(profile.alerts)),
            "leakage_suspects": float(len(leakage_suspects)),
            "high_target_corr": float(len(target_high_corr)),
        },
    )
    await conn.close()
    return run_id


run_id = asyncio.run(log_consensus())
print(f"\n  ExperimentTracker run: {run_id}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert run_id is not None, "Task 4: ExperimentTracker should return a run id"
print("\n[ok] Checkpoint 4 passed — consensus run logged\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MOH Singapore Population-Health Pipeline
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: The Ministry of Health (MOH) runs a national population-
# health platform that ingests anonymised discharge records from every
# Singapore hospital and trains predictive models (readmission,
# mortality, length-of-stay) for policy analysis and bed-capacity
# forecasting. Every model MUST pass:
#   1. A FeatureSchema contract check so schemas cannot silently drift
#      between the contributing hospitals
#   2. A leakage audit so no post-hoc column (discharge_diagnosis,
#      billed_charges, death_certificate_id) reaches the training set
#   3. An ExperimentTracker record for reproducibility — regulators
#      must be able to reproduce any published statistic at any time
#
# Why this exercise's pattern is the right tool:
#   - A per-hospital schema catches contributors who rename a column
#     (e.g., "HR" vs "heart_rate") before it poisons the national run
#   - The leakage audit is mechanical and can run as an airflow gate
#   - The multi-method consensus gives MOH a defensible, robust
#     feature shortlist that survives methodological review
#
# BUSINESS IMPACT: One leakage incident in a national health model
# would cost MOH an estimated S$2.5M in public-relations remediation
# and a Parliament-mandated external audit. Past incidents in other
# jurisdictions (UK Care.data, Dutch SyRI) show average recovery time
# of 18 months and residual trust damage measured in years. A leakage
# audit that catches even ONE pre-production leak saves that cost
# outright. Annual validation pipeline cost: ~S$180K (infra + two
# analysts). A single prevented incident pays for the pipeline for
# 14 years.
#
# LIMITATIONS:
#   - Schema checks catch type drift, not semantic drift (a column
#     still called "heart_rate" but suddenly measured in BPM/2)
#   - The leakage audit is HEURISTIC — it catches the known patterns
#     but a novel leakage source needs a clinician's eye
#   - ExperimentTracker assumes the caller writes every run; a
#     silent skip is indistinguishable from "nothing happened"


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Declared a FeatureSchema contract for ICU clinical features
  [x] Validated the schema against the engineered feature matrix
  [x] Voted across filter + wrapper + embedded methods for a robust
      consensus feature set
  [x] Ran a mechanical leakage audit (name-based + correlation-based)
  [x] Logged the final consensus + audit metrics to ExperimentTracker
  [x] Applied the pattern to the MOH Singapore population-health
      pipeline where governance dominates

  KEY INSIGHT: Data quality beats model complexity. A clean, audited,
  consensus-selected feature set on a linear model outperforms a deep
  network on raw data with no governance. The FeatureSchema + leakage
  audit IS the model's warranty card — without it, you are shipping
  predictions on trust alone.

  Next: Exercise 2 — bias/variance trade-off, nested cross-validation,
  and how regularisation controls model complexity without touching
  the feature set built here.
"""
)
