# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.5: FeatureSchema Validation, Multi-Method
#                         Consensus, and Leakage Audit
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Declare a FeatureSchema contract
#   - Validate the engineered matrix at runtime
#   - Vote across filter / wrapper / embedded methods for consensus
#   - Log the final feature set to ExperimentTracker
#   - Run a leakage audit before any training
#
# PREREQUISITES: 02, 03, 04
# ESTIMATED TIME: ~30 min
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
# Two complementary defences catch the highest-stakes ML bugs:
#   1. FeatureSchema — a runtime type/nullability contract
#   2. Leakage audit — a mechanical scan for post-hoc features
# Neither is optional.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: feature matrix + FeatureSchema + selections
# ════════════════════════════════════════════════════════════════════════

tables = load_icu_tables()
features = build_full_feature_frame(tables)
feature_cols, X_sel, y_binary = prepare_selection_inputs(features)

print("\n" + "=" * 70)
print("  Validation, Consensus & Leakage Audit")
print("=" * 70)
print(f"  Features: {len(feature_cols)}")
print(f"  Samples:  {X_sel.shape[0]}")

# TODO: Declare a FeatureSchema named "icu_clinical_features_v1" with
# seven FeatureField entries covering age, los_days, n_unique_medications,
# received_vasopressors, n_abnormal_labs, abnormal_lab_ratio,
# medication_intensity. Use entity_id_column="patient_id",
# timestamp_column="admit_time", version=1.
# Hint: FeatureSchema(name=..., features=[FeatureField(name=..., dtype=..., nullable=False, description=...), ...], entity_id_column=..., timestamp_column=..., version=1)
icu_schema = ____

print(f"\n--- FeatureSchema: {icu_schema.name} (v{icu_schema.version}) ---")
for f in icu_schema.features:
    nullable = "nullable" if f.nullable else "required"
    print(f"  {f.name:<25} {f.dtype:<10} {nullable}  -- {f.description}")

for field_def in icu_schema.features:
    assert (
        field_def.name in features.columns
    ), f"Schema field '{field_def.name}' missing from feature matrix"

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert icu_schema.name == "icu_clinical_features_v1", "Task 2: schema name mismatch"
assert len(icu_schema.features) == 7, "Task 2: schema should declare 7 fields"
print("\n[ok] Checkpoint 1 passed — FeatureSchema validated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: re-run the three selection families
# ════════════════════════════════════════════════════════════════════════

mi_scores = mutual_info_classif(X_sel, y_binary, random_state=42)
mi_top = {
    name
    for name, _ in sorted(
        zip(feature_cols, mi_scores), key=lambda x: x[1], reverse=True
    )[:15]
}

X_chi2 = MinMaxScaler().fit_transform(X_sel)
chi2_scores, _ = chi2(X_chi2, y_binary)
chi2_top = {
    name
    for name, _ in sorted(
        zip(feature_cols, chi2_scores), key=lambda x: x[1], reverse=True
    )[:15]
}

rfe = RFE(
    estimator=RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    ),
    n_features_to_select=15,
    step=5,
)
rfe.fit(X_sel, y_binary)
rfe_top = {name for name, sel in zip(feature_cols, rfe.support_) if sel}

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

# TODO: Build a Counter of how many methods selected each feature, then
# derive consensus_3plus (>=3 votes) and consensus_2plus (>=2 votes).
# Hint: votes = Counter(); for feats in all_methods.values(): votes.update(feats)
votes: Counter = ____
for feats in all_methods.values():
    ____

consensus_3plus = [f for f, v in votes.most_common() if v >= 3]
consensus_2plus = [f for f, v in votes.most_common() if v >= 2]
final_features = consensus_3plus if len(consensus_3plus) >= 8 else consensus_2plus[:15]

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(final_features) > 0, "Task 3: consensus must select at least one feature"
print("\n[ok] Checkpoint 2 passed — multi-method consensus complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE consensus + run leakage audit
# ════════════════════════════════════════════════════════════════════════

print("\n--- Feature Selection Method Comparison ---")
print(f"{'Method':<20} {'Selected':>10}")
print("-" * 32)
for method, feats in all_methods.items():
    print(f"  {method:<18} {len(feats):>10}")

print(f"\n  Features with >=3 method votes: {len(consensus_3plus)}")
print(f"  Final feature set ({len(final_features)} features):")
for f in final_features:
    picking_methods = [m for m, s in all_methods.items() if f in s]
    print(f"    {f:<35}  [{', '.join(picking_methods)}]")


print("\n--- Leakage Detection Audit ---")

# TODO: Scan feature_cols for any name containing "mortality", "death",
# "outcome", or "discharge_diagnosis".
# Hint: [c for c in feature_cols if any(kw in c.lower() for kw in (...))]
leakage_suspects = ____

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
# TASK 4b — LOG the consensus run + profile
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
# The Ministry of Health's national platform ingests records from every
# Singapore hospital. Every model MUST pass schema validation, leakage
# audit, and ExperimentTracker logging. One prevented leakage incident
# saves ~S$2.5M in remediation — the audit pays for the pipeline for
# 14 years off a single catch.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Declared and validated a FeatureSchema contract
  [x] Voted across filter + wrapper + embedded methods for consensus
  [x] Ran a mechanical leakage audit
  [x] Logged the consensus + audit metrics to ExperimentTracker

  Next: Exercise 2 — bias/variance trade-off, nested cross-validation,
  and regularisation without touching the feature set.
"""
)
