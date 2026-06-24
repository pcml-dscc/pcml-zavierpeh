# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.1: Clinical Feature Engineering with
#                         Point-in-Time Correctness
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Join five ICU tables with temporal correctness (no future leakage)
#   - Aggregate irregular time-series vitals into per-admission statistics
#   - Flag clinically meaningful medication and lab patterns
#   - Engineer interaction features that encode domain knowledge
#   - Apply to early-warning scoring at Singapore General Hospital
#
# PREREQUISITES: MLFP02 complete (polars group-by, joins, temporal filters)
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — why point-in-time correctness matters
#   2. Build — load tables, aggregate vitals, meds, labs
#   3. Train — no training; build the full feature matrix
#   4. Visualise — preview engineered columns + interaction distributions
#   5. Apply — Singapore General Hospital early-warning scoring
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared.mlfp03.ex_1 import (
    build_full_feature_frame,
    load_icu_tables,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Point-in-Time Correctness Matters
# ════════════════════════════════════════════════════════════════════════
# A feature built from "all vitals this patient ever had" leaks the
# future. If prediction time is the moment of ICU admission, the
# patient's discharge-day vitals do not exist yet — using them inflates
# validation accuracy and fails catastrophically in production.
#
# The fix is a temporal filter: every feature only uses data recorded
# BETWEEN admit_time and discharge_time for THAT admission.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: load tables and construct per-admission features
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Clinical Feature Engineering — ICU Multi-Table")
print("=" * 70)

# TODO: Call the shared helper to load all five ICU tables as a dict.
# Hint: load_icu_tables() — returns dict with keys patients, admissions,
# vitals, medications, labs.
tables = ____

for name, df in tables.items():
    print(f"  {name}: {df.shape}")

# TODO: Build the full feature matrix using the shared helper.
# Hint: build_full_feature_frame(tables) applies the point-in-time filter
# for vitals/meds/labs and computes all interaction features.
features = ____

print(f"\nFeature matrix: {features.shape}")
print(f"Total columns: {len(features.columns)}")


# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert features.height > 0, "Task 2: feature matrix must not be empty"
assert "shock_index" in features.columns, "Task 2: shock_index (interaction) missing"
assert "abnormal_lab_ratio" in features.columns, "Task 2: lab ratio missing"
assert features["abnormal_lab_ratio"].null_count() == 0, "Task 2: null in lab ratio"
print("\n[ok] Checkpoint 1 passed — feature matrix built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (quality audit of the feature matrix)
# ════════════════════════════════════════════════════════════════════════

print("\n--- Feature Matrix Quality Audit ---")

# TODO: Compute the global null rate across the whole feature matrix.
# Hint: sum features[c].null_count() for every column c, then divide by
# (features.height * len(features.columns)).
null_rate = ____

numeric_cols = [
    c
    for c in features.columns
    if features[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
]
bool_cols = [c for c in features.columns if features[c].dtype == pl.Boolean]

print(f"  Rows:        {features.height}")
print(f"  Columns:     {len(features.columns)}")
print(f"  Numeric:     {len(numeric_cols)}")
print(f"  Boolean:     {len(bool_cols)}")
print(f"  Global null rate: {null_rate:.4f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert null_rate < 0.20, f"Task 3: null rate {null_rate:.4f} exceeds 20%"
print("\n[ok] Checkpoint 2 passed — feature quality audit OK\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the interaction features
# ════════════════════════════════════════════════════════════════════════

print("\n--- Clinical Interaction Feature Distributions ---")
interaction_cols = [
    c
    for c in ("shock_index", "map_mean", "fever_tachycardia", "treatment_burden_score")
    if c in features.columns
]

# TODO: Build a one-row summary with mean + std for each interaction column.
# Hint: features.select(*[pl.col(c).mean().alias(f"{c}_mean") for c in ...],
#                       *[pl.col(c).std().alias(f"{c}_std") for c in ...])
summary = ____

for c in interaction_cols:
    mean = summary[f"{c}_mean"].item() or 0.0
    std = summary[f"{c}_std"].item() or 0.0
    print(f"  {c:<25} mean={mean:>10.3f}  std={std:>10.3f}")

if "shock_index" in features.columns:
    print("\n  shock_index buckets (clinical interpretation):")
    buckets = (
        features.select(
            pl.when(pl.col("shock_index") < 0.7)
            .then(pl.lit("normal (<0.7)"))
            .when(pl.col("shock_index") < 0.9)
            .then(pl.lit("concerning (0.7-0.9)"))
            .when(pl.col("shock_index") < 1.2)
            .then(pl.lit("emergency (0.9-1.2)"))
            .otherwise(pl.lit("critical (>=1.2)"))
            .alias("bucket")
        )
        .group_by("bucket")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    for row in buckets.iter_rows(named=True):
        bar = "#" * min(40, int(row["n"] / 5))
        print(f"    {row['bucket']:<24} {row['n']:>5}  {bar}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert "shock_index" in features.columns, "Task 4: shock_index missing"
assert features["shock_index"].null_count() == 0, "Task 4: null in shock_index"
print("\n[ok] Checkpoint 3 passed — interaction distributions plausible\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore General Hospital Early-Warning Scoring
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Singapore General Hospital runs ~2,500 ICU admissions per
# year. The clinical informatics team wants an early-warning score that
# flags deteriorating patients 4-6 hours before a code-blue event.
#
# Why the engineered features solve it:
#   - shock_index (HR / SBP) is a validated early-warning marker
#   - abnormal_lab_ratio integrates a "drifting off baseline" signal
#   - medication_intensity proxies clinician concern
#
# BUSINESS IMPACT: SGH estimates each prevented code-blue saves ~S$18K.
# Catching 5 extra events/month = S$1.08M/year in cost avoidance.
# Feature-engineering cost: ~S$350K/year. 3x ROI in year one.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Joined five ICU tables with point-in-time correctness
  [x] Aggregated irregular vital-sign time series per admission
  [x] Engineered clinical interaction features
  [x] Quantified business impact at Singapore General Hospital

  Next: 02_filter_selection.py — rank features with mutual information
  and chi-squared.
"""
)
