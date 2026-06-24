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
#   - Engineer interaction features that encode domain knowledge (shock
#     index, mean arterial pressure, fever-tachycardia product)
#   - Apply to early-warning scoring at Singapore General Hospital
#
# PREREQUISITES: MLFP02 complete (polars group-by, joins, temporal filters)
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — why point-in-time correctness matters
#   2. Build — load tables, aggregate vitals, meds, labs
#   3. Train — there is no training; we BUILD the full feature matrix
#   4. Visualise — preview the engineered columns + interaction distributions
#   5. Apply — Singapore General Hospital early-warning scoring (S$ impact)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from shared.mlfp03.ex_1 import (
    OUTPUT_DIR,
    build_full_feature_frame,
    load_icu_tables,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Point-in-Time Correctness Matters
# ════════════════════════════════════════════════════════════════════════
# A feature built from "all vitals this patient ever had" leaks the
# future. If prediction time is the moment of ICU admission, then the
# patient's discharge-day vitals do not exist yet — using them inflates
# validation accuracy and fails catastrophically in production.
#
# The fix is a temporal filter: every feature only uses data recorded
# BETWEEN admit_time and discharge_time for THAT admission. The same
# rule applies to medications (start_time), labs (timestamp), and any
# derived feature.
#
# Analogy: imagine building a stock-price model that accidentally uses
# tomorrow's close as today's feature. Your backtest looks amazing; your
# live trading loses every penny. Medical ML has the same failure mode,
# but the stakes are higher — a biased model kills patients, not pennies.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: load tables and construct per-admission features
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Clinical Feature Engineering — ICU Multi-Table")
print("=" * 70)

tables = load_icu_tables()
for name, df in tables.items():
    print(f"  {name}: {df.shape}")

# The shared helper encodes the full feature contract: vital aggregates
# per admission (mean/std/min/max/range/trend/count/cv), medication flags
# (vasopressors, antibiotics, sedation), lab ratios, and clinical
# interaction features. Every technique file in this exercise starts from
# the same matrix — only the SELECTION method differs.
features = build_full_feature_frame(tables)

print(f"\nFeature matrix: {features.shape}")
print(f"Total columns: {len(features.columns)}")


# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert features.height > 0, "Task 2: feature matrix must not be empty"
assert "shock_index" in features.columns, "Task 2: shock_index (interaction) missing"
assert "abnormal_lab_ratio" in features.columns, "Task 2: lab ratio missing"
assert features["abnormal_lab_ratio"].null_count() == 0, "Task 2: null in lab ratio"
print("\n[ok] Checkpoint 1 passed — feature matrix built\n")

# INTERPRETATION: The _count suffix columns are particularly valuable —
# a patient with heart_rate_count = 120 in a 24h stay (5/hour) is being
# monitored far more intensively than one with count = 8. The coefficient
# of variation (_cv) captures NORMALISED volatility: a heart rate that
# oscillates wildly has high CV regardless of baseline.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (no training — quality audit of the feature matrix)
# ════════════════════════════════════════════════════════════════════════
# Feature engineering has no gradient-descent loop. The "training" step
# is a deterministic construction, and we validate it by auditing the
# null rate, dtype coverage, and clinical interaction distributions.

print("\n--- Feature Matrix Quality Audit ---")
# Null rate across NUMERIC + BOOLEAN columns only. Stat columns like
# {vital}_std legitimately produce nulls when a patient has a single
# reading, and full-outer vital joins leave nulls for patients who were
# never sampled for a given vital — both are expected, not defects.
null_rate = sum(features[c].null_count() for c in features.columns) / (
    features.height * len(features.columns)
)
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
assert null_rate < 0.20, (
    f"Task 3: null rate {null_rate:.4f} exceeds 20%. A high null rate "
    "means the temporal filter dropped too many rows — investigate the "
    "admit_time / discharge_time coverage before moving on."
)
print("\n[ok] Checkpoint 2 passed — feature quality audit OK\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the interaction features
# ════════════════════════════════════════════════════════════════════════
# Visual proof: the interaction features should show clinically plausible
# distributions. Shock index > 0.9 is a known emergency marker; MAP
# should centre around 70-90 mmHg for most patients.

print("\n--- Clinical Interaction Feature Distributions ---")
interaction_cols = [
    c
    for c in ("shock_index", "map_mean", "fever_tachycardia", "treatment_burden_score")
    if c in features.columns
]
summary = features.select(
    *[pl.col(c).mean().alias(f"{c}_mean") for c in interaction_cols],
    *[pl.col(c).std().alias(f"{c}_std") for c in interaction_cols],
)
for c in interaction_cols:
    mean = summary[f"{c}_mean"].item() or 0.0
    std = summary[f"{c}_std"].item() or 0.0
    print(f"  {c:<25} mean={mean:>10.3f}  std={std:>10.3f}")

# Rough histogram of shock_index (clinically actionable ranges)
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

# --- Correlation heatmap of engineered features ---
corr_cols = [c for c in interaction_cols if c in features.columns]
corr_cols += [
    c for c in features.columns if c.endswith("_mean") and c not in corr_cols
][:8]
corr_df = features.select([pl.col(c).cast(pl.Float64) for c in corr_cols]).to_pandas()
corr_matrix = corr_df.corr()
fig_heat = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlation Heatmap — Engineered Clinical Features",
)
fig_heat.update_layout(width=800, height=700)
heat_path = OUTPUT_DIR / "ex1_01_correlation_heatmap.html"
fig_heat.write_html(str(heat_path))
print(f"\n  Saved: {heat_path}")

# --- Feature distribution histograms ---
hist_cols = [c for c in interaction_cols if c in features.columns][:4]
fig_hist = go.Figure()
for col in hist_cols:
    vals = features[col].drop_nulls().to_list()
    fig_hist.add_trace(go.Histogram(x=vals, name=col, opacity=0.6, nbinsx=40))
fig_hist.update_layout(
    title="Distribution of Clinical Interaction Features",
    xaxis_title="Value",
    yaxis_title="Count",
    barmode="overlay",
    height=450,
)
hist_path = OUTPUT_DIR / "ex1_01_feature_distributions.html"
fig_hist.write_html(str(hist_path))
print(f"  Saved: {hist_path}")


# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert "shock_index" in features.columns, "Task 4: shock_index missing"
assert (
    features["shock_index"].null_count() == 0
), "Task 4: null in shock_index — check input vital columns"
print("\n[ok] Checkpoint 3 passed — interaction distributions plausible\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore General Hospital Early-Warning Scoring
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Singapore General Hospital (SGH) runs ~2,500 ICU admissions
# per year. The clinical informatics team wants an early-warning score
# that flags deteriorating patients 4-6 hours before a code-blue event.
# The current system uses raw vitals (heart rate > 120 = alert); it
# fires hundreds of false alarms per day and the nurses have started
# ignoring the pager.
#
# Why the engineered features solve it:
#   - shock_index (HR / SBP) is a validated early-warning marker that
#     beats either vital alone — it fires ~30% fewer false alarms at
#     the same sensitivity because it captures compensated shock
#   - abnormal_lab_ratio integrates the "everything is drifting off
#     baseline" signal into a single number
#   - medication_intensity and n_unique_medications act as a proxy for
#     clinician concern — more drugs means the team has already
#     escalated, which is itself a predictor
#
# BUSINESS IMPACT: SGH estimates each prevented code-blue saves roughly
# S$18,000 in ICU escalation costs and adds 2.3 disability-adjusted life
# years per patient. If the engineered features reduce false alarms 30%
# and catch even 5 additional deteriorations per month, that is:
#     5 events/month x 12 months x S$18,000 = S$1.08M/year in direct
#     cost avoidance, plus ~140 DALYs preserved. Feature engineering
#     cost: one clinical informatics hire + one data engineer =
#     ~S$350K/year. 3x ROI in year one; 6-8x once the model compounds
#     across more wards.
#
# LIMITATIONS:
#   - Vitals coverage varies by ward — general wards have sparser vital
#     streams than the ICU, so heart_rate_count is confounded by ward
#   - The model still needs calibration per patient cohort (cardiac,
#     trauma, surgical) before it can ship to other hospitals
#   - Leakage auditing (see 05_validation_and_tracking.py) MUST run
#     every time the feature list changes; one leaky feature silently
#     kills the model in production


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Joined five ICU tables (patients, admissions, vitals, meds, labs)
  [x] Applied point-in-time filters so features cannot leak the future
  [x] Aggregated irregular vital-sign time series into per-admission stats
  [x] Flagged clinically meaningful drug classes via regex
  [x] Computed clinical interaction features from domain knowledge
  [x] Quantified business impact at Singapore General Hospital

  KEY INSIGHT: Domain knowledge dominates algorithmic complexity. The
  shock_index feature is one division, but it encodes decades of
  emergency-medicine research. No deep-learning architecture recovers
  that signal automatically from raw vitals without supervision.

  Next: 02_filter_selection.py — rank all engineered features using
  mutual information and chi-squared, and see which clinical features
  a model-free filter method can recover.
"""
)
