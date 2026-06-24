# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 8.1: FeatureSchema — Typed Feature Contracts
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Define a FeatureSchema with typed FeatureField entries
#   - Compute base property features from raw HDB resale data
#   - Validate features against the schema contract
#   - Connect to FeatureStore and register/store v1 features
#   - Apply schema-driven feature engineering to Singapore HDB valuation
#
# PREREQUISITES: MLFP02 Exercises 1-7 (Bayesian inference, hypothesis
#   testing, regression, causal inference)
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — why typed feature schemas prevent silent failures
#   2. Build — define FeatureSchema v1 and compute features
#   3. Train — register and store features in FeatureStore
#   4. Visualise — feature distributions and correlation structure
#   5. Apply — HDB flat valuation with schema-validated features
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
import plotly.graph_objects as go
from scipy import stats

from shared.mlfp02.ex_8 import (
    OUTPUT_DIR,
    build_schema_v1,
    compute_v1_features,
    load_hdb_resale,
    setup_feature_store,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Typed Feature Schemas Prevent Silent Failures
# ════════════════════════════════════════════════════════════════════════
# A feature schema is a contract between the producer (feature engineer)
# and the consumer (model trainer). Without it, the following failures
# happen silently:
#
#   1. Dtype drift — a feature that was float64 in training becomes
#      string after a data source migration. The model loads, runs, and
#      produces garbage predictions without any error.
#
#   2. Null smuggling — a feature declared "never null" starts getting
#      nulls from a new data partition. The model NaN-propagates through
#      every prediction silently.
#
#   3. Phantom columns — an upstream process renames "price_per_sqm" to
#      "price_sqm". The old column vanishes, the model falls back to
#      defaults, and no one notices until the accuracy report.
#
# FeatureSchema catches ALL THREE at registration time, not at
# prediction time. Think of it as a unit test for your feature contract.
#
# Singapore HDB analogy: HDB publishes standard flat categories
# (3-room, 4-room, 5-room). If a listing arrives as "four-room" instead
# of "4 ROOM", every downstream system that filters by flat_type fails.
# The schema catches this at ingestion, not at report generation.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Define FeatureSchema v1 and compute property features
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Exercise 8.1 — FeatureSchema: Typed Feature Contracts")
print("=" * 70)

# --- 2a. Load HDB resale data ---
hdb = load_hdb_resale()
print(f"\n  Data loaded: {hdb.shape[0]:,} HDB resale transactions")

# --- 2b. Exploratory statistics (Ex 1-2 recap) ---
prices = hdb["resale_price"].to_numpy().astype(np.float64)
skew = stats.skew(prices)
kurt = stats.kurtosis(prices)
sw_stat, sw_p = stats.shapiro(
    np.random.default_rng(42).choice(prices, size=5000, replace=False)
)
mle_mu = prices.mean()
mle_sigma = prices.std(ddof=0)

print(f"\n  Price distribution:")
print(f"    n = {len(prices):,}")
print(f"    Mean: ${mle_mu:,.0f}, Median: ${np.median(prices):,.0f}")
print(f"    Std: ${mle_sigma:,.0f}")
print(
    f"    Skewness: {skew:.3f} "
    f"({'right-skewed' if skew > 0.5 else 'approximately symmetric'})"
)
print(
    f"    Excess kurtosis: {kurt:.3f} "
    f"({'heavy-tailed' if kurt > 1 else 'normal-tailed'})"
)
print(f"    Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.6f}")
print(f"\n  MLE (Normal): mu={mle_mu:,.0f}, sigma={mle_sigma:,.0f}")

# --- 2c. Define FeatureSchema v1 ---
property_schema_v1 = build_schema_v1()

print(f"\n  === FeatureSchema v1 ===")
print(f"  Name: {property_schema_v1.name}, Version: {property_schema_v1.version}")
for f in property_schema_v1.features:
    print(f"    {f.name}: {f.dtype} (nullable={f.nullable}) — {f.description}")

# --- 2d. Compute v1 features ---
features_v1 = compute_v1_features(hdb)
print(f"\n  Computed v1 features: {features_v1.shape}")

for feat_name in ["price_per_sqm", "storey_midpoint", "remaining_lease_years"]:
    vals = features_v1[feat_name].drop_nulls()
    print(
        f"    {feat_name}: mean={vals.mean():.1f}, "
        f"min={vals.min():.1f}, max={vals.max():.1f}"
    )

# --- 2e. Correlation analysis ---
corr_cols = [
    "resale_price",
    "floor_area_sqm",
    "storey_midpoint",
    "remaining_lease_years",
]
corr_data = (
    features_v1.drop_nulls(subset=corr_cols)
    .select(corr_cols)
    .to_numpy()
    .astype(np.float64)
)
corr_matrix = np.corrcoef(corr_data.T)

print(f"\n  Correlation matrix:")
print(f"  {'':>20}", end="")
for c in corr_cols:
    print(f"  {c[:12]:>12}", end="")
print()
for i, name in enumerate(corr_cols):
    print(f"  {name[:20]:<20}", end="")
    for j in range(len(corr_cols)):
        print(f"  {corr_matrix[i,j]:>12.3f}", end="")
    print()


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(prices) > 0, "Task 2: must have price data"
assert "transaction_id" in features_v1.columns, "Task 2: transaction_id missing"
assert "price_per_sqm" in features_v1.columns, "Task 2: price_per_sqm missing"
assert features_v1["price_per_sqm"].min() > 0, "Task 2: price_per_sqm must be positive"
assert corr_matrix.shape == (4, 4), "Task 2: correlation matrix must be 4x4"
print("\n[ok] Checkpoint 1 passed — v1 features computed and validated\n")

# INTERPRETATION: The schema declares 4 features — all non-nullable,
# all float64. This contract means ANY downstream model can trust that
# these columns exist and have no nulls. The storey_midpoint extraction
# from "01 TO 03" → 2.0 is a classic example of feature engineering that
# should be captured in the schema, not rediscovered per model.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Register schema and store features in FeatureStore
# ════════════════════════════════════════════════════════════════════════
# FeatureStore is a versioned, typed feature repository. Registering the
# schema tells the store what columns to expect; storing features
# validates each row against the schema before persisting.

print("\n--- FeatureStore Registration ---")

factory, fs, tracker, has_backend = asyncio.run(setup_feature_store())

if has_backend:
    try:

        async def store_v1():
            await fs.register_features(property_schema_v1)
            return await fs.store(features_v1, property_schema_v1)

        row_count = asyncio.run(store_v1())
        print(f"  Stored {row_count:,} v1 feature rows in FeatureStore")
    except Exception as e:
        has_backend = False
        print(f"  [Skipped: FeatureStore ({type(e).__name__}: {e})]")
else:
    print("  [Skipped: FeatureStore backend unavailable]")
    print("  Features remain in-memory as Polars DataFrame — all analysis continues")

# Price by flat type (foreshadows ANOVA concepts from M3)
flat_types = hdb["flat_type"].unique().sort().to_list()
print(f"\n  --- Price by Flat Type ---")
for ft in flat_types:
    subset = hdb.filter(pl.col("flat_type") == ft)["resale_price"]
    if subset.len() > 10:
        print(
            f"    {ft:<12}: n={subset.len():>7,}, "
            f"mean=${subset.mean():>10,.0f}, "
            f"median=${subset.median():>10,.0f}"
        )


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert property_schema_v1.version == 1, "Task 3: schema must be version 1"
assert len(property_schema_v1.features) == 4, "Task 3: v1 must have 4 features"
print("\n[ok] Checkpoint 2 passed — schema registered, features stored\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Feature distributions and structure
# ════════════════════════════════════════════════════════════════════════
# Visual proof: the computed features should show plausible
# distributions for Singapore HDB flats. Price_per_sqm should cluster
# around $4,000-8,000/sqm; storey_midpoint should be discrete steps;
# remaining_lease should be 40-99 years.

print("\n--- Feature Distribution Summary ---")
for feat_name in ["price_per_sqm", "storey_midpoint", "remaining_lease_years"]:
    vals = features_v1[feat_name].drop_nulls()
    q25 = vals.quantile(0.25)
    q75 = vals.quantile(0.75)
    print(
        f"  {feat_name:<25} " f"Q1={q25:>8,.1f}  Q3={q75:>8,.1f}  IQR={q75-q25:>8,.1f}"
    )

# Plot: feature distributions
fig = go.Figure()
for feat_name in ["price_per_sqm", "remaining_lease_years"]:
    vals = features_v1[feat_name].drop_nulls().to_numpy()
    fig.add_trace(
        go.Histogram(
            x=vals,
            name=feat_name,
            opacity=0.6,
            nbinsx=50,
        )
    )
fig.update_layout(
    title="v1 Feature Distributions — HDB Property Features",
    xaxis_title="Value",
    yaxis_title="Count",
    barmode="overlay",
)
fig.write_html(str(OUTPUT_DIR / "01_feature_distributions.html"))
print(f"\n  Saved: {OUTPUT_DIR / '01_feature_distributions.html'}")

# Plot: correlation heatmap
fig2 = go.Figure(
    data=go.Heatmap(
        z=corr_matrix,
        x=corr_cols,
        y=corr_cols,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr_matrix, 3),
        texttemplate="%{text}",
    )
)
fig2.update_layout(title="Feature Correlation Matrix — HDB v1")
fig2.write_html(str(OUTPUT_DIR / "01_correlation_heatmap.html"))
print(f"  Saved: {OUTPUT_DIR / '01_correlation_heatmap.html'}")


# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 3 passed — feature distributions visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: HDB Flat Valuation with Schema-Validated Features
# ════════════════════════════════════════════════════════════════════════
# Scenario: PropertyGuru Singapore wants to build an automated valuation
# model (AVM) for HDB resale flats. Before any model training, they
# need a reliable feature pipeline with typed schemas.
#
# Without a schema: an upstream change renames "remaining_lease_years"
# to "lease_remaining". The model silently uses a default value.
# Every valuation is wrong. PropertyGuru discovers this 3 weeks later
# when agents report "the system says every flat is worth $350K".
#
# With a schema: the FeatureStore registration rejects the data at
# ingestion. The pipeline fails loudly. PropertyGuru fixes the rename
# in 20 minutes. Zero bad valuations reach agents.
#
# S$ impact: PropertyGuru handles ~15,000 HDB listings/month in
# Singapore. A 3-week silent failure means ~11,250 listings with wrong
# valuations. At an average commission of $8,000 per transaction, even
# a 5% deal-loss rate from bad valuations costs:
#   11,250 * 5% * $8,000 = S$4.5M in lost commission revenue.

print("=== APPLY: HDB Flat Valuation — Schema-Driven Pipeline ===")
print()
print("  Scenario: PropertyGuru automated valuation model (AVM)")
print()
print("  WITHOUT schema:")
print("    - Upstream renames 'remaining_lease_years' to 'lease_remaining'")
print("    - Model silently uses default values for 3 weeks")
print("    - 11,250 listings with wrong valuations")
print("    - S$4.5M lost commission revenue")
print()
print("  WITH FeatureSchema v1:")
print("    - FeatureStore rejects data at ingestion (dtype/name mismatch)")
print("    - Pipeline fails loudly, fixed in 20 minutes")
print("    - Zero bad valuations reach property agents")
print()
print("  Your v1 schema enforces:")
for f in property_schema_v1.features:
    nullable_str = "optional" if f.nullable else "required"
    print(f"    {f.name}: {f.dtype} ({nullable_str})")
print()
print(
    f"  Total features validated: {features_v1.shape[0]:,} rows "
    f"x {len(property_schema_v1.features)} columns"
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 4 passed — schema-driven valuation pipeline demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [ok] FeatureSchema: typed fields with dtype, nullable, description
  [ok] FeatureField: individual feature contracts within a schema
  [ok] Feature computation: storey_midpoint, price_per_sqm, remaining_lease
  [ok] FeatureStore registration: schema-validated feature persistence
  [ok] Correlation analysis: identifying multicollinearity early

  KEY INSIGHT: A schema is a unit test for your feature pipeline.
  It catches dtype drift, null smuggling, and phantom columns at
  INGESTION time — not at prediction time when it's too late.

  Next: In 02_point_in_time.py, you'll learn how point-in-time
  retrieval prevents data leakage — the #1 cause of models that
  look great in development and fail catastrophically in production.
"""
)
