# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 8.3: Rolling Features — Temporal Market Context
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Define FeatureSchema v2 with rolling market-context features
#   - Compute rolling statistics with Polars group_by_dynamic
#   - Understand rolling window warm-up periods and null handling
#   - Track schema evolution from v1 to v2 with versioned FeatureStore
#   - Apply rolling market features to Singapore town-level analytics
#
# PREREQUISITES: Exercise 8.1-8.2 (FeatureSchema v1, PIT retrieval)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — why rolling features capture market momentum
#   2. Build — define schema v2 and compute rolling town statistics
#   3. Train — register v2 and store in FeatureStore with versioning
#   4. Visualise — rolling price trends and transaction volumes by town
#   5. Apply — ERA Realty town-level investment advisory
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from shared.mlfp02.ex_8 import (
    OUTPUT_DIR,
    build_schema_v1,
    build_schema_v2,
    compute_v1_features,
    compute_v2_features,
    load_hdb_resale,
    setup_feature_store,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Rolling Features Capture Market Momentum
# ════════════════════════════════════════════════════════════════════════
# A single transaction price is a noisy signal. It depends on the
# specific flat's condition, the buyer's urgency, the agent's skill,
# and random timing. But the MEDIAN price in a town over the past 6
# months is a stable signal — it captures the local market's direction.
#
# Rolling features aggregate noisy individual observations into smooth
# market-level statistics:
#
#   - town_median_price: "What's the typical price in this town lately?"
#   - town_transaction_volume: "Is this a hot market or a cold one?"
#   - town_price_trend: "Are prices going up, down, or flat?"
#
# These three features transform a model from "predict based on this
# flat's characteristics" to "predict based on this flat in THIS market
# context". The same flat in a booming town sells for more than in a
# stagnant one.
#
# Polars group_by_dynamic is the engine: it buckets transactions into
# monthly windows per town, then rolling_mean/rolling_sum aggregates
# across a 6-month trailing window. The first 6 months per town have
# nulls — this is the warm-up period where the rolling window hasn't
# filled yet.
#
# Singapore context: HDB towns like Bishan, Tampines, and Woodlands
# have very different price trajectories. A 4-room flat in Bishan
# (mature estate, near MRT) appreciates differently from Woodlands
# (non-mature estate). Rolling features capture this divergence.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Define FeatureSchema v2 and compute rolling features
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Exercise 8.3 — Rolling Features: Temporal Market Context")
print("=" * 70)

# --- 2a. Load and compute v1 features (baseline) ---
hdb = load_hdb_resale()
features_v1 = compute_v1_features(hdb)

# --- 2b. Define FeatureSchema v2 ---
property_schema_v1 = build_schema_v1()
property_schema_v2 = build_schema_v2()

n_new = len(property_schema_v2.features) - len(property_schema_v1.features)
print(f"\n  === FeatureSchema v2 (+{n_new} market features) ===")
for f in property_schema_v2.features:
    tag = (
        " [NEW]"
        if f.name not in [ff.name for ff in property_schema_v1.features]
        else ""
    )
    print(f"    {f.name}: {f.dtype} (nullable={f.nullable}){tag}")

# --- 2c. Compute v2 features ---
features_v2 = compute_v2_features(hdb)
n_with_market = features_v2.filter(pl.col("town_median_price").is_not_null()).height
pct_with_market = n_with_market / features_v2.height

print(f"\n  Computed v2 features: {features_v2.shape}")
print(f"  Rows with market context: {n_with_market:,} ({pct_with_market:.1%})")
print(f"  (First 6 months per town have nulls — rolling window warm-up)")

# --- 2c-bis. FeatureEngineer — add temporal calendar features ---
# FeatureEngineer's temporal strategy extracts calendar components (month,
# day-of-week, hour) from any datetime column declared in the schema.
# Here we extract month and quarter manually (quarter is not part of the
# built-in temporal strategy); month/dow/hour would come from engineer.generate.
features_v2 = features_v2.with_columns(
    pl.col("transaction_date").dt.month().alias("transaction_date_month"),
    pl.col("transaction_date").dt.quarter().alias("transaction_date_quarter"),
)
print("\n  FeatureEngineer-equivalent temporal features: month, quarter")
print(f"  Columns after temporal extraction: {features_v2.shape[1]}")

# INTERPRETATION: Calendar features capture seasonality that rolling
# aggregates alone miss (e.g. Q1 vs Q4 buyer behaviour). The rolling
# market features above are domain-specific; temporal calendar features
# are mechanical and reused across ML pipelines.

# --- 2d. Show sample rolling values for a few towns ---
sample_towns = ["ANG MO KIO", "BISHAN", "TAMPINES", "WOODLANDS"]
for town in sample_towns:
    town_data = features_v2.filter(
        (pl.col("town") == town) & pl.col("town_median_price").is_not_null()
    )
    if town_data.height > 0:
        latest = town_data.sort("transaction_date").tail(1)
        median_p = latest["town_median_price"].item()
        volume = latest["town_transaction_volume"].item()
        trend = latest["town_price_trend"].item()
        print(
            f"    {town:<15}: median=${median_p:>10,.0f}  "
            f"volume={volume:>5}  trend={trend:>+.1f}%"
        )


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert (
    features_v2.height == features_v1.height
), "Task 2: v2 should have same row count as v1"
assert (
    "town_median_price" in features_v2.columns
), "Task 2: town_median_price must be computed"
assert (
    "town_transaction_volume" in features_v2.columns
), "Task 2: town_transaction_volume must be computed"
assert (
    "town_price_trend" in features_v2.columns
), "Task 2: town_price_trend must be computed"
assert (
    pct_with_market > 0.5
), f"Task 2: at least 50% of rows should have market context, got {pct_with_market:.1%}"
print("\n[ok] Checkpoint 1 passed — v2 features computed with rolling market context\n")

# INTERPRETATION: The v2 schema adds three nullable columns. They're
# nullable because the first 6 months per town can't compute a
# trailing window — that's the warm-up period, not a data quality bug.
# Downstream models must drop_nulls on these columns before training.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Register v2 schema and store versioned features
# ════════════════════════════════════════════════════════════════════════

print("--- FeatureStore Schema Evolution (v1 -> v2) ---")

factory, fs, tracker, has_backend = asyncio.run(setup_feature_store())

if has_backend:
    try:

        async def store_v2():
            await fs.register_features(property_schema_v2)
            return await fs.store(features_v2, property_schema_v2)

        row_count = asyncio.run(store_v2())
        print(f"  Stored {row_count:,} v2 feature rows")
        print(f"  Schema version: {property_schema_v2.version}")
    except Exception as e:
        has_backend = False
        print(f"  [Skipped: v2 store ({type(e).__name__}: {e})]")
else:
    print("  [Skipped: FeatureStore backend unavailable]")
    print("  v2 features remain in-memory as Polars DataFrame")

print(f"\n  Schema evolution:")
print(f"    v1: {len(property_schema_v1.features)} features (basic property)")
print(f"    v2: {len(property_schema_v2.features)} features (+{n_new} market context)")
print(f"    New fields: town_median_price, town_transaction_volume, town_price_trend")


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert property_schema_v2.version == 2, "Task 3: v2 schema must be version 2"
assert len(property_schema_v2.features) == 7, "Task 3: v2 must have 7 features"
print("\n[ok] Checkpoint 2 passed — v2 schema registered and stored\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Rolling price trends and volumes by town
# ════════════════════════════════════════════════════════════════════════

print("--- Town-Level Rolling Market Trends ---")

# Aggregate monthly medians per town for plotting
town_monthly = (
    features_v2.filter(pl.col("town_median_price").is_not_null())
    .group_by(["town", "transaction_date"])
    .agg(
        pl.col("town_median_price").first(),
        pl.col("town_transaction_volume").first(),
        pl.col("town_price_trend").first(),
    )
    .sort("transaction_date")
)

# Plot rolling median prices for selected towns
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=[
        "Rolling 6-Month Median Price by Town",
        "Rolling 6-Month Transaction Volume by Town",
    ],
    vertical_spacing=0.12,
)

for town in sample_towns:
    town_data = town_monthly.filter(pl.col("town") == town).sort("transaction_date")
    if town_data.height > 0:
        dates = town_data["transaction_date"].to_list()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=town_data["town_median_price"].to_list(),
                name=town,
                mode="lines",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=town_data["town_transaction_volume"].to_list(),
                name=town,
                showlegend=False,
                mode="lines",
            ),
            row=2,
            col=1,
        )

fig.update_layout(
    title="HDB Market Trends by Town (Rolling 6-Month Window)",
    height=700,
    width=1000,
)
fig.update_yaxes(title_text="Median Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Transaction Count", row=2, col=1)

fig.write_html(str(OUTPUT_DIR / "03_town_trends.html"))
print(f"\n  Saved: {OUTPUT_DIR / '03_town_trends.html'}")

# Price trend distribution (how many towns are appreciating vs declining)
trend_data = (
    features_v2.filter(pl.col("town_price_trend").is_not_null())
    .group_by("town")
    .agg(pl.col("town_price_trend").mean().alias("avg_trend"))
)

fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=trend_data.sort("avg_trend")["town"].to_list(),
        y=trend_data.sort("avg_trend")["avg_trend"].to_list(),
        marker_color=[
            "firebrick" if t < 0 else "seagreen"
            for t in trend_data.sort("avg_trend")["avg_trend"].to_list()
        ],
    )
)
fig2.update_layout(
    title="Average 6-Month Price Trend by Town (%)",
    xaxis_title="Town",
    yaxis_title="Average Price Trend (%)",
    xaxis_tickangle=-45,
)
fig2.write_html(str(OUTPUT_DIR / "03_town_price_trends.html"))
print(f"  Saved: {OUTPUT_DIR / '03_town_price_trends.html'}")


# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 3 passed — rolling market trends visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: ERA Realty Town-Level Investment Advisory
# ════════════════════════════════════════════════════════════════════════
# Scenario: ERA Realty advisors help HDB upgraders decide WHEN and
# WHERE to buy. With rolling market features, advisors can identify
# towns where prices are trending up (buy soon) vs towns where prices
# are stagnant (negotiate harder).
#
# Without rolling features: advisors rely on gut feel and last month's
# newspaper headlines. "Bishan is always expensive" — but IS it still
# appreciating, or has it plateaued?
#
# With rolling features: advisors see quantified 6-month trends per
# town. "Bishan median up 3.2% vs Tampines up 7.1% — Tampines is
# gaining ground. Buy Tampines now before the gap closes."
#
# ERA has ~6,800 agents in Singapore. If rolling-feature-based advice
# helps each agent close 1 additional deal per quarter (conservative),
# at an average commission of $5,000:
#   6,800 agents * 1 deal/quarter * $5,000 = S$34M additional revenue/year

print("=== APPLY: ERA Realty Town-Level Investment Advisory ===")

# Rank towns by recent trend
latest_trends = (
    features_v2.filter(pl.col("town_price_trend").is_not_null())
    .group_by("town")
    .agg(
        pl.col("town_price_trend").last().alias("latest_trend"),
        pl.col("town_median_price").last().alias("latest_median"),
        pl.col("town_transaction_volume").last().alias("latest_volume"),
    )
    .sort("latest_trend", descending=True)
)

print()
print("  Top 5 appreciating towns (buy-soon signal):")
for row in latest_trends.head(5).iter_rows(named=True):
    print(
        f"    {row['town']:<15}: trend={row['latest_trend']:>+6.1f}%  "
        f"median=${row['latest_median']:>10,.0f}  volume={row['latest_volume']:>5}"
    )

print()
print("  Bottom 5 towns (negotiate-harder signal):")
for row in latest_trends.tail(5).iter_rows(named=True):
    print(
        f"    {row['town']:<15}: trend={row['latest_trend']:>+6.1f}%  "
        f"median=${row['latest_median']:>10,.0f}  volume={row['latest_volume']:>5}"
    )

print()
print("  ERA advisory impact:")
print("    - 6,800 agents with quantified town-level trends")
print("    - 1 additional deal/quarter per agent (conservative)")
print("    - S$34M additional revenue per year")
print()
print("  Key insight: 'Bishan is expensive' is qualitative.")
print("  'Bishan median up 3.2% vs Tampines up 7.1% over 6 months'")
print("  is quantitative and actionable.")


# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert latest_trends.height > 0, "Task 5: must have town trend data"
print("\n[ok] Checkpoint 4 passed — town-level advisory demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [ok] FeatureSchema v2: extending v1 with rolling market-context fields
  [ok] group_by_dynamic: monthly bucketing of transactions per town
  [ok] Rolling statistics: trailing 6-month median, volume, trend
  [ok] Warm-up periods: why the first 6 months per town have nulls
  [ok] Schema versioning: v1 -> v2 evolution with backward compatibility

  KEY INSIGHT: Rolling features transform a model from "what is this
  flat worth?" to "what is this flat worth IN THIS MARKET?" The same
  flat in a booming town sells for more than in a stagnant one — and
  rolling features capture that difference quantitatively.

  Next: In 04_modeling_lineage.py, you'll build a full regression model
  on v2 features, apply hypothesis tests and Bayesian posteriors to the
  coefficients, and create a complete audit trail from data to model.
"""
)
