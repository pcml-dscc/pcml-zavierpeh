# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 5: Window Functions and Trends
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Compute rolling averages and YoY changes using window functions
#   - Use .over() to partition window calculations by group
#   - Identify trends and seasonality in time-series data
#   - Understand when lazy evaluation (.lazy() / .collect()) helps
#   - Rank and classify towns by growth trajectory
#
# PREREQUISITES: Complete Exercise 4 first (joins, multi-table data).
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  Build monthly price series per town using group_by + sort
#   2.  Rolling averages with rolling_mean() and over()
#   3.  Multiple rolling windows — short-term vs long-term signals
#   4.  Year-over-year price change with shift()
#   5.  National-level YoY and market cycle detection
#   6.  Cumulative statistics — running totals and expanding windows
#   7.  Lazy frames — defer execution until collect()
#   8.  Rank towns by recent price growth
#   9.  Trend leaders, followers, and laggards
#   10. Momentum analysis — accelerating vs decelerating markets
#
# DATASET: Singapore HDB resale flat transactions (time-series focus)
#   Source: Housing & Development Board (data.gov.sg)
#   Rows: ~500,000 transactions spanning multiple years
#   Key columns: month, town, resale_price, floor_area_sqm
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

# Parse dates and add derived columns used throughout
hdb = hdb.with_columns(
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
)

print("=" * 60)
print("  MLFP01 Exercise 5: Window Functions and Trends")
print("=" * 60)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.height:,} rows, {hdb.width} columns)")
print(
    f"  Date range: {hdb['transaction_date'].min()} to {hdb['transaction_date'].max()}"
)
print(f"  You're ready to start!\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build monthly price series per town
# ══════════════════════════════════════════════════════════════════════
# Before applying window functions we need a monthly aggregate.
# Each row represents one (town, month) combination.
# Sorting by town then date is critical — rolling windows depend on order.

monthly_prices = (
    hdb.group_by("town", "transaction_date")
    .agg(
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        pl.col("resale_price").median().alias("median_resale_price"),
        pl.col("resale_price").mean().alias("mean_resale_price"),
        pl.col("floor_area_sqm").median().alias("median_area_sqm"),
        pl.len().alias("transaction_count"),
    )
    .sort("town", "transaction_date")
)

print(f"=== Monthly Price Series ===")
print(f"Shape: {monthly_prices.shape}  (one row per town per month)")
print(f"Towns: {monthly_prices['town'].n_unique()}")
print(
    f"Date range: {monthly_prices['transaction_date'].min()} to "
    f"{monthly_prices['transaction_date'].max()}"
)

# Show a sample town
print(f"\nBishan sample (first 6 months):")
print(monthly_prices.filter(pl.col("town") == "BISHAN").head(6))
# INTERPRETATION: This monthly aggregate is the foundation for all trend
# analysis. median_price_sqm normalises for flat size mix — if one month
# has more large flats sold, raw resale_price would be inflated even
# if the underlying market hasn't moved.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert monthly_prices.height > 0, "monthly_prices should have rows"
n_unique = monthly_prices.select(["town", "transaction_date"]).unique().height
assert monthly_prices.height == n_unique, "One row per (town, date) pair"
bishan = monthly_prices.filter(pl.col("town") == "BISHAN")
assert (
    bishan["transaction_date"][0] <= bishan["transaction_date"][-1]
), "Data should be sorted by date"
print("\n✓ Checkpoint 1 passed — monthly price series built correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Rolling averages with rolling_mean() and over()
# ══════════════════════════════════════════════════════════════════════
# A window function computes a value for each row using a sliding window
# of surrounding rows — without collapsing the DataFrame like group_by.
#
# rolling_mean(window_size=12) computes a 12-row moving average.
# .over("town") partitions by town so the window never crosses boundaries.

monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm")
    .rolling_mean(window_size=12)
    .over("town")
    .alias("rolling_12m_price_sqm"),
)

# A shorter 3-month window for a more reactive signal
monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm")
    .rolling_mean(window_size=3)
    .over("town")
    .alias("rolling_3m_price_sqm"),
)

print(f"=== Rolling Averages — Bishan (last 18 months) ===")
bishan = monthly_prices.filter(pl.col("town") == "BISHAN").tail(18)
print(
    bishan.select(
        "transaction_date",
        "median_price_sqm",
        "rolling_3m_price_sqm",
        "rolling_12m_price_sqm",
    )
)
# INTERPRETATION: rolling_3m reacts quickly to price changes — useful for
# detecting early market turns. rolling_12m is smoother — it shows the
# underlying trend. When rolling_3m crosses above rolling_12m, it often
# signals an accelerating market (a "golden cross" in technical analysis).

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert "rolling_12m_price_sqm" in monthly_prices.columns
assert "rolling_3m_price_sqm" in monthly_prices.columns
bishan_all = monthly_prices.filter(pl.col("town") == "BISHAN").sort("transaction_date")
assert (
    bishan_all["rolling_12m_price_sqm"][0] is None
), "First row of rolling_12m should be null"
print("\n✓ Checkpoint 2 passed — rolling averages computed per town\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Multiple rolling windows — short vs long signals
# ══════════════════════════════════════════════════════════════════════

# --- 3a: Rolling standard deviation — volatility measure ---
monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm")
    .rolling_std(window_size=12)
    .over("town")
    .alias("rolling_12m_std"),
)

# --- 3b: Rolling min and max — price envelope ---
monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm")
    .rolling_min(window_size=12)
    .over("town")
    .alias("rolling_12m_min"),
    pl.col("median_price_sqm")
    .rolling_max(window_size=12)
    .over("town")
    .alias("rolling_12m_max"),
)

# --- 3c: Price position within the rolling range ---
# Where does the current price sit between its 12-month min and max?
# 0% = at the bottom, 100% = at the top of its recent range
monthly_prices = monthly_prices.with_columns(
    (
        (pl.col("median_price_sqm") - pl.col("rolling_12m_min"))
        / (pl.col("rolling_12m_max") - pl.col("rolling_12m_min"))
        * 100
    ).alias("price_position_pct"),
)

# --- 3d: Golden cross / death cross signals ---
# When the short-term average crosses above the long-term, it may signal
# a market upturn ("golden cross"). Below = potential downturn ("death cross").
monthly_prices = monthly_prices.with_columns(
    pl.when(
        (pl.col("rolling_3m_price_sqm") > pl.col("rolling_12m_price_sqm"))
        & pl.col("rolling_3m_price_sqm").is_not_null()
        & pl.col("rolling_12m_price_sqm").is_not_null()
    )
    .then(pl.lit("bullish"))
    .when(
        (pl.col("rolling_3m_price_sqm") < pl.col("rolling_12m_price_sqm"))
        & pl.col("rolling_3m_price_sqm").is_not_null()
        & pl.col("rolling_12m_price_sqm").is_not_null()
    )
    .then(pl.lit("bearish"))
    .otherwise(pl.lit("neutral"))
    .alias("market_signal"),
)

# Summary of signals across the latest month
latest_date = monthly_prices["transaction_date"].max()
latest_signals = monthly_prices.filter(pl.col("transaction_date") == latest_date)

signal_counts = (
    latest_signals.group_by("market_signal")
    .agg(pl.len().alias("towns"))
    .sort("towns", descending=True)
)
print(f"=== Market Signals ({latest_date}) ===")
print(signal_counts)

# Show bullish towns
bullish_towns = latest_signals.filter(pl.col("market_signal") == "bullish")
if bullish_towns.height > 0:
    print(f"\nBullish towns (3m > 12m average):")
    print(
        bullish_towns.select(
            "town",
            "median_price_sqm",
            "rolling_3m_price_sqm",
            "rolling_12m_price_sqm",
            "price_position_pct",
        )
        .sort("price_position_pct", descending=True)
        .head(10)
    )

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert "rolling_12m_std" in monthly_prices.columns
assert "price_position_pct" in monthly_prices.columns
assert "market_signal" in monthly_prices.columns
signal_values = set(monthly_prices["market_signal"].unique().to_list())
assert (
    "bullish" in signal_values or "bearish" in signal_values
), "Should have at least one bullish or bearish signal"
print("\n✓ Checkpoint 3 passed — multiple rolling windows and signals computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Year-over-year price change with shift()
# ══════════════════════════════════════════════════════════════════════
# shift(n) moves every value n positions forward, filling first n with null.
# Combined with .over("town"), each town shifts independently.
# YoY = (current - 12_months_ago) / 12_months_ago * 100

monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm").shift(12).over("town").alias("price_sqm_12m_ago"),
)

monthly_prices = monthly_prices.with_columns(
    (
        (pl.col("median_price_sqm") - pl.col("price_sqm_12m_ago"))
        / pl.col("price_sqm_12m_ago")
        * 100
    ).alias("yoy_price_change_pct"),
)

# --- 4a: Month-over-month change ---
monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm").shift(1).over("town").alias("price_sqm_1m_ago"),
)
monthly_prices = monthly_prices.with_columns(
    (
        (pl.col("median_price_sqm") - pl.col("price_sqm_1m_ago"))
        / pl.col("price_sqm_1m_ago")
        * 100
    ).alias("mom_price_change_pct"),
)

# --- 4b: Show YoY and MoM for a sample town ---
print(f"=== Bishan Price Changes (last 24 months) ===")
print(
    monthly_prices.filter(pl.col("town") == "BISHAN")
    .tail(24)
    .select(
        "transaction_date",
        "median_price_sqm",
        "yoy_price_change_pct",
        "mom_price_change_pct",
    )
)
# INTERPRETATION: yoy_price_change_pct > 0 means prices are higher than
# the same month a year ago. MoM changes are noisier but detect turning
# points faster. When YoY is positive but MoM turns negative, the market
# may be decelerating — still up annually but losing momentum monthly.

# --- 4c: Largest single-month YoY gains and declines ---
valid_yoy = monthly_prices.filter(pl.col("yoy_price_change_pct").is_not_null())

top_gains = valid_yoy.sort("yoy_price_change_pct", descending=True).head(10)
top_declines = valid_yoy.sort("yoy_price_change_pct").head(10)

print(f"\n=== Top 10 YoY Gains (any town, any month) ===")
print(
    top_gains.select(
        "town", "transaction_date", "median_price_sqm", "yoy_price_change_pct"
    )
)

print(f"\n=== Top 10 YoY Declines ===")
print(
    top_declines.select(
        "town", "transaction_date", "median_price_sqm", "yoy_price_change_pct"
    )
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert "yoy_price_change_pct" in monthly_prices.columns
assert "mom_price_change_pct" in monthly_prices.columns
bishan_sorted = monthly_prices.filter(pl.col("town") == "BISHAN").sort(
    "transaction_date"
)
assert (
    bishan_sorted["yoy_price_change_pct"][0] is None
), "First yoy per town should be null"
print("\n✓ Checkpoint 4 passed — YoY and MoM changes computed correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: National-level YoY and market cycle detection
# ══════════════════════════════════════════════════════════════════════

# --- 5a: National monthly median ---
national_monthly = (
    hdb.group_by("transaction_date")
    .agg(
        pl.col("price_per_sqm").median().alias("national_median_sqm"),
        pl.col("resale_price").median().alias("national_median_price"),
        pl.len().alias("national_volume"),
    )
    .sort("transaction_date")
    .with_columns(
        pl.col("national_median_sqm").shift(12).alias("national_sqm_12m_ago"),
        pl.col("national_median_sqm")
        .rolling_mean(window_size=12)
        .alias("national_12m_avg"),
        pl.col("national_volume").rolling_mean(window_size=12).alias("volume_12m_avg"),
    )
    .with_columns(
        (
            (pl.col("national_median_sqm") - pl.col("national_sqm_12m_ago"))
            / pl.col("national_sqm_12m_ago")
            * 100
        ).alias("national_yoy_pct"),
    )
)

# --- 5b: Market cycle classification ---
national_monthly = national_monthly.with_columns(
    pl.when(pl.col("national_yoy_pct") > 5)
    .then(pl.lit("boom"))
    .when(pl.col("national_yoy_pct") > 0)
    .then(pl.lit("growth"))
    .when(pl.col("national_yoy_pct") > -5)
    .then(pl.lit("correction"))
    .when(pl.col("national_yoy_pct").is_not_null())
    .then(pl.lit("decline"))
    .otherwise(pl.lit("insufficient_data"))
    .alias("market_cycle"),
)

print(f"=== National Market Cycle ===")
cycle_counts = (
    national_monthly.filter(pl.col("market_cycle") != "insufficient_data")
    .group_by("market_cycle")
    .agg(
        pl.len().alias("months"),
        pl.col("national_yoy_pct").mean().alias("avg_yoy"),
    )
    .sort("avg_yoy", descending=True)
)
print(cycle_counts)

# --- 5c: Top national YoY months ---
print(f"\n=== National YoY — Top 10 Months ===")
print(
    national_monthly.drop_nulls("national_yoy_pct")
    .sort("national_yoy_pct", descending=True)
    .select(
        "transaction_date", "national_median_sqm", "national_yoy_pct", "market_cycle"
    )
    .head(10)
)

# --- 5d: Volume trend ---
print(f"\n=== Volume Trend (last 12 months) ===")
print(
    national_monthly.tail(12).select(
        "transaction_date", "national_volume", "volume_12m_avg"
    )
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert "national_yoy_pct" in national_monthly.columns
assert "market_cycle" in national_monthly.columns
assert national_monthly.height > 0
print("\n✓ Checkpoint 5 passed — national YoY and market cycles computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Cumulative statistics — running totals and expanding windows
# ══════════════════════════════════════════════════════════════════════
# Cumulative functions compute a running total from the first row to the
# current row. Unlike rolling windows (fixed size), cumulative windows
# grow with each row.

# --- 6a: Cumulative transaction count per town ---
monthly_prices = monthly_prices.with_columns(
    pl.col("transaction_count").cum_sum().over("town").alias("cum_transactions"),
)

# --- 6b: Cumulative mean price (expanding mean) ---
# The expanding mean is the average of all values from the start to now.
monthly_prices = monthly_prices.with_columns(
    (pl.col("median_price_sqm").cum_sum().over("town") / pl.col("median_price_sqm").cum_count().over("town")).alias("expanding_mean_psm"),
)

# --- 6c: Cumulative max and min ---
monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm").cum_max().over("town").alias("all_time_high_psm"),
    pl.col("median_price_sqm").cum_min().over("town").alias("all_time_low_psm"),
)

# --- 6d: Distance from all-time high ---
# How far is the current price from its all-time high?
# Negative = below ATH; 0 = at ATH
monthly_prices = monthly_prices.with_columns(
    (
        (pl.col("median_price_sqm") - pl.col("all_time_high_psm"))
        / pl.col("all_time_high_psm")
        * 100
    ).alias("pct_from_ath"),
)

print(f"=== Cumulative Stats — Bishan (last 12 months) ===")
print(
    monthly_prices.filter(pl.col("town") == "BISHAN")
    .tail(12)
    .select(
        "transaction_date",
        "median_price_sqm",
        "expanding_mean_psm",
        "all_time_high_psm",
        "pct_from_ath",
    )
)
# INTERPRETATION: pct_from_ath tells you how far the current price is from
# its historical peak. If it's 0%, the market is at an all-time high.
# If it's -10%, the market has corrected 10% from its peak. Markets at
# ATH are often seen as either "strong" (momentum) or "overheated" (caution).

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert "cum_transactions" in monthly_prices.columns
assert "expanding_mean_psm" in monthly_prices.columns
assert "all_time_high_psm" in monthly_prices.columns
assert "pct_from_ath" in monthly_prices.columns
# ATH should be >= current price
bishan_latest = monthly_prices.filter(pl.col("town") == "BISHAN").tail(1)
assert (
    bishan_latest["all_time_high_psm"][0] >= bishan_latest["median_price_sqm"][0]
), "ATH should be >= current price"
print("\n✓ Checkpoint 6 passed — cumulative statistics computed correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Lazy frames — defer execution until collect()
# ══════════════════════════════════════════════════════════════════════
# .lazy() converts a DataFrame into a LazyFrame. Nothing executes
# immediately — Polars builds a query plan instead. .collect() triggers
# execution with optimisations:
#   - Predicate pushdown: filters applied as early as possible
#   - Projection pushdown: unused columns dropped before reading
#   - Query fusion: adjacent operations combined

# --- 7a: Basic lazy pipeline ---
recent_yoy = (
    monthly_prices.lazy()
    .filter(pl.col("transaction_date") >= pl.date(2021, 1, 1))
    .drop_nulls("yoy_price_change_pct")
    .group_by("town")
    .agg(
        pl.col("yoy_price_change_pct").mean().alias("mean_yoy_pct"),
        pl.col("yoy_price_change_pct").std().alias("std_yoy_pct"),
        pl.col("yoy_price_change_pct").max().alias("peak_yoy_pct"),
        pl.col("yoy_price_change_pct").min().alias("trough_yoy_pct"),
        pl.col("yoy_price_change_pct").median().alias("median_yoy_pct"),
        pl.len().alias("months_of_data"),
    )
    .sort("mean_yoy_pct", descending=True)
    .collect()
)

print(f"=== Town YoY Growth Rankings (2021-present) ===")
print(f"Towns analysed: {recent_yoy.height}")
print(recent_yoy.head(10))

# --- 7b: Explain the query plan ---
# Use .explain() to see what Polars will do before executing
plan = (
    monthly_prices.lazy()
    .filter(pl.col("transaction_date") >= pl.date(2021, 1, 1))
    .select("town", "yoy_price_change_pct")
    .group_by("town")
    .agg(pl.col("yoy_price_change_pct").mean())
    .explain()
)
print(f"\n=== Query Plan ===")
print(plan[:500] + "..." if len(plan) > 500 else plan)

# --- 7c: scan_csv for large files (concept) ---
# For very large files, scan_csv() reads lazily without loading everything.
# We demonstrate the concept but use our existing data.
lazy_result = (
    monthly_prices.lazy()
    .filter(pl.col("town") == "BISHAN")
    .select("transaction_date", "median_price_sqm", "yoy_price_change_pct")
    .tail(12)
    .collect()
)
print(f"\n=== Lazy Evaluation Result ===")
print(lazy_result)
# INTERPRETATION: For this dataset (500k rows), lazy evaluation saves
# marginal time. For datasets with millions of rows, the savings are
# significant because Polars avoids materialising intermediate results.

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert recent_yoy.height > 0, "recent_yoy should have rows"
assert "mean_yoy_pct" in recent_yoy.columns
assert (
    recent_yoy["mean_yoy_pct"][0] >= recent_yoy["mean_yoy_pct"][-1]
), "Should be sorted descending"
print("\n✓ Checkpoint 7 passed — lazy evaluation pipeline collected correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Rank towns by recent price growth
# ══════════════════════════════════════════════════════════════════════

# --- 8a: Growth rank ---
recent_yoy = recent_yoy.with_columns(
    pl.col("mean_yoy_pct").rank(method="ordinal", descending=True).alias("growth_rank"),
)

# --- 8b: Volatility rank ---
recent_yoy = recent_yoy.with_columns(
    pl.col("std_yoy_pct")
    .rank(method="ordinal", descending=False)
    .alias("stability_rank"),
)
# Low std = more stable = better stability rank

# --- 8c: Composite score ---
# A simple composite: high growth + high stability = good investment
recent_yoy = recent_yoy.with_columns(
    (pl.col("growth_rank") + pl.col("stability_rank")).alias("composite_rank"),
)

print(f"=== Town Rankings ===")
print(
    recent_yoy.sort("composite_rank")
    .select(
        "town",
        "mean_yoy_pct",
        "std_yoy_pct",
        "growth_rank",
        "stability_rank",
        "composite_rank",
    )
    .head(15)
)
# INTERPRETATION: The composite rank combines growth (how much prices rose)
# with stability (how consistently they rose). A town ranked #1 in growth
# but #20 in stability had volatile gains — riskier for investment.
# A town ranked #5 in both had solid, consistent appreciation.

# --- 8d: Risk-return classification ---
mean_growth_all = recent_yoy["mean_yoy_pct"].mean()
mean_std_all = recent_yoy["std_yoy_pct"].mean()

recent_yoy = recent_yoy.with_columns(
    pl.when(
        (pl.col("mean_yoy_pct") > mean_growth_all)
        & (pl.col("std_yoy_pct") <= mean_std_all)
    )
    .then(pl.lit("high_return_low_risk"))
    .when(
        (pl.col("mean_yoy_pct") > mean_growth_all)
        & (pl.col("std_yoy_pct") > mean_std_all)
    )
    .then(pl.lit("high_return_high_risk"))
    .when(
        (pl.col("mean_yoy_pct") <= mean_growth_all)
        & (pl.col("std_yoy_pct") <= mean_std_all)
    )
    .then(pl.lit("low_return_low_risk"))
    .otherwise(pl.lit("low_return_high_risk"))
    .alias("risk_return_quadrant"),
)

quadrant_counts = (
    recent_yoy.group_by("risk_return_quadrant")
    .agg(pl.len().alias("towns"), pl.col("town").first().alias("example"))
    .sort("risk_return_quadrant")
)
print(f"\n=== Risk-Return Quadrants ===")
print(quadrant_counts)

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert "growth_rank" in recent_yoy.columns
assert "composite_rank" in recent_yoy.columns
assert "risk_return_quadrant" in recent_yoy.columns
print("\n✓ Checkpoint 8 passed — rankings and risk-return computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Trend leaders, followers, and laggards
# ══════════════════════════════════════════════════════════════════════

mean_growth = recent_yoy["mean_yoy_pct"].mean()
std_growth = recent_yoy["mean_yoy_pct"].std()

recent_yoy = recent_yoy.with_columns(
    pl.when(pl.col("mean_yoy_pct") > mean_growth + std_growth)
    .then(pl.lit("leader"))
    .when(pl.col("mean_yoy_pct") < mean_growth - std_growth)
    .then(pl.lit("laggard"))
    .otherwise(pl.lit("follower"))
    .alias("trend_category"),
)

print(f"=== Trend Classification ===")
print(f"Mean YoY (all towns): {mean_growth:.2f}%")
print(f"Std dev: {std_growth:.2f}%")
print(f"Leader threshold: > {mean_growth + std_growth:.2f}%")
print(f"Laggard threshold: < {mean_growth - std_growth:.2f}%")

category_counts = (
    recent_yoy.group_by("trend_category")
    .agg(
        pl.len().alias("count"),
        pl.col("mean_yoy_pct").mean().alias("avg_yoy"),
        pl.col("mean_yoy_pct").std().alias("std_yoy"),
    )
    .sort("avg_yoy", descending=True)
)
print(category_counts)

print(f"\n=== Trend Leaders (fastest-growing towns) ===")
leaders = recent_yoy.filter(pl.col("trend_category") == "leader")
print(
    leaders.select("town", "mean_yoy_pct", "peak_yoy_pct", "growth_rank").sort(
        "mean_yoy_pct", descending=True
    )
)

print(f"\n=== Trend Laggards (slowest-growing towns) ===")
laggards = recent_yoy.filter(pl.col("trend_category") == "laggard")
print(
    laggards.select("town", "mean_yoy_pct", "trough_yoy_pct", "growth_rank").sort(
        "mean_yoy_pct"
    )
)

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert "trend_category" in recent_yoy.columns
categories = set(recent_yoy["trend_category"].unique().to_list())
assert categories == {"leader", "follower", "laggard"}, f"Got: {categories}"
print("\n✓ Checkpoint 9 passed — trend classification complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Momentum analysis — accelerating vs decelerating
# ══════════════════════════════════════════════════════════════════════
# Momentum measures whether growth is speeding up or slowing down.
# A town can have positive YoY (prices still rising) but negative
# momentum (the rate of increase is slowing).

# --- 10a: YoY change of YoY change (second derivative) ---
monthly_prices = monthly_prices.with_columns(
    pl.col("yoy_price_change_pct").shift(12).over("town").alias("yoy_12m_ago"),
)

monthly_prices = monthly_prices.with_columns(
    (pl.col("yoy_price_change_pct") - pl.col("yoy_12m_ago")).alias("yoy_acceleration"),
)

# --- 10b: Latest momentum by town ---
latest_momentum = (
    monthly_prices.filter(
        pl.col("transaction_date") == monthly_prices["transaction_date"].max()
    )
    .filter(pl.col("yoy_acceleration").is_not_null())
    .select("town", "yoy_price_change_pct", "yoy_acceleration", "market_signal")
    .sort("yoy_acceleration", descending=True)
)

# --- 10c: Classify momentum ---
latest_momentum = latest_momentum.with_columns(
    pl.when((pl.col("yoy_price_change_pct") > 0) & (pl.col("yoy_acceleration") > 0))
    .then(pl.lit("accelerating_growth"))
    .when((pl.col("yoy_price_change_pct") > 0) & (pl.col("yoy_acceleration") <= 0))
    .then(pl.lit("decelerating_growth"))
    .when((pl.col("yoy_price_change_pct") <= 0) & (pl.col("yoy_acceleration") > 0))
    .then(pl.lit("recovering"))
    .otherwise(pl.lit("declining"))
    .alias("momentum"),
)

print(f"=== Current Market Momentum ===")
momentum_summary = (
    latest_momentum.group_by("momentum")
    .agg(pl.len().alias("towns"))
    .sort("towns", descending=True)
)
print(momentum_summary)

print(f"\n=== Accelerating Growth Towns ===")
accel = latest_momentum.filter(pl.col("momentum") == "accelerating_growth")
if accel.height > 0:
    print(accel.head(10))
else:
    print("  No towns currently in accelerating growth phase")

print(f"\n=== Decelerating Growth Towns ===")
decel = latest_momentum.filter(pl.col("momentum") == "decelerating_growth")
if decel.height > 0:
    print(decel.head(10))
else:
    print("  No towns currently in decelerating growth phase")

# --- 10d: Final comprehensive summary ---
print(f"\n{'═' * 70}")
print(f"  SINGAPORE HDB MARKET TREND SUMMARY")
print(f"{'═' * 70}")
print(f"  Analysis period: 2021 to present")
print(f"  Towns analysed:  {recent_yoy.height}")
print(f"  National avg YoY: {mean_growth:.2f}%")
print(f"")
print(f"  Trend Classification:")
for row in category_counts.iter_rows(named=True):
    print(
        f"    {row['trend_category']:<12} {row['count']:>3} towns  avg YoY={row['avg_yoy']:.1f}%"
    )
print(f"")
print(f"  Current Momentum:")
for row in momentum_summary.iter_rows(named=True):
    print(f"    {row['momentum']:<25} {row['towns']:>3} towns")
print(f"{'═' * 70}")

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert "yoy_acceleration" in monthly_prices.columns
assert "momentum" in latest_momentum.columns
momentum_values = set(latest_momentum["momentum"].unique().to_list())
assert len(momentum_values) > 0, "Should have at least one momentum category"
print("\n✓ Checkpoint 10 passed — momentum analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print(
    """
  ✓ Monthly aggregation: building a time-series base table
  ✓ rolling_mean(): smoothing noisy price data with a sliding window
  ✓ .over("town"): partitioning window functions by group
  ✓ Multiple rolling functions: std, min, max for volatility and range
  ✓ Market signals: golden/death cross from short vs long averages
  ✓ shift(12): comparing each month to the same month a year ago
  ✓ YoY and MoM calculation: annual and monthly price changes
  ✓ Cumulative functions: cum_sum, cum_mean, cum_max, cum_min
  ✓ Distance from ATH: tracking how far markets have fallen from peaks
  ✓ .lazy() / .collect(): deferring execution for query optimization
  ✓ rank(): adding ranking columns without reordering
  ✓ Risk-return quadrants: classifying growth vs stability
  ✓ Trend classification: leader / follower / laggard using std bands
  ✓ Momentum analysis: accelerating vs decelerating market phases

  NEXT: In Exercise 6, you'll turn these numbers into interactive
  charts using ModelVisualizer — the Kailash engine wrapping Plotly.
  You'll build histograms, scatter plots, bar charts, heatmaps, and
  line charts to make the patterns you've found visible.
"""
)
