# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 3: Functions and Aggregation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Define functions that accept parameters and return values
#   - Use loops to process collections and iterate over DataFrame rows
#   - Aggregate data by groups using group_by() + agg()
#   - Apply multiple statistics (mean, median, std, quantile) in one call
#   - Write reusable helper functions for common data analysis tasks
#
# PREREQUISITES: Complete Exercise 2 first (filtering, with_columns, chaining).
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  Write basic functions — def, parameters, return, type hints
#   2.  Functions with default parameters and docstrings
#   3.  Lists, dictionaries, and iteration patterns
#   4.  group_by() + agg() — the core aggregation pattern
#   5.  Multiple aggregations in one call — the full statistics toolkit
#   6.  Derived columns from aggregated results
#   7.  Multi-key group_by — cross-tabulation analysis
#   8.  Time-series aggregation — annual and quarterly trends
#   9.  Reusable analysis functions — parameterised district reports
#   10. Comprehensive ranked district report with for loop iteration
#
# DATASET: Singapore HDB resale flat transactions
#   Source: Housing & Development Board (data.gov.sg)
#   Rows: ~500,000 transactions | Columns: month, town, flat_type,
#   floor_area_sqm, resale_price, and more
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

# Add derived columns used throughout the exercise
hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
    pl.col("month").str.slice(5, 2).cast(pl.Int32).alias("month_num"),
)

print("=" * 60)
print("  MLFP01 Exercise 3: Functions and Aggregation")
print("=" * 60)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.height:,} rows, {hdb.width} columns)")
print(f"  You're ready to start!\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Writing basic functions — def, parameters, return
# ══════════════════════════════════════════════════════════════════════
# A function packages reusable logic under a name. Without functions,
# you'd copy-paste the same code every time you need the same operation.
# def introduces the function, parameters go in parentheses,
# the body is indented, and return sends a value back to the caller.


def format_sgd(amount: float) -> str:
    """Format a number as Singapore dollars with thousands separator.

    Args:
        amount: The numeric value to format.

    Returns:
        A string like "S$485,000".
    """
    # The : tells Python this is a type hint — it documents what type
    # the parameter should be, but Python does not enforce it at runtime.
    # -> str says this function returns a string.
    return f"S${amount:,.0f}"


def format_pct(value: float, decimals: int = 1) -> str:
    """Format a decimal as a percentage string.

    Args:
        value: Decimal value (e.g., 0.85 for 85%).
        decimals: Number of decimal places (default 1).

    Returns:
        A string like "85.0%".
    """
    return f"{value * 100:.{decimals}f}%"


def price_range_label(price: float) -> str:
    """Classify a resale price into a human-readable tier.

    This uses if/elif/else — Python's branching mechanism. Each condition
    is checked in order; the first True branch executes and the rest are skipped.
    """
    if price < 350_000:
        return "Budget (<350k)"
    elif price < 500_000:
        return "Mid-range (350k-500k)"
    elif price < 700_000:
        return "Premium (500k-700k)"
    else:
        return "Luxury (700k+)"


# --- Test the functions before using them in a pipeline ---
print("=== Function Tests ===")
print(f"format_sgd(485000):      {format_sgd(485_000)}")
print(f"format_sgd(1200000):     {format_sgd(1_200_000)}")
print(f"format_pct(0.85):        {format_pct(0.85)}")
print(f"format_pct(0.85, 2):     {format_pct(0.85, 2)}")
print(f"price_range_label(300k): {price_range_label(300_000)}")
print(f"price_range_label(485k): {price_range_label(485_000)}")
print(f"price_range_label(720k): {price_range_label(720_000)}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert format_sgd(485_000) == "S$485,000", f"format_sgd returned: {format_sgd(485_000)}"
assert format_pct(0.5) == "50.0%", f"format_pct returned: {format_pct(0.5)}"
assert "Mid-range" in price_range_label(485_000), "485k should be Mid-range"
assert "Luxury" in price_range_label(720_000), "720k should be Luxury"
assert "Budget" in price_range_label(200_000), "200k should be Budget"
print("\n✓ Checkpoint 1 passed — basic functions work correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Functions with default parameters and validation
# ══════════════════════════════════════════════════════════════════════


def compute_iqr(series: pl.Series) -> float:
    """Compute the interquartile range (Q3 - Q1) of a Polars Series.

    IQR measures spread without being skewed by extreme outliers.
    A wide IQR means prices vary a lot within the district.
    """
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    if q75 is None or q25 is None:
        return 0.0
    return q75 - q25


def compute_cv(series: pl.Series) -> float:
    """Compute the coefficient of variation (std / mean * 100).

    CV normalises spread by the mean — useful for comparing variability
    between different towns or time periods regardless of absolute price level.
    """
    mean = series.mean()
    std = series.std()
    if mean is None or std is None or mean == 0:
        return 0.0
    return std / mean * 100


def describe_series(series: pl.Series, name: str = "Series") -> dict:
    """Compute a comprehensive statistical summary of a Polars Series.

    Returns a dict with all the key statistics — useful as a building
    block for more complex reports.
    """
    return {
        "name": name,
        "count": series.len(),
        "nulls": series.null_count(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "q25": series.quantile(0.25),
        "q75": series.quantile(0.75),
        "iqr": compute_iqr(series),
        "cv": compute_cv(series),
    }


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning a default if denominator is zero.

    Safe division prevents ZeroDivisionError — common when computing ratios
    from data that might have zero counts in some categories.
    """
    if denominator == 0:
        return default
    return numerator / denominator


# --- Test the functions ---
test_prices = pl.Series("prices", [300_000, 400_000, 500_000, 600_000, 700_000])
print("=== Statistical Functions ===")
print(f"IQR:  {format_sgd(compute_iqr(test_prices))}")
print(f"CV:   {compute_cv(test_prices):.1f}%")

desc = describe_series(hdb["resale_price"], "Resale Price")
print(f"\n=== describe_series() output ===")
for key, val in desc.items():
    if isinstance(val, float) and val > 1000:
        print(f"  {key:>10}: {format_sgd(val)}")
    else:
        print(f"  {key:>10}: {val}")

print(f"\nSafe divide: 100/0 = {safe_divide(100, 0)}")
print(f"Safe divide: 100/3 = {safe_divide(100, 3):.2f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (
    compute_iqr(test_prices) == 200_000.0
), f"IQR should be 200,000, got {compute_iqr(test_prices)}"
assert compute_cv(test_prices) > 0, "CV should be positive for non-constant series"
assert len(desc) >= 10, "describe_series should return at least 10 statistics"
assert desc["count"] == hdb.height, "count should match DataFrame height"
assert safe_divide(100, 0) == 0.0, "safe_divide by zero should return default"
print("\n✓ Checkpoint 2 passed — statistical helper functions work correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Lists, dictionaries, and iteration patterns
# ══════════════════════════════════════════════════════════════════════
# Before diving into group_by, let's solidify the iteration patterns
# you'll use to process aggregated results.

# --- 3a: Building a lookup dictionary from data ---
# Useful for mapping codes to human-readable names
flat_type_labels = {
    "1 ROOM": "Studio",
    "2 ROOM": "2-Room",
    "3 ROOM": "3-Room",
    "4 ROOM": "4-Room",
    "5 ROOM": "5-Room",
    "EXECUTIVE": "Executive",
    "MULTI-GENERATION": "Multi-Gen",
}

# --- 3b: Processing data with a for loop and accumulator ---
# Count transactions per flat type manually (to understand what group_by does)
flat_counts: dict[str, int] = {}
for ft in hdb["flat_type"].to_list():
    if ft in flat_counts:
        flat_counts[ft] += 1
    else:
        flat_counts[ft] = 1

print("=== Manual Flat Type Counts ===")
for ft, count in sorted(flat_counts.items(), key=lambda x: x[1], reverse=True):
    label = flat_type_labels.get(ft, ft)
    pct = count / hdb.height * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<12} {count:>8,} ({pct:5.1f}%) {bar}")

# --- 3c: Zip and enumerate in practice ---
# zip() pairs two sequences element-by-element
# enumerate() adds a counter to each item
towns_sample = ["BISHAN", "QUEENSTOWN", "JURONG EAST", "WOODLANDS", "TAMPINES"]
town_medians = [
    hdb.filter(pl.col("town") == t)["resale_price"].median() for t in towns_sample
]

print(f"\n=== Sample Town Medians (zip + enumerate) ===")
for i, (town, median) in enumerate(zip(towns_sample, town_medians), start=1):
    print(f"  {i}. {town:<18} {format_sgd(median)}")

# --- 3d: List comprehensions with conditions ---
# Find towns where the median price exceeds S$500k
expensive_towns = [
    town
    for town, median in zip(towns_sample, town_medians)
    if median is not None and median > 500_000
]
print(f"\nTowns with median > S$500k: {expensive_towns}")

# --- 3e: Nested data structures ---
# A list of dicts — the pattern you'll see with .iter_rows(named=True)
town_profiles = [
    {
        "town": town,
        "median": median,
        "label": "premium" if median and median > 500_000 else "standard",
    }
    for town, median in zip(towns_sample, town_medians)
]

print(f"\n=== Town Profiles (list of dicts) ===")
for profile in town_profiles:
    print(
        f"  {profile['town']:<18} {format_sgd(profile['median'])}  [{profile['label']}]"
    )

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    len(flat_counts) == hdb["flat_type"].n_unique()
), "Manual count should match unique flat types"
assert sum(flat_counts.values()) == hdb.height, "Sum of counts should equal total rows"
assert len(town_medians) == len(towns_sample), "Should have one median per town"
assert len(expensive_towns) > 0, "Should have at least one expensive town"
print("\n✓ Checkpoint 3 passed — lists, dicts, and iteration patterns working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: group_by() + agg() — the core aggregation pattern
# ══════════════════════════════════════════════════════════════════════
# group_by() splits the DataFrame into groups, one per unique value.
# agg() then computes a summary statistic for each group.
# The result has one row per group and one column per aggregation.
#
# This is the most important Polars pattern in data analysis:
# "For each [group], compute [statistics]."

# --- 4a: Basic group_by — one statistic per group ---
town_counts = (
    hdb.group_by("town")
    .agg(pl.len().alias("transaction_count"))
    .sort("transaction_count", descending=True)
)

print("=== Transaction Volume by Town ===")
print(f"Towns: {town_counts.height}")
print(town_counts.head(10))

# --- 4b: Flat type breakdown ---
flat_type_stats = (
    hdb.group_by("flat_type")
    .agg(
        pl.len().alias("count"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("floor_area_sqm").median().alias("median_area"),
    )
    .sort("median_price")
)

print(f"\n=== Flat Type Summary ===")
for row in flat_type_stats.iter_rows(named=True):
    label = flat_type_labels.get(row["flat_type"], row["flat_type"])
    print(
        f"  {label:<12} {row['count']:>8,}  "
        f"median={format_sgd(row['median_price'])}  "
        f"area={row['median_area']:.0f}sqm"
    )
# INTERPRETATION: There's a clear price ladder from 1-room to Executive.
# The median area also climbs — bigger flats cost more partly because
# they ARE more. The key question is: which flat types offer the best
# price per sqm? That answers "where is the value?"

# --- 4c: Compare group_by to manual counting ---
# Verify that group_by gives the same result as our manual count
manual_total = sum(flat_counts.values())
groupby_total = flat_type_stats["count"].sum()
print(f"\nManual total: {manual_total:,}  group_by total: {groupby_total:,}")
assert manual_total == groupby_total, "Totals should match"

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert town_counts.height > 0, "town_counts should have rows"
assert town_counts.height == hdb["town"].n_unique(), "One row per unique town"
assert "transaction_count" in town_counts.columns, "Missing transaction_count column"
assert flat_type_stats.height == hdb["flat_type"].n_unique(), "One row per flat type"
print("\n✓ Checkpoint 4 passed — basic group_by/agg working correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Multiple aggregations — the full statistics toolkit
# ══════════════════════════════════════════════════════════════════════

district_stats = (
    hdb.group_by("town")
    .agg(
        # Volume
        pl.len().alias("transaction_count"),
        # Price statistics — multiple measures give a richer picture
        pl.col("resale_price").mean().alias("mean_price"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("resale_price").std().alias("std_price"),
        pl.col("resale_price").min().alias("min_price"),
        pl.col("resale_price").max().alias("max_price"),
        pl.col("resale_price").quantile(0.25).alias("q25_price"),
        pl.col("resale_price").quantile(0.75).alias("q75_price"),
        # Price per sqm — normalised measure for cross-size comparison
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        pl.col("price_per_sqm").mean().alias("mean_price_sqm"),
        # Area statistics
        pl.col("floor_area_sqm").median().alias("median_area_sqm"),
        pl.col("floor_area_sqm").mean().alias("mean_area_sqm"),
    )
    .sort("median_price", descending=True)
)

print(f"\n=== District Statistics (Top 10) ===")
print(
    district_stats.select(
        "town",
        "transaction_count",
        "median_price",
        "std_price",
        "median_price_sqm",
    ).head(10)
)

# --- Mean vs Median comparison ---
# When mean >> median, the distribution is right-skewed (expensive outliers)
print(f"\n=== Mean vs Median Price (Top 5) ===")
for row in district_stats.head(5).iter_rows(named=True):
    mean = row["mean_price"]
    median = row["median_price"]
    skew_indicator = "→ right-skewed" if mean > median * 1.05 else "→ roughly symmetric"
    print(
        f"  {row['town']:<20} mean={format_sgd(mean)}  "
        f"median={format_sgd(median)}  {skew_indicator}"
    )
# INTERPRETATION: Sorting by median_price shows the hierarchy of Singapore's
# housing market. When mean > median, expensive outliers pull the average up.
# For reporting, use median for "typical" price and mean for "total value."

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert district_stats.height > 0, "district_stats should have rows"
assert "transaction_count" in district_stats.columns, "Missing transaction_count"
assert "median_price" in district_stats.columns, "Missing median_price"
top_median = district_stats["median_price"][0]
bottom_median = district_stats["median_price"][-1]
assert top_median >= bottom_median, "Results should be sorted descending"
print("\n✓ Checkpoint 5 passed — multiple aggregations producing correct statistics\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Derived columns from aggregated results
# ══════════════════════════════════════════════════════════════════════
# Once you have aggregated stats, you can add further derived columns.

district_stats = district_stats.with_columns(
    # IQR: spread of the middle 50% of prices
    (pl.col("q75_price") - pl.col("q25_price")).alias("iqr_price"),
    # Coefficient of variation: std / mean * 100
    (pl.col("std_price") / pl.col("mean_price") * 100).alias("cv_price_pct"),
    # Premium ratio: what fraction of the max price is the median?
    (pl.col("median_price") / pl.col("max_price")).alias("premium_ratio"),
    # Price range: max - min — absolute spread
    (pl.col("max_price") - pl.col("min_price")).alias("price_range"),
    # Size-adjusted price premium: how much more than average does this town cost?
    (
        pl.col("median_price_sqm") / pl.col("median_price_sqm").mean().over(pl.lit(1))
    ).alias("price_index"),
)

# --- Market position classification ---
overall_median_psm = hdb["price_per_sqm"].median()

district_stats = district_stats.with_columns(
    pl.when(pl.col("median_price_sqm") > overall_median_psm * 1.15)
    .then(pl.lit("premium"))
    .when(pl.col("median_price_sqm") < overall_median_psm * 0.85)
    .then(pl.lit("value"))
    .otherwise(pl.lit("mainstream"))
    .alias("market_position"),
)

print(f"\n=== District Stats with Derived Columns ===")
print(
    district_stats.select(
        "town",
        "transaction_count",
        "median_price",
        "iqr_price",
        "cv_price_pct",
        "market_position",
    ).head(10)
)

# Market position summary
position_summary = (
    district_stats.group_by("market_position")
    .agg(
        pl.len().alias("towns"),
        pl.col("median_price").mean().alias("avg_median_price"),
        pl.col("cv_price_pct").mean().alias("avg_cv"),
    )
    .sort("avg_median_price", descending=True)
)
print(f"\n=== Market Position Summary ===")
print(position_summary)
# INTERPRETATION: cv_price_pct is key. A district with CV=25% has prices
# clustered tightly. CV=40%+ means a wide mix — budget and premium side by side.
# Districts with high CV are harder to generalise about.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert "iqr_price" in district_stats.columns, "iqr_price should be added"
assert "cv_price_pct" in district_stats.columns, "cv_price_pct should be added"
assert "market_position" in district_stats.columns, "market_position should be added"
row = district_stats.row(0, named=True)
expected_iqr = row["q75_price"] - row["q25_price"]
assert abs(row["iqr_price"] - expected_iqr) < 1, "iqr should equal q75 - q25"
print("\n✓ Checkpoint 6 passed — derived columns computed from aggregates correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Multi-key group_by — cross-tabulation analysis
# ══════════════════════════════════════════════════════════════════════
# group_by() accepts multiple columns — creating a group for every
# unique combination of values.

# --- 7a: Town x Flat Type cross-tab ---
town_flat_stats = (
    hdb.group_by("town", "flat_type")
    .agg(
        pl.len().alias("count"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
    )
    .sort("town", "flat_type")
)

print(f"\n=== Town x Flat Type Statistics ===")
print(f"Groups: {town_flat_stats.height}")
print(f"\nAng Mo Kio breakdown:")
print(town_flat_stats.filter(pl.col("town") == "ANG MO KIO"))

# --- 7b: Which town has the best value for 4-room flats? ---
four_room_ranking = (
    town_flat_stats.filter(pl.col("flat_type") == "4 ROOM")
    .filter(pl.col("count") >= 100)  # Only towns with enough data
    .sort("median_price_sqm")
)

print(f"\n=== Best Value Towns for 4-Room Flats (by price/sqm) ===")
print(f"{'Town':<22} {'Count':>7} {'Median Price':>14} {'Price/sqm':>12}")
print(f"{'─' * 57}")
for row in four_room_ranking.head(10).iter_rows(named=True):
    print(
        f"{row['town']:<22} {row['count']:>7,} "
        f"{format_sgd(row['median_price']):>14} "
        f"{format_sgd(row['median_price_sqm']):>12}"
    )
# INTERPRETATION: Sorting by price/sqm ascending shows where you get the
# most space for your money. Peripheral towns (Woodlands, Jurong West)
# typically lead — but the trade-off is commute time and amenity access.

# --- 7c: Flat type mix by town ---
# What percentage of each town's transactions are 4-room vs 5-room?
town_totals = town_flat_stats.group_by("town").agg(pl.col("count").sum().alias("total"))

flat_mix = (
    town_flat_stats.join(town_totals, on="town")
    .with_columns((pl.col("count") / pl.col("total") * 100).alias("pct_of_town"))
    .filter(pl.col("flat_type").is_in(["4 ROOM", "5 ROOM"]))
    .sort("town", "flat_type")
)

print(f"\n=== 4/5-Room Mix for Sample Towns ===")
for town in ["BISHAN", "TAMPINES", "WOODLANDS"]:
    town_data = flat_mix.filter(pl.col("town") == town)
    print(f"\n  {town}:")
    for row in town_data.iter_rows(named=True):
        print(
            f"    {row['flat_type']}: {row['pct_of_town']:.1f}% ({row['count']:,} txns)"
        )

# ── Checkpoint 7 ─────────────────────────────────────────────────────
n_unique_pairs = hdb.select(["town", "flat_type"]).unique().height
assert (
    town_flat_stats.height == n_unique_pairs
), f"Expected {n_unique_pairs} groups, got {town_flat_stats.height}"
assert four_room_ranking.height > 0, "Should have 4-room rankings"
print("\n✓ Checkpoint 7 passed — multi-key group_by producing correct cross-tabs\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Time-series aggregation — annual and quarterly trends
# ══════════════════════════════════════════════════════════════════════

# --- 8a: Annual transaction volume and price trend ---
annual_stats = (
    hdb.group_by("year")
    .agg(
        pl.len().alias("transactions"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("resale_price").mean().alias("mean_price"),
        pl.col("price_per_sqm").median().alias("median_psm"),
    )
    .sort("year")
)

print(f"\n=== Annual HDB Market Summary ===")
prev_price = None
for row in annual_stats.iter_rows(named=True):
    price = row["median_price"]
    yoy = ""
    if prev_price is not None and prev_price > 0:
        change = (price - prev_price) / prev_price * 100
        arrow = "↑" if change > 0 else "↓"
        yoy = f"  {arrow} {change:+.1f}%"
    print(
        f"  {row['year']}: {format_sgd(price):>14}  "
        f"psm={format_sgd(row['median_psm']):>10}  "
        f"vol={row['transactions']:>7,}{yoy}"
    )
    prev_price = price

# --- 8b: Annual stats by town — which towns grew fastest? ---
annual_by_town = (
    hdb.group_by("year", "town")
    .agg(
        pl.col("resale_price").median().alias("median_price"),
        pl.len().alias("transactions"),
    )
    .sort("year", "town")
)

# Compare 2020 to latest year for each town
years_available = sorted(hdb["year"].unique().to_list())
early_year = 2020 if 2020 in years_available else years_available[0]
late_year = years_available[-1]

early = annual_by_town.filter(pl.col("year") == early_year)
late = annual_by_town.filter(pl.col("year") == late_year)

growth = (
    early.select("town", pl.col("median_price").alias("price_early"))
    .join(
        late.select("town", pl.col("median_price").alias("price_late")),
        on="town",
    )
    .with_columns(
        (
            (pl.col("price_late") - pl.col("price_early")) / pl.col("price_early") * 100
        ).alias("growth_pct")
    )
    .sort("growth_pct", descending=True)
)

print(f"\n=== Price Growth {early_year} → {late_year} (Top 10 Towns) ===")
for row in growth.head(10).iter_rows(named=True):
    print(
        f"  {row['town']:<22} "
        f"{format_sgd(row['price_early'])} → {format_sgd(row['price_late'])}  "
        f"({row['growth_pct']:+.1f}%)"
    )

# --- 8c: Quarterly seasonality ---
quarter_map = {
    1: "Q1",
    2: "Q1",
    3: "Q1",
    4: "Q2",
    5: "Q2",
    6: "Q2",
    7: "Q3",
    8: "Q3",
    9: "Q3",
    10: "Q4",
    11: "Q4",
    12: "Q4",
}

hdb_q = hdb.with_columns(
    pl.col("month_num").replace_strict(quarter_map, default="Q?").alias("quarter")
)

quarterly_volume = (
    hdb_q.group_by("quarter").agg(pl.len().alias("avg_transactions")).sort("quarter")
)
print(f"\n=== Average Quarterly Volume ===")
print(quarterly_volume)
# INTERPRETATION: If Q1 has fewer transactions, that's the Chinese New Year
# effect — the market slows during the holiday period.

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert annual_stats.height > 0, "Should have annual stats"
assert growth.height > 0, "Should have growth data"
assert "growth_pct" in growth.columns, "growth_pct should exist"
print("\n✓ Checkpoint 8 passed — time-series aggregation working correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Reusable analysis functions — parameterised reports
# ══════════════════════════════════════════════════════════════════════
# The real power of functions appears when you parameterise your analysis.
# Instead of copying code for each town, write one function and call it
# with different arguments.


def town_deep_dive(data: pl.DataFrame, town_name: str) -> dict:
    """Generate a comprehensive statistical profile for a single town.

    Args:
        data: Full HDB DataFrame.
        town_name: Town to analyse (must match data values exactly).

    Returns:
        Dictionary with all computed statistics.
    """
    town_data = data.filter(pl.col("town") == town_name)

    if town_data.height == 0:
        return {"town": town_name, "error": "No data found"}

    prices = town_data["resale_price"]
    psm = town_data["price_per_sqm"]

    # Flat type breakdown
    flat_breakdown = (
        town_data.group_by("flat_type")
        .agg(pl.len().alias("count"), pl.col("resale_price").median().alias("median"))
        .sort("count", descending=True)
    )

    return {
        "town": town_name,
        "transactions": town_data.height,
        "pct_of_market": town_data.height / data.height * 100,
        "median_price": prices.median(),
        "mean_price": prices.mean(),
        "std_price": prices.std(),
        "iqr_price": compute_iqr(prices),
        "cv_pct": compute_cv(prices),
        "median_psm": psm.median(),
        "min_price": prices.min(),
        "max_price": prices.max(),
        "top_flat_type": flat_breakdown["flat_type"][0],
        "top_flat_count": flat_breakdown["count"][0],
        "years_active": town_data["year"].n_unique(),
    }


def print_town_profile(profile: dict) -> None:
    """Pretty-print a town profile dictionary."""
    if "error" in profile:
        print(f"  {profile['town']}: {profile['error']}")
        return

    print(f"\n  ┌{'─' * 50}┐")
    print(f"  │ {profile['town']:^48} │")
    print(f"  ├{'─' * 50}┤")
    print(
        f"  │  Transactions:  {profile['transactions']:>8,} ({profile['pct_of_market']:.1f}% of market) │"
    )
    print(
        f"  │  Median Price:  {format_sgd(profile['median_price']):>12}                  │"
    )
    print(
        f"  │  Mean Price:    {format_sgd(profile['mean_price']):>12}                  │"
    )
    print(
        f"  │  Price Range:   {format_sgd(profile['min_price'])} - {format_sgd(profile['max_price'])}     │"
    )
    print(
        f"  │  IQR:           {format_sgd(profile['iqr_price']):>12}                  │"
    )
    print(f"  │  CV:            {profile['cv_pct']:>8.1f}%                        │")
    print(
        f"  │  Median PSM:    {format_sgd(profile['median_psm']):>12}                  │"
    )
    print(
        f"  │  Top Flat Type: {profile['top_flat_type']:<12} ({profile['top_flat_count']:,})      │"
    )
    print(f"  │  Years Active:  {profile['years_active']:>8}                        │")
    print(f"  └{'─' * 50}┘")


# --- Run deep dives for several towns ---
target_towns = ["BISHAN", "QUEENSTOWN", "TAMPINES", "WOODLANDS", "BUKIT MERAH"]
profiles = [town_deep_dive(hdb, town) for town in target_towns]

print("=== Town Deep Dives ===")
for profile in profiles:
    print_town_profile(profile)

# --- Compare towns side by side ---
print(f"\n=== Town Comparison ===")
print(f"{'Town':<18} {'Median':>12} {'PSM':>10} {'CV%':>6} {'Volume':>8}")
print(f"{'─' * 56}")
for p in sorted(profiles, key=lambda x: x.get("median_psm", 0), reverse=True):
    if "error" not in p:
        print(
            f"{p['town']:<18} {format_sgd(p['median_price']):>12} "
            f"{format_sgd(p['median_psm']):>10} "
            f"{p['cv_pct']:>5.1f}% "
            f"{p['transactions']:>8,}"
        )

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert len(profiles) == len(target_towns), "Should have one profile per town"
assert all("town" in p for p in profiles), "Each profile should have a town key"
bishan_profile = next(p for p in profiles if p["town"] == "BISHAN")
assert bishan_profile["transactions"] > 0, "Bishan should have transactions"
assert bishan_profile["median_price"] > 0, "Bishan median should be positive"
print("\n✓ Checkpoint 9 passed — reusable analysis functions working correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Comprehensive ranked district report
# ══════════════════════════════════════════════════════════════════════
# Combine everything into a publication-quality district ranking report.


def district_report_line(row: dict) -> str:
    """Format one district row as a human-readable report line."""
    town = row["town"]
    median = format_sgd(row["median_price"])
    count = row["transaction_count"]
    cv = row["cv_price_pct"]
    sqm = format_sgd(row["median_price_sqm"])
    position = row.get("market_position", "?")
    return (
        f"  {town:<22} {median:>12}  {count:>8,}  "
        f"CV={cv:5.1f}%  {sqm:>10}/sqm  [{position}]"
    )


# --- Full ranked report ---
print(f"\n{'═' * 78}")
print(f"  SINGAPORE HDB DISTRICT PRICE REPORT (All Years)")
print(f"{'═' * 78}")
print(
    f"  {'Town':<22} {'Median':>12}  {'Txns':>8}  "
    f"{'Spread':>8}  {'Per sqm':>10}  {'Position'}"
)
print(f"  {'─' * 74}")

for row in district_stats.iter_rows(named=True):
    print(district_report_line(row))

print(f"{'═' * 78}")

# --- Summary statistics across all districts ---
all_medians = district_stats["median_price"]
all_cvs = district_stats["cv_price_pct"]
all_counts = district_stats["transaction_count"]

print(f"\n=== Cross-District Summary ===")
print(f"  Total districts:           {district_stats.height}")
print(f"  Total transactions:        {all_counts.sum():,}")
print(f"  Most expensive district:   {format_sgd(all_medians.max())}")
print(f"  Least expensive district:  {format_sgd(all_medians.min())}")
print(f"  Average district median:   {format_sgd(all_medians.mean())}")
print(
    f"  Price gap (max - min):     {format_sgd(all_medians.max() - all_medians.min())}"
)
print(f"  Average CV:                {all_cvs.mean():.1f}%")
print(
    f"  Most active district:      {district_stats.sort('transaction_count', descending=True)['town'][0]}"
)
print(
    f"  Least active district:     {district_stats.sort('transaction_count')['town'][0]}"
)

# --- Market position breakdown ---
position_counts = (
    district_stats.group_by("market_position")
    .agg(
        pl.len().alias("districts"),
        pl.col("transaction_count").sum().alias("total_transactions"),
        pl.col("median_price").mean().alias("avg_median"),
    )
    .sort("avg_median", descending=True)
)

print(f"\n=== Market Position Breakdown ===")
for row in position_counts.iter_rows(named=True):
    print(
        f"  {row['market_position']:<12} "
        f"{row['districts']:>3} districts  "
        f"{row['total_transactions']:>8,} txns  "
        f"avg median={format_sgd(row['avg_median'])}"
    )

# INTERPRETATION: The spread between the most and least expensive districts
# quantifies Singapore's housing inequality. A wide spread (>S$200k) indicates
# that geography still matters significantly — where you buy affects your
# long-term asset value, not just your commute.

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert all_medians.max() > all_medians.min(), "Most expensive > least expensive"
assert all_counts.sum() == hdb.height, "Total transactions should match dataset"
assert position_counts.height > 0, "Should have market position breakdown"
print("\n✓ Checkpoint 10 passed — comprehensive report generated correctly\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print(
    """
  ✓ Functions: def, parameters, return, type hints, docstrings
  ✓ Default parameters: optional arguments with sensible defaults
  ✓ Conditional logic: if/elif/else inside functions
  ✓ Statistical helpers: IQR, CV, safe_divide, describe_series
  ✓ Lists and dicts: comprehensions, zip, enumerate, nested structures
  ✓ group_by() + agg(): the core pattern for grouped statistics
  ✓ Multiple aggregations: mean, median, std, min, max, quantile
  ✓ Derived columns: IQR, CV, market position from aggregated data
  ✓ Multi-key grouping: group by (town, flat_type) simultaneously
  ✓ Time-series aggregation: annual trends, YoY growth, quarterly patterns
  ✓ Reusable functions: town_deep_dive() — parameterised analysis
  ✓ for loops: .iter_rows(named=True) to process each row as a dict
  ✓ Report building: combining statistics into formatted output

  NEXT: In Exercise 4, you'll combine data from multiple tables
  using joins — merging HDB transactions with MRT station proximity
  and school density data. You'll learn when to use left vs inner
  joins, how to handle NULLs after a join, and how to enrich a
  dataset with spatial context.
"""
)
