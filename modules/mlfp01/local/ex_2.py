# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 2: Filtering and Transforming Data
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Filter rows using Boolean logic with comparison operators (>, <, ==, !=)
#   - Combine filters with & (AND), | (OR), and ~ (NOT)
#   - Select, rename, and reorder columns to focus your analysis
#   - Create new derived columns with with_columns() and alias()
#   - Apply conditional column logic with pl.when().then().otherwise()
#   - Chain multiple Polars operations together in a readable pipeline
#
# PREREQUISITES: Complete Exercise 1 first (variables, DataFrames, describe()).
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  Inspect the HDB dataset and understand its structure
#   2.  Single-condition filters — one rule at a time
#   3.  Compound filters — combining multiple conditions with & and |
#   4.  Negation and set membership — ~ and .is_in()
#   5.  Select, rename, and reorder columns
#   6.  Create derived columns with with_columns() and alias()
#   7.  Date parsing and extraction — string to date, extract year/month
#   8.  Conditional columns with pl.when().then().otherwise()
#   9.  Sorting — single-key, multi-key, and descending
#   10. Method chaining — building analysis pipelines step by step
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

print("=" * 60)
print("  MLFP01 Exercise 2: Filtering and Transforming Data")
print("=" * 60)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.height:,} rows, {hdb.width} columns)")
print(f"  You're ready to start!\n")

print("=== HDB Resale Dataset ===")
print(f"Shape: {hdb.shape}")
print(f"Columns: {hdb.columns}")
print(hdb.head(3))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Inspect the HDB dataset — understand before you filter
# ══════════════════════════════════════════════════════════════════════
# Before writing any filter, understand what values exist in each column.
# Otherwise you'll filter for "Ang Mo Kio" when the data says "ANG MO KIO"
# and get zero rows back — a common beginner mistake.

# --- 1a: Data types and null counts ---
print("\n=== Column Details ===")
for col_name, dtype in zip(hdb.columns, hdb.dtypes):
    null_count = hdb[col_name].null_count()
    null_pct = null_count / hdb.height * 100
    print(f"  {col_name:>25}: {str(dtype):<12} nulls={null_count:,} ({null_pct:.1f}%)")

# --- 1b: Unique values for categorical columns ---
towns = hdb["town"].unique().sort()
flat_types = hdb["flat_type"].unique().sort()

print(f"\n=== Unique Towns ({towns.len()}) ===")
print(f"  {towns.to_list()}")
print(f"\n=== Flat Types ({flat_types.len()}) ===")
print(f"  {flat_types.to_list()}")

# --- 1c: Price distribution overview ---
print(f"\n=== Price Overview ===")
print(f"  Min:    S${hdb['resale_price'].min():>12,.0f}")
print(f"  Q1:     S${hdb['resale_price'].quantile(0.25):>12,.0f}")
print(f"  Median: S${hdb['resale_price'].quantile(0.50):>12,.0f}")
print(f"  Q3:     S${hdb['resale_price'].quantile(0.75):>12,.0f}")
print(f"  Max:    S${hdb['resale_price'].max():>12,.0f}")

# --- 1d: Date range ---
date_min = hdb["month"].min()
date_max = hdb["month"].max()
print(f"\n=== Date Range ===")
print(f"  Earliest: {date_min}")
print(f"  Latest:   {date_max}")
# INTERPRETATION: Understanding the data range is essential for filtering.
# If the earliest date is 2012-01 and you filter for 2010, you'll get zero
# rows and think your code is broken. Always check ranges first.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert towns.len() > 0, "Should have at least one town"
assert flat_types.len() > 0, "Should have at least one flat type"
assert hdb["resale_price"].min() > 0, "Min price should be positive"
assert date_min is not None, "Date min should not be None"
print("\n✓ Checkpoint 1 passed — dataset inspected, ranges understood\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Single-condition filters — one rule at a time
# ══════════════════════════════════════════════════════════════════════
# A Boolean is True or False. Polars lets you write conditions using
# pl.col("column_name") and comparison operators: ==, !=, >, <, >=, <=
# .filter() keeps rows where the expression evaluates to True.

# --- 2a: Filter by exact string match ---
# TODO: Filter hdb for rows where town equals "ANG MO KIO"
ang_mo_kio = hdb.filter(pl.col("town") == ____)  # Hint: "ANG MO KIO"
print(f"\nAng Mo Kio transactions: {ang_mo_kio.height:,}")
print(f"  ({ang_mo_kio.height / hdb.height:.1%} of all transactions)")

# --- 2b: Filter by numeric comparison ---
expensive = hdb.filter(pl.col("resale_price") > 1_000_000)
print(f"Million-dollar HDBs (>S$1M): {expensive.height:,}")
print(f"  Most common town: {expensive['town'].mode()[0]}")

# --- 2c: Filter by price range ---
affordable = hdb.filter(
    (pl.col("resale_price") >= 300_000) & (pl.col("resale_price") <= 500_000)
)
print(f"Transactions S$300k-500k: {affordable.height:,}")

# --- 2d: Filter by flat type ---
# TODO: Filter for 4-room flats
four_room = hdb.filter(pl.col("flat_type") == ____)  # Hint: "4 ROOM"
print(f"4-room flats: {four_room.height:,}")

# --- 2e: Filter by floor area ---
large_flats = hdb.filter(pl.col("floor_area_sqm") >= 120)
print(f"Large flats (>=120 sqm): {large_flats.height:,}")
print(f"  Avg price: S${large_flats['resale_price'].mean():,.0f}")

# --- 2f: String-based filter with .str accessor ---
mature_estates = hdb.filter(pl.col("town").str.contains("QUEENSTOWN|BISHAN|TOA PAYOH"))
print(f"Mature estates (Queenstown/Bishan/Toa Payoh): {mature_estates.height:,}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert ang_mo_kio.height > 0, "AMK filter returned no rows — check column values"
assert affordable.height < hdb.height, "Affordable filter should reduce row count"
assert four_room.height > 0, "4-room filter should return rows"
assert large_flats.height < hdb.height, "Large flat filter should reduce count"
assert mature_estates.height > 0, "Mature estates filter should return rows"
print("\n✓ Checkpoint 2 passed — single-condition filters working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compound filters — combining multiple conditions
# ══════════════════════════════════════════════════════════════════════
# & means AND — BOTH conditions must be True
# | means OR — EITHER condition can be True
# You MUST wrap each condition in parentheses when combining.

# --- 3a: AND — all conditions must be True ---
# Business question: "How many 4-room flats in Ang Mo Kio sold under S$500k?"
# TODO: Combine three conditions: town == AMK AND flat_type == 4 ROOM AND price <= 500k
amk_4room_affordable = hdb.filter(
    (pl.col("town") == "ANG MO KIO")
    & (pl.col("flat_type") == ____)  # Hint: "4 ROOM"
    & (pl.col("resale_price") <= ____)  # Hint: 500_000
)
print(f"AMK 4-room under S$500k: {amk_4room_affordable.height:,}")

# --- 3b: OR — either condition can be True ---
extreme_prices = hdb.filter(
    (pl.col("resale_price") < 200_000) | (pl.col("resale_price") > 900_000)
)
print(f"Extreme prices (<200k or >900k): {extreme_prices.height:,}")

# --- 3c: AND + OR combined ---
central_towns = ["BISHAN", "TOA PAYOH", "QUEENSTOWN", "BUKIT MERAH"]
central_special = hdb.filter(
    pl.col("town").is_in(central_towns)
    & ((pl.col("floor_area_sqm") >= 130) | (pl.col("resale_price") >= 800_000))
)
print(f"Central large-or-expensive: {central_special.height:,}")

# --- 3d: Progressive filtering (chaining .filter() calls) ---
progressive = (
    hdb.filter(pl.col("town") == "BISHAN")
    .filter(pl.col("flat_type").is_in(["4 ROOM", "5 ROOM"]))
    .filter(pl.col("resale_price") >= 700_000)
    .filter(pl.col("resale_price") <= 1_000_000)
)
print(f"Bishan 4/5-room S$700k-1M: {progressive.height:,}")
# INTERPRETATION: Compound filters answer specific business questions.
# A property investor asking "What can I get in Bishan for S$700k-1M?"
# would use exactly this filter (Bishan is a mature, central estate, so
# 4/5-room flats sit in this band).

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    amk_4room_affordable.height <= ang_mo_kio.height
), "Combined filter should be a subset of single-town filter"
assert extreme_prices.height > 0, "Should have some extreme-price transactions"
assert central_special.height > 0, "Central special filter should return rows"
assert progressive.height > 0, "Progressive filter should return rows"
print("\n✓ Checkpoint 3 passed — compound filters working correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Negation and set membership — ~ and .is_in()
# ══════════════════════════════════════════════════════════════════════

# --- 4a: Negation with ~ ---
not_amk = hdb.filter(~(pl.col("town") == "ANG MO KIO"))
print(f"Not Ang Mo Kio: {not_amk.height:,}")
assert (
    not_amk.height + ang_mo_kio.height == hdb.height
), "AMK + not-AMK should equal total"
print(f"  Check: {ang_mo_kio.height:,} + {not_amk.height:,} = {hdb.height:,} ✓")

# --- 4b: .is_in() — match against a list of values ---
central_towns_list = ["BISHAN", "TOA PAYOH", "QUEENSTOWN", "BUKIT MERAH", "BUKIT TIMAH"]
# TODO: Filter for rows where town is in central_towns_list
central = hdb.filter(pl.col("town").is_in(____))  # Hint: central_towns_list
print(
    f"\nCentral towns ({len(central_towns_list)} towns): {central.height:,} transactions"
)

# --- 4c: NOT in a set ---
peripheral = hdb.filter(~pl.col("town").is_in(central_towns_list))
print(f"Peripheral towns: {peripheral.height:,} transactions")
assert (
    central.height + peripheral.height == hdb.height
), "Central + peripheral should equal total"

# --- 4d: .is_between() — range filter shorthand ---
mid_range = hdb.filter(pl.col("resale_price").is_between(400_000, 600_000))
print(f"\nS$400k-600k (is_between): {mid_range.height:,}")

# --- 4e: .is_null() and .is_not_null() ---
for col_name in hdb.columns:
    nc = hdb[col_name].null_count()
    if nc > 0:
        print(f"  {col_name}: {nc:,} nulls")

non_null_rows = hdb.filter(pl.col("resale_price").is_not_null())
print(f"Rows with non-null price: {non_null_rows.height:,}")

# INTERPRETATION: .is_in() is the Polars equivalent of SQL's IN clause.
# It's faster and more readable than chaining OR conditions.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert not_amk.height > 0, "Negation filter should return rows"
assert central.height > 0, "is_in filter should return rows"
assert central.height + peripheral.height == hdb.height, "Central + peripheral = total"
assert mid_range.height > 0, "is_between should return rows"
print("\n✓ Checkpoint 4 passed — negation, set membership, and null checks working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Select, rename, and reorder columns
# ══════════════════════════════════════════════════════════════════════

# --- 5a: .select() — keep only the columns you need ---
core_cols = hdb.select(
    "month",
    "town",
    "flat_type",
    "floor_area_sqm",
    "resale_price",
)
print(f"\nAfter select: {core_cols.columns}")
print(f"Columns: {core_cols.width} (was {hdb.width})")

# --- 5b: .select() with expressions — compute during selection ---
summary_view = hdb.select(
    "town",
    "flat_type",
    "resale_price",
    # TODO: Compute price per sqm and alias it as "price_per_sqm"
    (pl.col("resale_price") / pl.col(____)).alias(
        "price_per_sqm"
    ),  # Hint: "floor_area_sqm"
)
print(f"\nSummary view columns: {summary_view.columns}")
print(summary_view.head(3))

# --- 5c: .rename() — change column names ---
renamed = core_cols.rename(
    {
        "month": "sale_month",
        "floor_area_sqm": "area_sqm",
        "resale_price": "price",
    }
)
print(f"\nAfter rename: {renamed.columns}")
print(renamed.head(3))

# --- 5d: Column ordering ---
reordered = hdb.select(
    "resale_price",  # Price first
    "town",
    "flat_type",
    "floor_area_sqm",
    "month",
)
print(f"\nReordered columns: {reordered.columns}")

# --- 5e: .drop() — remove specific columns ---
dropped = hdb.drop("block", "street_name")
print(f"\nAfter dropping block, street_name: {dropped.width} columns (was {hdb.width})")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert core_cols.width == 5, "core_cols should have exactly 5 columns"
assert "resale_price" not in renamed.columns, "resale_price should be renamed to price"
assert "price" in renamed.columns, "renamed DataFrame should have a 'price' column"
assert "price_per_sqm" in summary_view.columns, "summary_view should have price_per_sqm"
assert reordered.columns[0] == "resale_price", "First column should be resale_price"
print("\n✓ Checkpoint 5 passed — select, rename, reorder, and drop working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Derived columns with with_columns() and alias()
# ══════════════════════════════════════════════════════════════════════
# .with_columns() adds new columns without removing any other columns.
# .alias() gives the new column a name.

# --- 6a: Price per square metre ---
# TODO: Add price_per_sqm column using with_columns() and alias()
hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias(
        ____
    )  # Hint: "price_per_sqm"
)
print(f"\n=== Price per sqm ===")
print(f"  Mean:   S${hdb['price_per_sqm'].mean():,.0f}/sqm")
print(f"  Median: S${hdb['price_per_sqm'].median():,.0f}/sqm")

# --- 6b: Multiple columns in one call ---
hdb = hdb.with_columns(
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
    pl.col("month").str.slice(5, 2).cast(pl.Int32).alias("month_num"),
)

# --- 6c: Floor area categories ---
hdb = hdb.with_columns(
    pl.when(pl.col("floor_area_sqm") < 70)
    .then(pl.lit("small"))
    .when(pl.col("floor_area_sqm") < 100)
    .then(pl.lit("medium"))
    .when(pl.col("floor_area_sqm") < 130)
    .then(pl.lit("large"))
    .otherwise(pl.lit("executive"))
    .alias("size_category"),
)

# --- 6d: Price per sqm quartile flag ---
psm_q75 = hdb["price_per_sqm"].quantile(0.75)
psm_q25 = hdb["price_per_sqm"].quantile(0.25)

hdb = hdb.with_columns(
    pl.when(pl.col("price_per_sqm") >= psm_q75)
    .then(pl.lit("premium"))
    .when(pl.col("price_per_sqm") <= psm_q25)
    .then(pl.lit("value"))
    .otherwise(pl.lit("mainstream"))
    .alias("market_segment"),
)

print(f"\n=== After adding derived columns ===")
new_cols = [
    "price_per_sqm",
    "transaction_date",
    "year",
    "month_num",
    "size_category",
    "market_segment",
]
print(hdb.select(new_cols).head(5))

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert "price_per_sqm" in hdb.columns, "price_per_sqm column should be added"
assert "transaction_date" in hdb.columns, "transaction_date column should be added"
assert "year" in hdb.columns, "year column should be added"
assert "size_category" in hdb.columns, "size_category column should be added"
assert "market_segment" in hdb.columns, "market_segment column should be added"
sample_psm = hdb["price_per_sqm"].drop_nulls()[0]
assert sample_psm > 0, "price_per_sqm should be positive"
print("\n✓ Checkpoint 6 passed — derived columns created correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Date parsing and extraction
# ══════════════════════════════════════════════════════════════════════
# Dates are crucial for time-series analysis. Polars stores dates as
# actual Date types, not strings — this enables arithmetic and extraction.

# --- 7a: Date arithmetic ---
earliest = hdb["transaction_date"].min()
latest = hdb["transaction_date"].max()
date_span = (latest - earliest).days

print(f"=== Date Analysis ===")
print(f"Earliest transaction: {earliest}")
print(f"Latest transaction:   {latest}")
print(f"Span: {date_span:,} days ({date_span / 365.25:.1f} years)")

# --- 7b: Temporal distribution ---
yearly_counts = hdb.group_by("year").agg(pl.len().alias("count")).sort("year")
print(f"\n=== Transactions per Year ===")
for row in yearly_counts.iter_rows(named=True):
    bar = "█" * (row["count"] // 2000)
    print(f"  {row['year']}: {row['count']:>7,} {bar}")

# --- 7c: Monthly seasonality ---
monthly_counts = (
    hdb.group_by("month_num").agg(pl.len().alias("count")).sort("month_num")
)
print(f"\n=== Transactions by Month (all years) ===")
month_names = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}
for row in monthly_counts.iter_rows(named=True):
    name = month_names.get(row["month_num"], "?")
    bar = "█" * (row["count"] // 1000)
    print(f"  {name}: {row['count']:>7,} {bar}")

# --- 7d: Extract quarter ---
hdb = hdb.with_columns(
    ((pl.col("month_num") - 1) // 3 + 1).alias("quarter"),
)
quarterly = (
    hdb.group_by("year", "quarter")
    .agg(
        pl.len().alias("transactions"),
        pl.col("resale_price").median().alias("median_price"),
    )
    .sort("year", "quarter")
)
print(f"\n=== Quarterly Summary (last 8 quarters) ===")
print(quarterly.tail(8))

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert date_span > 0, "Date span should be positive"
assert yearly_counts.height > 0, "Should have yearly counts"
assert monthly_counts.height == 12, "Should have 12 monthly counts"
assert "quarter" in hdb.columns, "quarter column should exist"
print("\n✓ Checkpoint 7 passed — date parsing and temporal analysis working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Conditional columns with pl.when().then().otherwise()
# ══════════════════════════════════════════════════════════════════════
# pl.when() is Polars' if/else for column creation.
# You can chain .when().then() as many times as you need.

# --- 8a: Price tier classification ---
# TODO: Add a price_tier column: budget (<350k), mid_range (<500k),
#       premium (<700k), luxury (>=700k)
hdb = hdb.with_columns(
    pl.when(pl.col("resale_price") < 350_000)
    .then(pl.lit("budget"))
    .when(pl.col("resale_price") < ____)  # Hint: 500_000
    .then(pl.lit("mid_range"))
    .when(pl.col("resale_price") < 700_000)
    .then(pl.lit("premium"))
    .otherwise(pl.lit(____))  # Hint: "luxury"
    .alias("price_tier")
)

tier_counts = (
    hdb.group_by("price_tier")
    .agg(
        pl.len().alias("count"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("floor_area_sqm").median().alias("median_area"),
    )
    .sort("median_price")
)
print(f"\n=== Price Tier Distribution ===")
for row in tier_counts.iter_rows(named=True):
    pct = row["count"] / hdb.height * 100
    bar = "█" * int(pct)
    print(
        f"  {row['price_tier']:<12} {row['count']:>8,} ({pct:5.1f}%) "
        f"median=S${row['median_price']:>10,.0f}  area={row['median_area']:.0f}sqm  {bar}"
    )

# --- 8b: Transaction era classification ---
hdb = hdb.with_columns(
    pl.when(pl.col("year") <= 2015)
    .then(pl.lit("pre-2016"))
    .when(pl.col("year") <= 2019)
    .then(pl.lit("2016-2019"))
    .when(pl.col("year") <= 2022)
    .then(pl.lit("2020-2022"))
    .otherwise(pl.lit("2023+"))
    .alias("transaction_era")
)

era_summary = (
    hdb.group_by("transaction_era")
    .agg(
        pl.len().alias("count"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_psm"),
    )
    .sort("transaction_era")
)
print(f"\n=== Price by Transaction Era ===")
print(era_summary)

# --- 8c: Boolean flag columns ---
hdb = hdb.with_columns(
    (pl.col("resale_price") >= 1_000_000).alias("is_million_dollar"),
    (pl.col("floor_area_sqm") >= 100).alias("is_large_flat"),
)

million_count = hdb.filter(pl.col("is_million_dollar")).height
large_count = hdb.filter(pl.col("is_large_flat")).height
print(f"\nMillion-dollar HDBs: {million_count:,} ({million_count / hdb.height:.1%})")
print(f"Large flats (>=100sqm): {large_count:,} ({large_count / hdb.height:.1%})")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert "price_tier" in hdb.columns, "price_tier column should be added"
tier_values = set(hdb["price_tier"].unique().to_list())
expected_tiers = {"budget", "mid_range", "premium", "luxury"}
assert (
    tier_values == expected_tiers
), f"Expected tiers {expected_tiers}, got {tier_values}"
assert "transaction_era" in hdb.columns, "transaction_era should exist"
assert "is_million_dollar" in hdb.columns, "is_million_dollar should exist"
print("\n✓ Checkpoint 8 passed — conditional columns created with all tiers\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Sorting — single-key, multi-key, and stable sorts
# ══════════════════════════════════════════════════════════════════════

# --- 9a: Single-key sort ---
# TODO: Sort hdb by resale_price descending
by_price_desc = hdb.sort(____, descending=____)  # Hint: "resale_price", True
print(f"=== Most Expensive Transactions ===")
print(
    by_price_desc.select(
        "town", "flat_type", "resale_price", "floor_area_sqm", "month"
    ).head(5)
)

by_price_asc = hdb.sort("resale_price")
print(f"\n=== Cheapest Transactions ===")
print(
    by_price_asc.select(
        "town", "flat_type", "resale_price", "floor_area_sqm", "month"
    ).head(5)
)

# --- 9b: Multi-key sort ---
by_town_price = hdb.sort("town", "resale_price", descending=[False, True])
print(f"\n=== Sorted by Town then Price (desc) ===")
seen_towns: set[str] = set()
for row in by_town_price.iter_rows(named=True):
    if row["town"] not in seen_towns:
        seen_towns.add(row["town"])
        print(f"  {row['town']:<20} S${row['resale_price']:>10,.0f}")
    if len(seen_towns) >= 5:
        break

# --- 9c: Sort by computed expression ---
by_psm = hdb.sort("price_per_sqm", descending=True)
print(f"\n=== Highest Price per sqm ===")
print(by_psm.select("town", "flat_type", "price_per_sqm", "resale_price").head(5))

# --- 9d: Sorting for analysis — find the middle of the market ---
n = hdb.height
middle_idx = n // 2
sorted_df = hdb.sort("resale_price")
middle_row = sorted_df.row(middle_idx, named=True)
print(f"\n=== Middle-of-Market Transaction (row {middle_idx:,} of {n:,}) ===")
print(f"  Town: {middle_row['town']}, Type: {middle_row['flat_type']}")
print(
    f"  Price: S${middle_row['resale_price']:,.0f}, Area: {middle_row['floor_area_sqm']:.0f} sqm"
)

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert (
    by_price_desc["resale_price"][0] >= by_price_desc["resale_price"][-1]
), "Descending sort: first should be >= last"
assert (
    by_price_asc["resale_price"][0] <= by_price_asc["resale_price"][-1]
), "Ascending sort: first should be <= last"
print("\n✓ Checkpoint 9 passed — sorting working correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Method chaining — building analysis pipelines
# ══════════════════════════════════════════════════════════════════════
# Method chaining: instead of saving each intermediate step to a variable,
# you attach the next operation with a dot. Polars is designed for this.

# --- 10a: Recent premium properties pipeline ---
recent_premium = (
    hdb.filter(pl.col("year") >= 2020)
    .filter(pl.col("price_tier").is_in(["premium", "luxury"]))
    .select(
        "transaction_date",
        "town",
        "flat_type",
        "price_per_sqm",
        "price_tier",
        "resale_price",
    )
    .sort("resale_price", descending=True)
)

print(f"\n=== Recent Premium/Luxury Transactions (2020+) ===")
print(f"Count: {recent_premium.height:,}")
print(recent_premium.head(10))

# --- 10b: Town comparison pipeline ---
top_towns = (
    recent_premium.group_by("town")
    .agg(
        pl.len().alias("transaction_count"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
    )
    .sort("transaction_count", descending=True)
)

print(f"\n=== Towns with Most Premium/Luxury Transactions (2020+) ===")
print(top_towns.head(10))

# --- 10c: Year-over-year price growth pipeline ---
annual_median = (
    hdb.group_by("year")
    .agg(
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_psm"),
        pl.len().alias("transactions"),
    )
    .sort("year")
)

print(f"\n=== Annual Market Summary ===")
prev_price = None
for row in annual_median.iter_rows(named=True):
    price = row["median_price"]
    if prev_price is not None:
        yoy = (price - prev_price) / prev_price * 100
        arrow = "↑" if yoy > 0 else "↓" if yoy < 0 else "→"
        print(
            f"  {row['year']}: S${price:>10,.0f}  "
            f"{arrow} {yoy:+.1f}%  "
            f"({row['transactions']:,} txns)"
        )
    else:
        print(
            f"  {row['year']}: S${price:>10,.0f}  "
            f"(baseline)  "
            f"({row['transactions']:,} txns)"
        )
    prev_price = price

# --- 10d: Full analysis pipeline ---
# TODO: Build a chained pipeline that filters 2022+ 4/5-room flats,
#       groups by town, aggregates volume + median + std, and sorts by median PSM
investment_report = (
    hdb.filter(pl.col("year") >= ____)  # Hint: 2022
    .filter(pl.col("flat_type").is_in(["4 ROOM", "5 ROOM"]))
    .group_by("town")
    .agg(
        pl.len().alias("volume"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_psm"),
        pl.col("resale_price").std().alias("price_volatility"),
    )
    .with_columns(
        (pl.col("price_volatility") / pl.col("median_price") * 100).alias("cv_pct"),
    )
    .sort("median_psm", descending=True)
)

print(f"\n=== Investment Report: 4/5-Room Flats (2022+) ===")
print(f"{'Town':<22} {'Vol':>6} {'Median':>12} {'PSM':>10} {'CV%':>6}")
print(f"{'─' * 58}")
for row in investment_report.head(15).iter_rows(named=True):
    print(
        f"{row['town']:<22} {row['volume']:>6,} "
        f"S${row['median_price']:>10,.0f} "
        f"S${row['median_psm']:>8,.0f} "
        f"{row['cv_pct']:>5.1f}%"
    )

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert recent_premium.height > 0, "recent_premium should have rows"
assert top_towns.height > 0, "top_towns should have rows"
assert annual_median.height > 0, "annual_median should have rows"
assert investment_report.height > 0, "investment_report should have rows"
print("\n✓ Checkpoint 10 passed — method chaining pipelines working\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print(
    """
  ✓ Single-condition filters: ==, !=, >, <, >=, <=
  ✓ Compound filters: & (AND), | (OR), ~ (NOT)
  ✓ Set membership: .is_in(), .is_between(), .is_null()
  ✓ Column selection: .select(), .drop(), .rename()
  ✓ Derived columns: .with_columns() + .alias()
  ✓ Date parsing: .str.to_date(), .str.slice(), .dt.truncate()
  ✓ Conditional columns: pl.when().then().otherwise()
  ✓ Sorting: single key, multi-key, descending
  ✓ Method chaining: building readable analysis pipelines

  NEXT: In Exercise 3, you'll learn to write reusable functions
  and aggregate data by groups using group_by() + agg(). This is
  the foundation for all market analysis and reporting.
"""
)
