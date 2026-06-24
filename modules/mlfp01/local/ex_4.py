# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 4: Joins and Multi-Table Data
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Write conditional logic for branching decisions (if/elif/else)
#   - Import and use external packages
#   - Join multiple DataFrames on shared keys using .join()
#   - Reason about which join type to use (left vs inner vs outer)
#   - Handle missing values that arise after a join with fill_null()
#
# PREREQUISITES: Complete Exercise 3 first (functions, group_by/agg).
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  Inspect HDB, MRT, and school datasets independently
#   2.  Join key analysis — overlap, mismatches, and data preparation
#   3.  Left join — enrich HDB with MRT station data
#   4.  Left join — enrich with school density data
#   5.  Inner join vs left join — understand the difference
#   6.  Handle nulls from joins — fill_null, coalesce, drop_nulls
#   7.  if/elif/else — conditional logic for data classification
#   8.  import and packages — using math and datetime modules
#   9.  District-level summary with spatial features
#   10. Correlation analysis — do amenities predict price?
#
# DATASET: Three Singapore datasets joined together:
#   - HDB resale transactions (Housing & Development Board, data.gov.sg)
#   - MRT station proximity by town (pre-computed from LTA data)
#   - School density by town (pre-computed from MOE data)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import polars as pl

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()

hdb = loader.load("mlfp01", "hdb_resale.parquet")
mrt_stations = loader.load("mlfp_assessment", "mrt_stations.parquet")
schools = loader.load("mlfp_assessment", "schools.parquet")

# Add derived columns to HDB
hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
)

print("=" * 60)
print("  MLFP01 Exercise 4: Joins and Multi-Table Data")
print("=" * 60)
print(f"\n  Data loaded:")
print(f"    hdb_resale.parquet        ({hdb.height:,} rows, {hdb.width} cols)")
print(
    f"    mrt_stations.parquet      ({mrt_stations.height:,} rows, {mrt_stations.width} cols)"
)
print(f"    schools.parquet           ({schools.height:,} rows, {schools.width} cols)")
print(f"  You're ready to start!\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Inspect each dataset independently
# ══════════════════════════════════════════════════════════════════════
# Before joining, always understand each table on its own:
# - What is the grain? (one row = one what?)
# - What are the key columns used for joining?
# - Are there nulls in the join keys?

# --- 1a: HDB resale data ---
print("=== HDB Resale Data ===")
print(f"Shape: {hdb.shape}")
print(f"Columns: {hdb.columns}")
print(f"Grain: one row = one resale transaction")
print(f"Join key: 'town'")
print(f"Towns: {hdb['town'].n_unique()} unique values")
print(f"Null towns: {hdb['town'].null_count()}")
print(hdb.head(3))

# --- 1b: MRT station data ---
print("\n=== MRT Stations ===")
print(f"Shape: {mrt_stations.shape}")
print(f"Columns: {mrt_stations.columns}")
print(f"Grain: one row = one MRT station/town record")
print(f"Towns: {mrt_stations['town'].n_unique()} unique values")
print(mrt_stations.head(5))

# --- 1c: Schools data ---
print("\n=== Schools ===")
print(f"Shape: {schools.shape}")
print(f"Columns: {schools.columns}")
print(f"Grain: one row = one school")
print(f"Towns: {schools['town'].n_unique()} unique values")
print(schools.head(5))
# INTERPRETATION: Each dataset has a different grain — HDB is per-transaction,
# MRT is per-town/station, and schools is per-school. When joining, you need
# to think about which table's grain determines the result. A left join on
# HDB keeps one row per transaction; the right table's values get repeated
# for every transaction in the same town.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert hdb.height > 0, "HDB dataset is empty"
assert mrt_stations.height > 0, "MRT stations dataset is empty"
assert schools.height > 0, "Schools dataset is empty"
assert "town" in hdb.columns, "HDB should have a 'town' column"
assert "town" in mrt_stations.columns, "MRT should have a 'town' column"
assert "town" in schools.columns, "Schools should have a 'town' column"
print("\n✓ Checkpoint 1 passed — all three datasets loaded and inspected\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Join key analysis — the most critical pre-join step
# ══════════════════════════════════════════════════════════════════════
# The #1 cause of broken joins is mismatched join keys. Town names in
# HDB might be "ANG MO KIO" while MRT data has "Ang Mo Kio". Case
# differences, extra spaces, and abbreviations all break joins silently.

# --- 2a: Compare town names across datasets ---
# HDB stores towns UPPERCASE ("BISHAN") while MRT/schools use Title Case
# ("Bishan"). Intersecting the RAW values matches nothing — the #1 silent
# join bug. The fix (and the lesson) is to normalise both sides to a common
# case BEFORE comparing or joining.
hdb_towns = set(hdb["town"].unique().to_list())
mrt_towns = {t.upper() for t in mrt_stations["town"].unique().to_list()}
school_towns = {t.upper() for t in schools["town"].unique().to_list()}

matched_mrt = hdb_towns & mrt_towns
unmatched_hdb_mrt = hdb_towns - mrt_towns
unmatched_mrt_hdb = mrt_towns - hdb_towns

matched_schools = hdb_towns & school_towns
unmatched_hdb_schools = hdb_towns - school_towns

print("=== Join Key Analysis: HDB ↔ MRT ===")
print(f"  HDB towns:           {len(hdb_towns)}")
print(f"  MRT towns:           {len(mrt_towns)}")
print(f"  Matched:             {len(matched_mrt)}")
print(f"  HDB-only (no MRT):   {len(unmatched_hdb_mrt)}")
if unmatched_hdb_mrt:
    print(f"    Towns: {sorted(unmatched_hdb_mrt)}")
print(f"  MRT-only (no HDB):   {len(unmatched_mrt_hdb)}")
if unmatched_mrt_hdb:
    print(f"    Towns: {sorted(unmatched_mrt_hdb)}")

print(f"\n=== Join Key Analysis: HDB ↔ Schools ===")
print(f"  Matched:             {len(matched_schools)}")
print(f"  HDB-only (no school):{len(unmatched_hdb_schools)}")
if unmatched_hdb_schools:
    print(f"    Towns: {sorted(unmatched_hdb_schools)}")

# --- 2b: Predict join outcomes ---
# Left join: all HDB rows kept, unmatched get NULLs
# Inner join: only matched rows kept, unmatched DROPPED
unmatched_hdb_txns = hdb.filter(~pl.col("town").is_in(list(matched_mrt))).height

print(f"\n=== Predicted Join Impact ===")
print(f"  Left join:  {hdb.height:,} rows (all HDB rows preserved)")
print(
    f"  Inner join: {hdb.height - unmatched_hdb_txns:,} rows "
    f"({unmatched_hdb_txns:,} unmatched rows would be dropped)"
)
print(f"  Drop rate:  {unmatched_hdb_txns / hdb.height:.1%}")
# INTERPRETATION: "Unmatched" towns will get NULL values for MRT distance
# after a left join. An inner join drops them entirely. The choice depends
# on your analysis: if missing MRT data means "no nearby MRT" (a real
# signal), use left join + fill_null(large_number). If it means "data
# quality issue", consider inner join to keep only clean records.

# --- 2c: Case sensitivity check ---
# Demonstrate why the normalisation above was necessary: the RAW values
# differ only in case, so a naive join silently matches nothing.
hdb_sample_town = hdb["town"][0]
mrt_sample_town = mrt_stations["town"][0]
print(f"\n  HDB town format (raw): {hdb_sample_town!r}")
print(f"  MRT town format (raw): {mrt_sample_town!r}")
print(f"  Same case (raw)?       {hdb_sample_town == mrt_sample_town}")
print(f"  Match after .upper()?  {hdb_sample_town == mrt_sample_town.upper()}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(matched_mrt) > 0, "At least some towns should match between HDB and MRT"
assert (
    len(matched_schools) > 0
), "At least some towns should match between HDB and Schools"
print("\n✓ Checkpoint 2 passed — join key analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Left join — enrich HDB with MRT station data
# ══════════════════════════════════════════════════════════════════════
# A left join keeps ALL rows from the left table (hdb) and adds
# matching columns from the right table (mrt_stations).
# Rows in hdb with no match get NULL for the new columns.
#
# how="left"  -> keep all HDB rows regardless of match
# on="town"   -> match rows where hdb.town == mrt_stations.town

# --- 3a: Select specific columns from the right table ---
# .select() on the right table prevents duplicate columns and limits
# which columns are brought across — always explicit about what you join.
mrt_join_cols = (
    mrt_stations.with_columns(pl.col("town").str.to_uppercase())
    .sort("distance_to_mrt_km")
    .group_by("town")
    .agg(
        pl.col("nearest_mrt").first(),
        pl.col("distance_to_mrt_km").first(),
    )
)
print(f"MRT columns to join: {mrt_join_cols.columns}")
print(mrt_join_cols.head(3))

# --- 3b: Execute the left join ---
# TODO: Join hdb with mrt_join_cols using a left join on "town"
hdb_with_mrt = hdb.join(
    mrt_join_cols,
    on=____,  # Hint: "town"
    how=____,  # Hint: "left"
)

print(f"\n=== After Left Join with MRT ===")
print(f"Shape before: {hdb.shape}")
print(f"Shape after:  {hdb_with_mrt.shape}")
print(f"Row count preserved: {hdb_with_mrt.height == hdb.height}")

# --- 3c: Check for nulls introduced by the join ---
for col in ["nearest_mrt", "distance_to_mrt_km"]:
    nc = hdb_with_mrt[col].null_count()
    pct = nc / hdb_with_mrt.height
    print(f"  {col}: {nc:,} nulls ({pct:.1%})")

# --- 3d: Inspect matched and unmatched rows ---
matched_rows = hdb_with_mrt.filter(pl.col("nearest_mrt").is_not_null())
unmatched_rows = hdb_with_mrt.filter(pl.col("nearest_mrt").is_null())

print(f"\n  Matched rows:   {matched_rows.height:,}")
print(f"  Unmatched rows: {unmatched_rows.height:,}")
if unmatched_rows.height > 0:
    print(f"  Unmatched towns: {unmatched_rows['town'].unique().to_list()}")

# Show a sample of enriched data
print(f"\n=== Sample Enriched Data ===")
print(
    hdb_with_mrt.select(
        "town", "flat_type", "resale_price", "nearest_mrt", "distance_to_mrt_km"
    ).head(5)
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    hdb_with_mrt.height == hdb.height
), f"Left join changed row count: {hdb.height} -> {hdb_with_mrt.height}"
assert "nearest_mrt" in hdb_with_mrt.columns, "nearest_mrt should be added"
assert (
    "distance_to_mrt_km" in hdb_with_mrt.columns
), "distance_to_mrt_km should be added"
print("\n✓ Checkpoint 3 passed — left join with MRT preserved all rows\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Left join — enrich with school density data
# ══════════════════════════════════════════════════════════════════════
# The schools table has one row per school. We need to aggregate to
# town level before joining — otherwise each HDB transaction would be
# duplicated once per school in its town.

# --- 4a: Aggregate schools to town level ---
# TODO: Group schools by "town" and count school names, alias as "school_count"
school_counts = (
    schools.with_columns(pl.col("town").str.to_uppercase())
    .group_by("town")
    .agg(
        pl.col("school_name").count().alias(____),  # Hint: "school_count"
        pl.col("school_name").first().alias("sample_school"),
    )
)
print(f"=== School Counts by Town ===")
print(f"Shape: {school_counts.shape}")
print(school_counts.sort("school_count", descending=True).head(5))

# --- 4b: Join onto the enriched HDB data ---
hdb_enriched = hdb_with_mrt.join(
    school_counts.select("town", "school_count"),
    on="town",
    how="left",
)

print(f"\n=== After School Join ===")
print(f"Shape: {hdb_enriched.shape}")
print(f"school_count nulls: {hdb_enriched['school_count'].null_count():,}")

# --- 4c: Fill nulls from the join ---
# Towns with no school data get NULL. For analysis, fill with 0
# (meaning "no schools in our dataset" — not necessarily "no schools exist").
# TODO: Fill null values in "school_count" with 0
hdb_enriched = hdb_enriched.with_columns(
    pl.col("school_count").fill_null(____),  # Hint: 0
)
print(f"school_count nulls after fill: {hdb_enriched['school_count'].null_count()}")

# --- 4d: Summary of all enriched columns ---
new_cols = [c for c in hdb_enriched.columns if c not in hdb.columns]
print(f"\n=== New Columns from Joins ({len(new_cols)}) ===")
for col in new_cols:
    nc = hdb_enriched[col].null_count()
    print(f"  {col:<25} nulls={nc:,} ({nc / hdb_enriched.height:.1%})")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert hdb_enriched.height == hdb.height, "Row count should be preserved"
assert "school_count" in hdb_enriched.columns, "school_count should be added"
assert (
    hdb_enriched["school_count"].null_count() == 0
), "school_count should have no nulls after fill"
print("\n✓ Checkpoint 4 passed — school join and null filling complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Inner join vs left join — understand the difference
# ══════════════════════════════════════════════════════════════════════
# Inner join keeps ONLY rows that match in BOTH tables.
# Left join keeps ALL rows from the left table.
# The choice matters — inner join silently drops data!

# --- 5a: Execute an inner join for comparison ---
# TODO: Join hdb with mrt_join_cols using an inner join on "town"
hdb_inner = hdb.join(
    mrt_join_cols,
    on="town",
    how=____,  # Hint: "inner"
)

print(f"=== Left Join vs Inner Join ===")
print(f"  Left join rows:  {hdb_with_mrt.height:,}")
print(f"  Inner join rows: {hdb_inner.height:,}")
print(f"  Rows lost:       {hdb_with_mrt.height - hdb_inner.height:,}")
print(
    f"  Loss rate:       {(hdb_with_mrt.height - hdb_inner.height) / hdb_with_mrt.height:.1%}"
)

# --- 5b: Which towns were dropped by inner join? ---
left_towns = set(hdb_with_mrt["town"].unique().to_list())
inner_towns = set(hdb_inner["town"].unique().to_list())
dropped_towns = left_towns - inner_towns

if dropped_towns:
    print(f"\n  Towns dropped by inner join: {sorted(dropped_towns)}")
    for town in sorted(dropped_towns):
        count = hdb.filter(pl.col("town") == town).height
        print(f"    {town}: {count:,} transactions lost")

# --- 5c: Cross join (cartesian product) — for understanding only ---
# A cross join pairs every row from the left with every row from the right.
# Almost never useful for real analysis, but important to understand why
# join keys matter. We'll demonstrate on tiny samples.
sample_left = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
sample_right = pl.DataFrame({"color": ["red", "blue", "green"]})
cross = sample_left.join(sample_right, how="cross")
print(f"\n=== Cross Join Demo (2 x 3 = 6 rows) ===")
print(cross)
# INTERPRETATION: Cross join produces len(left) * len(right) rows.
# On real data, this would be catastrophic: 500k * 150 = 75 million rows.
# This is why you always need a join key — it restricts which rows pair up.

# --- 5d: When to use which join ---
print(f"\n=== Join Type Decision Guide ===")
print(f"  LEFT:   Keep all rows from the main table. Use when missing data")
print(f"          in the lookup table is expected and you want NULLs.")
print(f"  INNER:  Keep only matched rows. Use when unmatched rows are")
print(f"          invalid and should be excluded from analysis.")
print(f"  OUTER:  Keep all rows from both tables. Use when you need a")
print(f"          complete picture of both datasets. Rare in practice.")
print(f"  CROSS:  Cartesian product. Use for generating all combinations")
print(f"          (e.g., parameter grids). Never for data enrichment.")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert hdb_inner.height <= hdb_with_mrt.height, "Inner join should have <= rows vs left"
assert hdb_inner["nearest_mrt"].null_count() == 0, "Inner join should have no nulls"
assert cross.height == 6, "Cross join of 2x3 should produce 6 rows"
print("\n✓ Checkpoint 5 passed — join type comparison complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Handle nulls from joins — the complete null toolkit
# ══════════════════════════════════════════════════════════════════════

# --- 6a: fill_null() — replace NULLs with a specific value ---
# For distance_to_mrt_km, NULL means "no MRT data". We can fill with
# a large number (e.g., 99) to mean "very far from MRT" — this is a
# modelling choice that should be documented.
# TODO: Fill null distance_to_mrt_km with 99.0, aliased as "distance_to_mrt_filled"
hdb_filled = hdb_enriched.with_columns(
    pl.col("distance_to_mrt_km")
    .fill_null(____)
    .alias("distance_to_mrt_filled"),  # Hint: 99.0
)

# --- 6b: fill_null with strategy — forward fill, mean, etc. ---
# For time-series data, forward_fill() carries the last known value forward.
# For cross-sectional data, filling with the column mean or median is common.
median_distance = hdb_enriched["distance_to_mrt_km"].median()
hdb_filled = hdb_filled.with_columns(
    pl.col("distance_to_mrt_km")
    .fill_null(median_distance)
    .alias("distance_median_filled"),
)

# --- 6c: coalesce — pick the first non-null value ---
# coalesce takes multiple columns and returns the first non-null value.
# Useful when you have primary and fallback data sources.
hdb_filled = hdb_filled.with_columns(
    pl.coalesce(
        pl.col("distance_to_mrt_km"),
        pl.col("distance_to_mrt_filled"),
    ).alias("distance_coalesced"),
)

# --- 6d: drop_nulls — remove rows with any null in specified columns ---
# TODO: Drop rows where "distance_to_mrt_km" or "nearest_mrt" is null
hdb_complete = hdb_enriched.drop_nulls(
    subset=[____, ____]
)  # Hint: "distance_to_mrt_km", "nearest_mrt"
print(f"=== Null Handling Results ===")
print(f"  Original rows:         {hdb_enriched.height:,}")
print(f"  After drop_nulls:      {hdb_complete.height:,}")
print(f"  Rows dropped:          {hdb_enriched.height - hdb_complete.height:,}")

# --- 6e: Null count summary ---
print(f"\n=== Null Counts After Each Strategy ===")
print(f"  {'Column':<30} {'Nulls':>8}")
print(f"  {'─' * 40}")
for col in [
    "distance_to_mrt_km",
    "distance_to_mrt_filled",
    "distance_median_filled",
    "distance_coalesced",
]:
    if col in hdb_filled.columns:
        nc = hdb_filled[col].null_count()
        print(f"  {col:<30} {nc:>8,}")

# --- 6f: is_null flag column ---
# Sometimes you want to keep the NULL but flag it for later analysis
hdb_enriched = hdb_enriched.with_columns(
    pl.col("distance_to_mrt_km").is_null().alias("missing_mrt_data"),
)
missing_count = hdb_enriched.filter(pl.col("missing_mrt_data")).height
print(f"\n  Rows flagged as missing MRT data: {missing_count:,}")

# INTERPRETATION: There's no universal "correct" way to handle NULLs.
# fill_null(99) treats missing MRT data as "very far" — a modelling assumption.
# fill_null(median) treats it as "average" — a different assumption.
# drop_nulls removes the uncertainty but loses data.
# The choice depends on your question and how much data you can afford to lose.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert (
    hdb_filled["distance_to_mrt_filled"].null_count() == 0
), "fill_null(99) should eliminate all nulls"
assert hdb_complete.height <= hdb_enriched.height, "drop_nulls should not add rows"
assert "missing_mrt_data" in hdb_enriched.columns, "missing flag should exist"
print("\n✓ Checkpoint 6 passed — null handling strategies applied correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: if/elif/else — conditional logic for data classification
# ══════════════════════════════════════════════════════════════════════
# Python's if/elif/else is the code-level branching mechanism.
# (pl.when().then() is the Polars expression-level equivalent.)
# You use if/elif/else in functions and control flow.


# TODO: Complete the function body using if/elif/else
def classify_mrt_proximity(distance_km: float | None) -> str:
    """Classify a property's MRT proximity into a market-relevant tier.

    This classification reflects Singapore property market conventions:
    - Within 500m (5-min walk) commands a significant premium
    - Within 1km (10-min walk) is considered "near MRT"
    - Beyond 1km requires bus/car and has limited MRT premium
    """
    if distance_km is None:
        return "unknown"
    elif distance_km <= ____:  # Hint: 0.5
        return "walkable"
    elif distance_km <= ____:  # Hint: 1.0
        return "near"
    elif distance_km <= ____:  # Hint: 2.0
        return "moderate"
    else:
        return "far"


def classify_school_density(count: int) -> str:
    """Classify a town's school density."""
    if count == 0:
        return "no_data"
    elif count <= 3:
        return "low"
    elif count <= 8:
        return "medium"
    else:
        return "high"


def describe_district(
    town: str, median_price: float, mrt_dist: float | None, school_count: int
) -> str:
    """Generate a one-sentence description of a district's characteristics."""
    price_label = "premium" if median_price > 500_000 else "affordable"
    mrt_label = classify_mrt_proximity(mrt_dist)
    school_label = classify_school_density(school_count)

    return (
        f"{town} is a {price_label} district "
        f"({mrt_label} to MRT, {school_label} school density)"
    )


# --- Test the classification functions ---
print("=== Classification Tests ===")
print(f"0.3 km: {classify_mrt_proximity(0.3)}")
print(f"0.8 km: {classify_mrt_proximity(0.8)}")
print(f"1.5 km: {classify_mrt_proximity(1.5)}")
print(f"3.0 km: {classify_mrt_proximity(3.0)}")
print(f"None:   {classify_mrt_proximity(None)}")

# --- Apply classification to the DataFrame ---
hdb_enriched = hdb_enriched.with_columns(
    pl.col("distance_to_mrt_km")
    .map_elements(classify_mrt_proximity, return_dtype=pl.String, skip_nulls=False)
    .alias("mrt_proximity"),
    pl.col("school_count")
    .map_elements(classify_school_density, return_dtype=pl.String)
    .alias("school_density"),
)

# Distribution of MRT proximity categories
prox_counts = (
    hdb_enriched.group_by("mrt_proximity")
    .agg(
        pl.len().alias("count"),
        pl.col("resale_price").median().alias("median_price"),
    )
    .sort("median_price", descending=True)
)
print(f"\n=== MRT Proximity Distribution ===")
for row in prox_counts.iter_rows(named=True):
    pct = row["count"] / hdb_enriched.height * 100
    print(
        f"  {row['mrt_proximity']:<12} {row['count']:>8,} ({pct:5.1f}%)  "
        f"median=S${row['median_price']:>10,.0f}"
    )

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert classify_mrt_proximity(0.3) == "walkable", "0.3km should be walkable"
assert classify_mrt_proximity(None) == "unknown", "None should be unknown"
assert "mrt_proximity" in hdb_enriched.columns, "mrt_proximity should exist"
assert "school_density" in hdb_enriched.columns, "school_density should exist"
print("\n✓ Checkpoint 7 passed — conditional classification functions working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: import and packages — using math and datetime
# ══════════════════════════════════════════════════════════════════════
# Python's standard library has hundreds of modules. You import them
# at the top of your file. Here we demonstrate math and datetime.

# --- 8a: The math module ---
# math provides mathematical functions beyond basic arithmetic
print("=== math module ===")
print(f"  pi:       {math.pi:.10f}")
print(f"  e:        {math.e:.10f}")
print(f"  sqrt(2):  {math.sqrt(2):.6f}")
print(f"  log(100): {math.log(100):.4f}  (natural log)")
print(f"  log10(100): {math.log10(100):.4f}")

# Log transformation — used in data analysis to handle skewed distributions
# Log-transforming prices makes the distribution more symmetric.
sample_prices = [300_000, 450_000, 600_000, 800_000, 1_200_000]
# TODO: Compute log of each price using math.log()
log_prices = [____ for p in sample_prices]  # Hint: math.log(p)
print(f"\nPrices:     {sample_prices}")
print(f"Log prices: {[round(lp, 3) for lp in log_prices]}")
# INTERPRETATION: The log transformation compresses large values and
# stretches small values. The distance between 300k and 600k (2x) is
# the same as between 600k and 1.2M (2x) in log space. This is useful
# because many statistical models assume normal (bell-curve) distributions.

# --- 8b: Apply log transform to the full dataset ---
# log() is only meaningful for valid positive prices. This dataset carries a
# few corrupt rows (e.g. a S$10 "sale") — taking log of those produces an
# extreme left-tail that would make the transform look WORSE than the raw
# skew. Guard with a sane floor so invalid prices become null (and are
# ignored by .skew()), which is the realistic clean-before-transform step.
hdb_enriched = hdb_enriched.with_columns(
    pl.when(pl.col("resale_price") >= 1000)
    .then(pl.col("resale_price").log())
    .alias("log_price"),
    pl.when(pl.col("price_per_sqm") >= 1)
    .then(pl.col("price_per_sqm").log())
    .alias("log_price_sqm"),
)

# Compare skewness before and after log transform
price_skew = hdb_enriched["resale_price"].skew()
log_price_skew = hdb_enriched["log_price"].skew()
print(f"\n=== Skewness Comparison ===")
print(f"  resale_price skewness: {price_skew:.3f}")
print(f"  log_price skewness:    {log_price_skew:.3f}")
print(f"  (closer to 0 = more symmetric)")

# --- 8c: datetime module ---
from datetime import date, timedelta

today = date.today()
one_year_ago = today - timedelta(days=365)
print(f"\n=== datetime module ===")
print(f"  Today:        {today}")
print(f"  One year ago: {one_year_ago}")
print(f"  Days apart:   {(today - one_year_ago).days}")


# --- 8d: Using math for Haversine distance preview ---
# This function computes the great-circle distance between two lat/lng points.
# You'll use this in Exercise 8 for computing trip distances.
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points on Earth.

    Args:
        lat1, lon1: Latitude and longitude of point 1 (in degrees).
        lat2, lon2: Latitude and longitude of point 2 (in degrees).

    Returns:
        Distance in kilometres.
    """
    R = 6371  # Earth radius in km
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


# Test: distance from Bishan MRT to Raffles Place MRT
bishan_lat, bishan_lon = 1.3513, 103.8490
raffles_lat, raffles_lon = 1.2830, 103.8513
dist = haversine_km(bishan_lat, bishan_lon, raffles_lat, raffles_lon)
print(f"\n  Bishan to Raffles Place: {dist:.2f} km (straight line)")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert "log_price" in hdb_enriched.columns, "log_price should exist"
assert abs(log_price_skew) < abs(price_skew), "Log transform should reduce skewness"
assert dist > 0, "Haversine distance should be positive"
assert dist < 20, "Bishan to Raffles should be under 20km"
print("\n✓ Checkpoint 8 passed — math, datetime, and log transforms working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: District-level summary with spatial features
# ══════════════════════════════════════════════════════════════════════
# Now that each transaction row carries spatial context, produce a
# district summary mixing price statistics with location features.

district_summary = (
    hdb_enriched.group_by("town")
    .agg(
        # Volume
        pl.len().alias("total_transactions"),
        # Price statistics
        pl.col("resale_price").median().alias("median_price"),
        pl.col("resale_price").mean().alias("mean_price"),
        pl.col("resale_price").std().alias("std_price"),
        pl.col("resale_price").quantile(0.25).alias("q25_price"),
        pl.col("resale_price").quantile(0.75).alias("q75_price"),
        # Normalised price
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        # Area
        pl.col("floor_area_sqm").median().alias("median_area_sqm"),
        # Spatial features — same value for every row in a town, so .first()
        pl.col("nearest_mrt").first().alias("nearest_mrt"),
        pl.col("distance_to_mrt_km").first().alias("distance_to_mrt_km"),
        pl.col("school_count").first().alias("school_count"),
        pl.col("mrt_proximity").first().alias("mrt_proximity"),
        pl.col("school_density").first().alias("school_density"),
    )
    .sort("median_price", descending=True)
)

# Add derived columns
district_summary = district_summary.with_columns(
    (pl.col("q75_price") - pl.col("q25_price")).alias("iqr_price"),
    (pl.col("std_price") / pl.col("mean_price") * 100).alias("cv_price_pct"),
)

# --- Full district report ---
print(f"\n{'═' * 80}")
print(f"  SINGAPORE HDB DISTRICT SUMMARY — WITH SPATIAL CONTEXT")
print(f"{'═' * 80}")
print(
    f"  {'Town':<20} {'Median':>10} {'PSM':>8} "
    f"{'MRT km':>7} {'Schools':>7} {'MRT Prox':>10} {'Txns':>7}"
)
print(f"  {'─' * 76}")

for row in district_summary.iter_rows(named=True):
    mrt_dist = row["distance_to_mrt_km"]
    mrt_str = f"{mrt_dist:.2f}" if mrt_dist is not None else "N/A"
    print(
        f"  {row['town']:<20} "
        f"S${row['median_price']:>8,.0f} "
        f"S${row['median_price_sqm']:>6,.0f} "
        f"{mrt_str:>7} "
        f"{row['school_count']:>7} "
        f"{row['mrt_proximity']:>10} "
        f"{row['total_transactions']:>7,}"
    )

print(f"{'═' * 80}")

# --- Towns closest to MRT ---
print(f"\n=== Towns Closest to MRT ===")
print(
    district_summary.filter(pl.col("distance_to_mrt_km").is_not_null())
    .sort("distance_to_mrt_km")
    .select("town", "nearest_mrt", "distance_to_mrt_km", "median_price")
    .head(10)
)

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert district_summary.height > 0, "district_summary should have rows"
assert (
    district_summary.height == hdb_enriched["town"].unique().len()
), "One row per town"
assert "iqr_price" in district_summary.columns, "iqr_price should be computed"
assert "mrt_proximity" in district_summary.columns, "mrt_proximity should exist"
print("\n✓ Checkpoint 9 passed — district summary with spatial features built\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Correlation analysis — do amenities predict price?
# ══════════════════════════════════════════════════════════════════════
# Correlation measures the linear relationship between two variables.
# +1 = perfect positive, -1 = perfect negative, 0 = no relationship.

# --- 10a: MRT distance vs price ---
# TODO: Compute Pearson correlation between "distance_to_mrt_km" and "median_price"
corr_mrt_price = district_summary.select(
    pl.corr(____, ____)  # Hint: "distance_to_mrt_km", "median_price"
).item()
print(f"=== Correlation Analysis (District Level) ===")
print(f"  MRT distance vs median price: {corr_mrt_price:.3f}")
if corr_mrt_price < 0:
    print(f"    Negative = closer MRT -> higher price")
else:
    print(f"    Positive = closer MRT -> lower price (unexpected)")

# --- 10b: School count vs price ---
corr_school_price = district_summary.select(
    pl.corr("school_count", "median_price")
).item()
print(f"  School count vs median price: {corr_school_price:.3f}")
if corr_school_price > 0:
    print(f"    Positive = more schools -> higher price")
else:
    print(f"    Negative = more schools -> lower price")

# --- 10c: MRT distance vs price per sqm ---
corr_mrt_psm = district_summary.select(
    pl.corr("distance_to_mrt_km", "median_price_sqm")
).item()
print(f"  MRT distance vs price/sqm:    {corr_mrt_psm:.3f}")

# --- 10d: School density vs volume ---
corr_school_volume = district_summary.select(
    pl.corr("school_count", "total_transactions")
).item()
print(f"  School count vs volume:       {corr_school_volume:.3f}")

# --- 10e: Summary interpretation ---
print(f"\n=== Correlation Summary ===")
correlations = [
    ("MRT distance vs price", corr_mrt_price),
    ("School count vs price", corr_school_price),
    ("MRT distance vs PSM", corr_mrt_psm),
    ("School count vs volume", corr_school_volume),
]

for label, corr in correlations:
    if corr is None:
        strength = "N/A"
    elif abs(corr) < 0.3:
        strength = "weak"
    elif abs(corr) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    direction = "positive" if corr and corr > 0 else "negative"
    print(f"  {label:<30} r={corr:.3f} ({strength} {direction})")

# INTERPRETATION: These correlations are at the district level (not
# transaction level). They reveal structural patterns in Singapore's
# housing geography. A negative MRT-price correlation means that
# districts with closer MRT access tend to have higher prices — but
# this is correlation, not causation. Desirable amenities cluster in
# the same places for historical reasons (urban planning, population
# density). Isolating the MRT effect requires controlling for flat
# type, age, and floor level — which you'll learn in M3 regression.

# --- 10f: District description generator ---
print(f"\n=== District Descriptions ===")
for row in district_summary.head(5).iter_rows(named=True):
    desc = describe_district(
        row["town"],
        row["median_price"],
        row["distance_to_mrt_km"],
        row["school_count"],
    )
    print(f"  {desc}")

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert isinstance(corr_mrt_price, float), "Correlation should be a float"
assert -1.0 <= corr_mrt_price <= 1.0, "Correlation must be [-1, 1]"
assert isinstance(corr_school_price, float), "Correlation should be a float"
print("\n✓ Checkpoint 10 passed — correlation analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print(
    """
  ✓ Dataset inspection: checking grain, keys, and nulls before joining
  ✓ Join key analysis: set intersection to predict match rates
  ✓ Left join: .join(how="left") preserves all rows from the left table
  ✓ Inner join: .join(how="inner") keeps only matched rows
  ✓ Pre-join aggregation: aggregate the right table before joining
  ✓ Null handling: fill_null(), coalesce(), drop_nulls(), is_null()
  ✓ if/elif/else: conditional logic for classification functions
  ✓ import: using math (sqrt, log, pi) and datetime modules
  ✓ Haversine formula: computing distances from coordinates
  ✓ Log transforms: reducing skewness for better statistical analysis
  ✓ .first() in agg(): extracting town-level values from transactions
  ✓ Pearson correlation: measuring linear relationships between columns
  ✓ District profiling: combining price + spatial + volume statistics

  NEXT: In Exercise 5, you'll move into time-series analysis with
  window functions. You'll compute rolling averages and year-over-year
  price changes — without leaving the DataFrame — using rolling_mean()
  and shift() with .over() partitioning. You'll also see lazy evaluation
  (scan_csv / collect) for the first time as a performance tool.
"""
)
