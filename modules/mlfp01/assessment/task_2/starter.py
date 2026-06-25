# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 2: HDB Feature Engineering

Complete the `solve()` function. Read problem.md for the full specification.
The raw strings are deliberately messy — read the data before you parse it.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def solve() -> pl.DataFrame:
    """Engineer the 10-column feature table from raw HDB resale data.

    See problem.md for the exact columns, the storey_range OCR fix, the
    dual-format remaining_lease parser + null imputation rule, the room
    ordinal map, and the derived features.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "hdb_resale.parquet")

    flat_type_map = {
        "2 ROOM": 2,
        "3 ROOM": 3,
        "4 ROOM": 4,
        "5 ROOM": 5,
        "EXECUTIVE": 6,
        "MULTI-GENERATION": 7,
    }

    sale_year = pl.col("month").str.slice(0, 4).cast(pl.Int64)

    storey_lo = (
        pl.col("storey_range")
        .str.extract(r"^([0-9O]+) TO ([0-9O]+)$", 1)
        .str.replace_all("O", "0")
        .cast(pl.Float64)
    )
    storey_hi = (
        pl.col("storey_range")
        .str.extract(r"^([0-9O]+) TO ([0-9O]+)$", 2)
        .str.replace_all("O", "0")
        .cast(pl.Float64)
    )

    remaining_years = pl.col("remaining_lease").str.extract(r"^(\d+)", 1).cast(pl.Float64)
    remaining_months = (
        pl.col("remaining_lease")
        .str.extract(r"^\d+ years (\d+) months$", 1)
        .cast(pl.Float64)
        .fill_null(0.0)
    )
    remaining_lease_years = (remaining_years + remaining_months / 12.0)

    df = df.with_columns(
        sale_year.alias("sale_year"),
        ((storey_lo + storey_hi) / 2).alias("storey_midpoint"),
        (sale_year - pl.col("lease_commence_date")).alias("flat_age_years"),
        (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
        pl.when(pl.col("flat_type") == "2 ROOM").then(pl.lit(2))
        .when(pl.col("flat_type") == "3 ROOM").then(pl.lit(3))
        .when(pl.col("flat_type") == "4 ROOM").then(pl.lit(4))
        .when(pl.col("flat_type") == "5 ROOM").then(pl.lit(5))
        .when(pl.col("flat_type") == "EXECUTIVE").then(pl.lit(6))
        .when(pl.col("flat_type") == "MULTI-GENERATION").then(pl.lit(7))
        .otherwise(pl.lit(None))
        .cast(pl.Int64)
        .alias("flat_type_rooms"),
        remaining_lease_years.alias("remaining_lease_years"),
    ).with_columns(
        pl.when(pl.col("remaining_lease_years").is_null())
        .then(99 - pl.col("flat_age_years"))
        .otherwise(pl.col("remaining_lease_years"))
        .cast(pl.Float64)
        .alias("remaining_lease_years")
    )

    result = df.select(
        "town",
        "flat_type",
        "flat_type_rooms",
        "sale_year",
        "storey_midpoint",
        "floor_area_sqm",
        "flat_age_years",
        "remaining_lease_years",
        "resale_price",
        "price_per_sqm",
    ).sort(["sale_year", "town"])

    return result


if __name__ == "__main__":
    print(solve().head())
