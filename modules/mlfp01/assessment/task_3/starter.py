# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 3: Window Functions & Price Trends

Complete the `solve()` function. Read problem.md for the full specification.
This task is about correct window partitioning and ordering.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def solve() -> pl.DataFrame:
    """Per-town, per-year HDB price-trend table (7 columns).

    See problem.md for the exact columns and the four window computations
    (YoY % within town, 3-year rolling average within town, rank within year).
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "hdb_resale.parquet")

    # 1. Derive sale_year from "YYYY-MM" month string
    df = df.with_columns(
        sale_year=pl.col("month").str.slice(0, 4).cast(pl.Int64)
    )

    # 2. Aggregate to one row per (town, sale_year): median_price and n_sales
    agg_df = df.group_by("town", "sale_year").agg(
        median_price=pl.col("resale_price").median(),
        n_sales=pl.len()
    ).sort(["town", "sale_year"])

    # 3. YoY pct: 100 * (median - prev_year_median) / prev_year_median
    #    within each town, ordered by year (first year = null)
    agg_df = agg_df.with_columns(
        yoy_pct=(
            100 * (
                pl.col("median_price") 
                - pl.col("median_price").shift(1).over("town")
            ) / pl.col("median_price").shift(1).over("town")
        )
    )

    # 4. Rolling 3-year average of median_price within town
    #    (min_periods=1 for partial windows on years 1-2)
    agg_df = agg_df.with_columns(
        rolling_3yr_avg=pl.col("median_price")
        .rolling_mean(window_size=3, min_periods=1)
        .over("town")
    )

    # 5. Rank of median_price within each year, descending (1=most expensive)
    #    method="min" for tied ranks
    agg_df = agg_df.with_columns(
        price_rank_in_year=pl.col("median_price")
        .rank(method="min", descending=True)
        .over("sale_year")
    )

    # 6. Select 7 columns in exact order, sorted by [town, sale_year]
    result = agg_df.select(
        "town", "sale_year", "n_sales", "median_price", 
        "yoy_pct", "rolling_3yr_avg", "price_rank_in_year"
    ).sort(["town", "sale_year"])

    return result


if __name__ == "__main__":
    print(solve().head())
