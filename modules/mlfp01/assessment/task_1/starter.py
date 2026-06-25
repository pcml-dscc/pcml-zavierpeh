# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 1: Taxi Trip Data Forensics

Complete the `solve()` function. Read problem.md for the full specification.
Your submission is auto-graded against strict invariants — every impossible
row, missing null, or wrong column will fail a check.

    python grader.py starter.py     # grade your attempt
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def solve() -> pl.DataFrame:
    """Clean the raw taxi-trip log into a 16-column analysis-ready table.

    See problem.md for the exact column list, parsing rules, plausibility
    filters, payment-normalisation mapping, imputation, dedup rule, and the
    four derived columns. Return the cleaned frame sorted by pickup_datetime.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "sg_taxi_trips.parquet")

    # TODO 1: Parse pickup_datetime and dropoff_datetime to Datetime
    #         (format "%Y-%m-%d %H:%M:%S").
    # TODO 2: Normalise payment_type to exactly {"Card", "Cash", "NETS", "Grab"}
    #         (the raw column has 15 spellings — see problem.md for the mapping).
    # TODO 3: Impute tip_sgd nulls -> 0.0; pickup_zone / dropoff_zone nulls -> "Unknown".
    # TODO 4: Derive trip_duration_min and implied_speed_kmh.
    # TODO 5: Drop physically impossible rows (fare, distance, passengers,
    #         duration, and implied-speed bounds — see problem.md).
    # TODO 6: Deduplicate by trip_id, keeping the highest-fare row.
    # TODO 7: Derive fare_per_km and is_airport.
    # TODO 8: Select the 16 columns in the required order, sort by pickup_datetime.

    # Parse datetimes
    df = df.with_columns(
        [
            pl.col("pickup_datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
            pl.col("dropoff_datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
            # Normalise payment_type to four canonical values (case-insensitive substring match)
            pl.when(pl.col("payment_type").str.to_lowercase().str.contains("grab"))
            .then(pl.lit("Grab"))
            .when(pl.col("payment_type").str.to_lowercase().str.contains("nets"))
            .then(pl.lit("NETS"))
            .when(pl.col("payment_type").str.to_lowercase().str.contains("cash"))
            .then(pl.lit("Cash"))
            .when(
                pl.col("payment_type").str.to_lowercase().str.contains("card")
                | pl.col("payment_type").str.to_lowercase().str.contains("visa")
                | pl.col("payment_type").str.to_lowercase().str.contains("mastercard")
                | pl.col("payment_type").str.to_lowercase().str.contains("credit")
            )
            .then(pl.lit("Card"))
            .otherwise(pl.col("payment_type")).alias("payment_type"),
            # Impute tips and zones
            pl.col("tip_sgd").fill_null(0.0),
            pl.col("pickup_zone").fill_null("Unknown"),
            pl.col("dropoff_zone").fill_null("Unknown"),
        ]
    )

    # Derive duration (minutes) and implied speed (km/h)
    # Use integer nanoseconds epoch subtraction then convert
    df = df.with_columns(
        [
            (
                (pl.col("dropoff_datetime").cast(pl.Int64) - pl.col("pickup_datetime").cast(pl.Int64))
                / 1e6
                / 60.0
            ).alias("trip_duration_min"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("distance_km") * 60.0 / pl.col("trip_duration_min")).alias("implied_speed_kmh")
        ]
    )

    # Drop physically impossible rows per spec
    df = df.filter(
        (pl.col("fare_sgd") > 0)
        & (pl.col("distance_km") > 0)
        & (pl.col("distance_km") <= 100)
        & (pl.col("passengers") >= 1)
        & (pl.col("trip_duration_min") > 0)
        & (pl.col("trip_duration_min") <= 180)
        & (pl.col("implied_speed_kmh") >= 2)
        & (pl.col("implied_speed_kmh") <= 120)
    )

    # Deduplicate by trip_id keeping highest fare_sgd (tie-break: latest dropoff_datetime)
    df = df.sort([pl.col("fare_sgd").reverse(), pl.col("dropoff_datetime").reverse()])
    df = df.unique(subset="trip_id")

    # Derive fare_per_km and is_airport
    df = df.with_columns(
        [
            (pl.col("fare_sgd") / pl.col("distance_km")).alias("fare_per_km"),
            (
                (pl.col("pickup_zone") == "Changi Airport")
                | (pl.col("dropoff_zone") == "Changi Airport")
            ).alias("is_airport"),
        ]
    )

    # Select the 16 columns in the exact order and sort by pickup_datetime
    cols = [
        "trip_id",
        "pickup_datetime",
        "dropoff_datetime",
        "pickup_zone",
        "dropoff_zone",
        "distance_km",
        "fare_sgd",
        "tip_sgd",
        "payment_type",
        "passengers",
        "pickup_latitude",
        "pickup_longitude",
        "trip_duration_min",
        "implied_speed_kmh",
        "fare_per_km",
        "is_airport",
    ]

    result = df.select(cols).sort("pickup_datetime")

    return result


if __name__ == "__main__":
    print(solve().head())
