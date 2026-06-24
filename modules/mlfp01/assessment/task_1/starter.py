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

    return df  # <- replace with your cleaned, 16-column frame


if __name__ == "__main__":
    print(solve().head())
