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

    # TODO 1: sale_year  <- first 4 chars of "month" ("YYYY-MM"), as Int.
    # TODO 2: storey_midpoint  <- parse "LO TO HI"; some digits are OCR'd as the
    #         letter "O" (e.g. "4O TO 42"). Fix only the numeric tokens, then
    #         average the two bounds. (Careful: "TO" itself contains an O.)
    # TODO 3: flat_age_years  <- sale_year - lease_commence_date.
    # TODO 4: price_per_sqm   <- resale_price / floor_area_sqm.
    # TODO 5: flat_type_rooms <- ordinal map (2/3/4/5 ROOM -> 2..5,
    #         EXECUTIVE -> 6, MULTI-GENERATION -> 7).
    # TODO 6: remaining_lease_years <- parse BOTH "X years Y months" and bare
    #         "X"; impute nulls as 99 - flat_age_years (statutory 99y lease).
    # TODO 7: select the 10 columns in order, sort by [sale_year, town].

    return df  # <- replace with your 10-column engineered frame


if __name__ == "__main__":
    print(solve().head())
