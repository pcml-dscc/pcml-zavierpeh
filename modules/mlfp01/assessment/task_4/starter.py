# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 4: Profile, Clean & Integrate with DataExplorer

Complete the `solve()` function. Read problem.md for the full specification.
This task uses the kailash-ml DataExplorer engine to PROVE your cleaning
improved data quality.

    python grader.py starter.py
"""
from __future__ import annotations

import asyncio

import polars as pl

from kailash_ml import DataExplorer
from shared import MLFPDataLoader


def solve() -> dict:
    """Return {"cleaned": pl.DataFrame, "raw_alert_count": int, "clean_alert_count": int}.

    See problem.md for the exact 8-column cleaned schema, the THREE period
    formats you must parse, the comma-stripped integer cast, the median
    imputation rule, and the DataExplorer alert-count requirement.
    """
    raw = MLFPDataLoader().load("mlfp01", "economic_indicators.csv")

    # 1. Keep only quarterly rows
    quarterly = raw.filter(pl.col("period_type") == "quarterly")

    # 2. Parse period into period_year and period_quarter
    # Handle three formats: "Q1 2000", "2001-Q1", "2001-2"
    def parse_period(period_str):
        """Parse period string into (year, quarter)."""
        if period_str is None or period_str == "":
            return None, None
        
        period_str = str(period_str).strip()
        
        # Format 1: "Q1 2000" or "Q1 2015"
        if 'Q' in period_str and ' ' in period_str:
            parts = period_str.split()
            quarter = int(parts[0].replace('Q', ''))
            year = int(parts[1])
            return year, quarter
        
        # Format 2: "2001-Q1" or "2001-Q2"
        if 'Q' in period_str and '-' in period_str:
            parts = period_str.split('-Q')
            year = int(parts[0])
            quarter = int(parts[1])
            return year, quarter
        
        # Format 3: "2001-2" (year-quarter_number)
        if '-' in period_str and 'Q' not in period_str:
            parts = period_str.split('-')
            if len(parts) == 2:
                year = int(parts[0])
                quarter = int(parts[1])
                return year, quarter
        
        return None, None

    # Parse period column using Polars expressions
    quarterly = quarterly.with_columns(
        period_year=pl.col("period").map_elements(lambda x: parse_period(x)[0], return_dtype=pl.Int64),
        period_quarter=pl.col("period").map_elements(lambda x: parse_period(x)[1], return_dtype=pl.Int64),
    )

    # 3. Strip thousands separators from tourist_arrivals and cast to Int64
    quarterly = quarterly.with_columns(
        tourist_arrivals=pl.col("tourist_arrivals")
        .cast(pl.Utf8)
        .str.replace_all(",", "")
        .cast(pl.Int64)
    )

    # 4. Impute inflation_rate and trade_balance_sgd_bn with quarterly median
    inflation_median = quarterly["inflation_rate"].median()
    trade_median = quarterly["trade_balance_sgd_bn"].median()
    
    quarterly = quarterly.with_columns(
        inflation_rate=pl.col("inflation_rate").fill_null(inflation_median),
        trade_balance_sgd_bn=pl.col("trade_balance_sgd_bn").fill_null(trade_median),
    )

    # 5. Select 8 columns in exact order and sort
    cleaned = quarterly.select(
        "period_year", "period_quarter", "gdp_growth_pct", "unemployment_rate",
        "inflation_rate", "trade_balance_sgd_bn", "property_price_index", "tourist_arrivals"
    ).sort(["period_year", "period_quarter"])

    # 6. Profile with DataExplorer
    explorer = DataExplorer()
    
    # Profile raw quarterly slice
    raw_profile = asyncio.run(explorer.profile(quarterly.select(
        [c for c in quarterly.columns if c not in ["period_year", "period_quarter"]]
    )))
    raw_alert_count = len(raw_profile.alerts) if hasattr(raw_profile, 'alerts') else 0
    
    # Profile cleaned frame
    clean_profile = asyncio.run(explorer.profile(cleaned.select(
        [c for c in cleaned.columns if c not in ["period_year", "period_quarter"]]
    )))
    clean_alert_count = len(clean_profile.alerts) if hasattr(clean_profile, 'alerts') else 0

    return {
        "cleaned": cleaned,
        "raw_alert_count": raw_alert_count,
        "clean_alert_count": clean_alert_count,
    }


if __name__ == "__main__":
    print(solve())
