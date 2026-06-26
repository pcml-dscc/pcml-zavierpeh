# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 4: Feature Engineering & Feature Store

Complete the `solve()` function. Read problem.md for the full specification.
You join five raw ICU tables into one admission-level feature table. The event
tables are messy: lab values contain junk strings, doses are like "34.8MG", and
many admissions have no vitals/labs at all. Your output is auto-graded
column-by-column against an independent re-derivation.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader

DT_FMT = "%Y-%m-%d %H:%M:%S"

FEATURE_COLUMNS = [
    "admission_id",
    "feature_timestamp",
    "age",
    "gender",
    "bmi",
    "diagnosis",
    "icu_type",
    "mean_heart_rate",
    "mean_systolic_bp",
    "min_spo2",
    "max_temperature",
    "n_vitals",
    "n_labs",
    "n_abnormal_labs",
    "mean_creatinine",
    "n_distinct_drugs",
    "n_iv_meds",
    "total_dose_mg",
    "los_days",
]


def solve() -> pl.DataFrame:
    """Build the 19-column admission-level feature-store table.

    See problem.md for the exact columns, aggregations, parsing rules, and the
    imputation policy. Return the table sorted ascending by admission_id.
    """
    loader = MLFPDataLoader()
    adm = loader.load("mlfp02", "icu_admissions.parquet")
    pat = loader.load("mlfp02", "icu_patients.parquet")
    vit = loader.load("mlfp02", "icu_vitals.parquet")
    labs = loader.load("mlfp02", "icu_labs.parquet")
    meds = loader.load("mlfp02", "icu_medications.parquet")

    # Base
    base = adm.select([
        pl.col("admission_id"),
        pl.col("patient_id"),
        pl.col("diagnosis"),
        pl.col("icu_type"),
        pl.col("los_days"),
        pl.col("admit_time").str.strptime(pl.Datetime, DT_FMT).alias("feature_timestamp"),
    ])

    base = base.join(pat.select(["patient_id", "age", "gender", "bmi"]), on="patient_id", how="left")

    # Vitals aggregation
    vit_agg = (
        vit.group_by("admission_id")
        .agg([
            pl.col("heart_rate").mean().alias("mean_heart_rate"),
            pl.col("systolic_bp").mean().alias("mean_systolic_bp"),
            pl.col("spo2").min().alias("min_spo2"),
            pl.col("temperature").max().alias("max_temperature"),
            pl.count().alias("n_vitals"),
        ])
    )

    # Labs parsing & aggregation
    labs = labs.with_columns(
        pl.col("value").cast(pl.Float64, strict=False).alias("value_f"),
        pl.col("flag").str.to_lowercase().alias("flag_l"),
    )
    labs_agg = (
        labs.group_by("admission_id")
        .agg([
            pl.count().alias("n_labs"),
            (pl.col("flag_l") == "abnormal").sum().alias("n_abnormal_labs"),
            pl.when(pl.col("test_name") == "Creatinine").then(pl.col("value_f")).mean().alias("mean_creatinine"),
        ])
    )

    # Medications parsing & aggregation
    meds = meds.with_columns(
        pl.col("dose").str.extract(r"([0-9]+\.?[0-9]*)").cast(pl.Float64).alias("dose_mg"),
        pl.col("route").str.to_uppercase().alias("route_u"),
    )
    meds_agg = (
        meds.group_by("admission_id")
        .agg([
            pl.col("drug_name").n_unique().alias("n_distinct_drugs"),
            (pl.col("route_u") == "IV").sum().alias("n_iv_meds"),
            pl.col("dose_mg").sum().alias("total_dose_mg"),
        ])
    )

    merged = base.join(vit_agg, on="admission_id", how="left").join(labs_agg, on="admission_id", how="left").join(meds_agg, on="admission_id", how="left")

    # Compute medians before filling
    med_cols = ["age", "bmi", "mean_heart_rate", "mean_systolic_bp", "min_spo2", "max_temperature", "mean_creatinine"]
    medians = {c: float(merged[c].median()) for c in med_cols}

    # Imputations
    merged = merged.with_columns(
        pl.col("gender").fill_null("Unknown"),
        pl.col("total_dose_mg").fill_null(0.0),
        pl.col("n_vitals").fill_null(0).cast(pl.Int64),
        pl.col("n_labs").fill_null(0).cast(pl.Int64),
        pl.col("n_abnormal_labs").fill_null(0).cast(pl.Int64),
        pl.col("n_distinct_drugs").fill_null(0).cast(pl.Int64),
        pl.col("n_iv_meds").fill_null(0).cast(pl.Int64),
    )

    for c in med_cols:
        merged = merged.with_columns(pl.col(c).fill_null(medians[c]).cast(pl.Float64))

    result = merged.select(FEATURE_COLUMNS).sort("admission_id")
    return result


if __name__ == "__main__":
    print(solve().head())
