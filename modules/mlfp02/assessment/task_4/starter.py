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

    # TODO 1: Base = admissions with admission_id, patient_id, diagnosis,
    #         icu_type, los_days, and feature_timestamp = admit_time parsed to
    #         Datetime (format DT_FMT). Left-join patient age, gender, bmi on
    #         patient_id.
    # TODO 2: Vitals -> group_by admission_id: mean_heart_rate, mean_systolic_bp,
    #         min_spo2, max_temperature, n_vitals = count of rows.
    # TODO 3: Labs -> parse value to Float64 (strict=False so junk like
    #         "HAEMOLYSED"/"<0.1" becomes null); lowercase flag. group_by
    #         admission_id: n_labs = row count, n_abnormal_labs = count where
    #         flag == "abnormal", mean_creatinine = mean parsed value where
    #         test_name == "Creatinine".
    # TODO 4: Medications -> parse leading numeric of dose via regex
    #         r"([0-9]+\.?[0-9]*)" -> Float64 mg. group_by admission_id:
    #         n_distinct_drugs = n_unique(drug_name), n_iv_meds = count where
    #         route == "IV", total_dose_mg = sum of parsed dose.
    # TODO 5: Left-join all three aggregate blocks onto the base.
    # TODO 6: Imputation policy:
    #           - gender null -> "Unknown"
    #           - total_dose_mg null -> 0.0
    #           - n_vitals/n_labs/n_abnormal_labs/n_distinct_drugs/n_iv_meds
    #             null -> 0 (cast to Int64)
    #           - age, bmi, mean_heart_rate, mean_systolic_bp, min_spo2,
    #             max_temperature, mean_creatinine null -> that column's MEDIAN
    #             (computed before filling; cast to Float64)
    # TODO 7: select FEATURE_COLUMNS in order, sort by admission_id.

    return adm  # <- replace with your 19-column feature table


if __name__ == "__main__":
    print(solve().head())
