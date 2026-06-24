# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP02 Exercise 3 — A/B testing & multiple comparisons.

Contains: experiment data loading, group extraction, SRM check, common constants,
and small statistical helpers reused across the four technique files:

    01_bootstrap_power.py    — bootstrap CIs + MDE + power curves
    02_hypothesis_testing.py — two-proportion z-test + effect sizes
    03_multiple_testing.py   — Bonferroni + BH-FDR + FDR simulation
    04_permutation_test.py   — distribution-free alternative

Technique-specific code (the actual corrections, permutation loops, power
formulas) does NOT belong here — each technique file owns its own logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from shared.data_loader import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════

ALPHA: float = 0.05
POWER_TARGET: float = 0.80
N_BOOTSTRAP: int = 10_000
N_PERMUTATIONS: int = 10_000
RANDOM_SEED: int = 42

OUTPUT_DIR = Path("outputs") / "mlfp02_ex3_ab_testing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore e-commerce A/B test
# ════════════════════════════════════════════════════════════════════════


def load_experiment() -> pl.DataFrame:
    """Load the A/B test fixture used across Exercise 3.

    Columns: user_id, experiment_group, metric_value, pre_metric_value,
             revenue, timestamp, segment, platform, country.

    Derives a binary `converted` flag (metric_value > 0) if missing.
    Groups are binarised into {control, treatment}: anything that isn't
    literally "control" is treated as treatment (variant_c, treatment_a,
    etc.). This keeps the two-group tests simple for M2 pedagogy.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "experiment_data.parquet")

    if "converted" not in df.columns:
        df = df.with_columns(
            (pl.col("metric_value") > 0).cast(pl.Int8).alias("converted")
        )

    df = df.with_columns(
        pl.when(pl.col("experiment_group") == "control")
        .then(pl.lit("control"))
        .otherwise(pl.lit("treatment"))
        .alias("group")
    )
    return df


def split_groups(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (control_df, treatment_df)."""
    control = df.filter(pl.col("group") == "control")
    treatment = df.filter(pl.col("group") == "treatment")
    return control, treatment


def conversion_arrays(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (control_converted, treatment_converted) as float64 arrays."""
    control, treatment = split_groups(df)
    c = control["converted"].to_numpy().astype(np.float64)
    t = treatment["converted"].to_numpy().astype(np.float64)
    return c, t


def revenue_arrays(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (control_revenue, treatment_revenue) as float64 arrays."""
    control, treatment = split_groups(df)
    c = control["revenue"].to_numpy().astype(np.float64)
    t = treatment["revenue"].to_numpy().astype(np.float64)
    return c, t


# ════════════════════════════════════════════════════════════════════════
# SANITY CHECKS — SRM
# ════════════════════════════════════════════════════════════════════════


def srm_check(
    n_control: int, n_treatment: int, expected_ratio: float = 0.5
) -> dict[str, Any]:
    """χ² goodness-of-fit test for Sample Ratio Mismatch.

    Returns dict with chi2, p_value, and a plain-language verdict.
    SRM indicates randomisation bugs, bot traffic, or pipeline issues —
    if p < 0.01 do NOT trust downstream test results.
    """
    n_total = n_control + n_treatment
    expected = np.array([n_total * expected_ratio, n_total * (1 - expected_ratio)])
    observed = np.array([n_control, n_treatment])
    chi2, p = stats.chisquare(observed, f_exp=expected)
    verdict = (
        "SRM DETECTED — investigate randomisation"
        if p < 0.01
        else "OK — sample split consistent"
    )
    return {"chi2": float(chi2), "p_value": float(p), "verdict": verdict}


# ════════════════════════════════════════════════════════════════════════
# SMALL REUSABLE STATS
# ════════════════════════════════════════════════════════════════════════


def two_proportion_ztest(
    p_control: float, p_treatment: float, n_control: int, n_treatment: int
) -> tuple[float, float]:
    """Pooled two-proportion z-test. Returns (z_stat, two_sided_p_value)."""
    p_pool = (p_control * n_control + p_treatment * n_treatment) / (
        n_control + n_treatment
    )
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    z = (p_treatment - p_control) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return float(2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1))))


def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Pooled Cohen's d effect size for two samples."""
    s_pool = np.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2)
    return float((x2.mean() - x1.mean()) / s_pool) if s_pool > 0 else 0.0


def interpret_magnitude(abs_effect: float) -> str:
    """Cohen convention: <0.2 negligible, <0.5 small, <0.8 medium, else large."""
    if abs_effect < 0.2:
        return "negligible"
    if abs_effect < 0.5:
        return "small"
    if abs_effect < 0.8:
        return "medium"
    return "large"


def print_header(title: str) -> None:
    """Consistent banner for each technique file."""
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
