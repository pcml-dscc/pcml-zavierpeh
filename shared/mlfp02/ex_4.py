# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP02 Exercise 4 — A/B Testing & Experiment Design.

Contains: experiment data loading, two-arm extraction, power-analysis
primitives (z-values, required-n helper), random number generator factory,
and a small print helper used across the four technique files.

Technique-specific logic (power curves, SRM detection, Welch's t-test,
adaptive design) lives in the per-technique files — not here.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
from scipy import stats

from shared.data_loader import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# DESIGN CONSTANTS
# ════════════════════════════════════════════════════════════════════════

ALPHA: float = 0.05
POWER_TARGET: float = 0.80
DESIGN_MDE_PCT: float = 2.0  # relative MDE (percent of baseline mean)
SEED: int = 42

# Output directory for plots and comparison tables
OUTPUT_DIR = Path("outputs") / "mlfp02_ex4_experiment_design"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATA CONTAINER
# ════════════════════════════════════════════════════════════════════════


class TwoArmAB(NamedTuple):
    """Two-arm A/B subset pulled from the raw experiment frame."""

    experiment: pl.DataFrame  # full frame (all arms)
    ab_data: pl.DataFrame  # control + treatment_a only
    ctrl_values: np.ndarray  # float64 numpy array
    treat_values: np.ndarray  # float64 numpy array
    n_control: int
    n_treatment: int
    n_total: int


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════


def load_experiment() -> TwoArmAB:
    """Load experiment_data.parquet and extract the two-arm A/B subset.

    Returns a TwoArmAB tuple with the raw frame, the filtered A/B frame,
    and numpy arrays for the control and treatment_a metric_value columns.
    """
    loader = MLFPDataLoader()
    experiment = loader.load("mlfp02", "experiment_data.parquet")

    ab_data = experiment.filter(
        pl.col("experiment_group").is_in(["control", "treatment_a"])
    )
    control = ab_data.filter(pl.col("experiment_group") == "control")
    treatment = ab_data.filter(pl.col("experiment_group") == "treatment_a")

    ctrl_values = control["metric_value"].to_numpy().astype(np.float64)
    treat_values = treatment["metric_value"].to_numpy().astype(np.float64)

    return TwoArmAB(
        experiment=experiment,
        ab_data=ab_data,
        ctrl_values=ctrl_values,
        treat_values=treat_values,
        n_control=control.height,
        n_treatment=treatment.height,
        n_total=ab_data.height,
    )


# ════════════════════════════════════════════════════════════════════════
# POWER-ANALYSIS PRIMITIVES
# ════════════════════════════════════════════════════════════════════════


def z_critical(
    alpha: float = ALPHA, power: float = POWER_TARGET
) -> tuple[float, float]:
    """Return (z_{alpha/2}, z_beta) — the two critical values used in sample-size formulas."""
    return stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)


def required_n_per_group(
    sigma: float,
    mde_absolute: float,
    alpha: float = ALPHA,
    power: float = POWER_TARGET,
) -> int:
    """Two-sample normal-approximation sample size per group.

    n = (z_{alpha/2} + z_beta)^2 * 2 * sigma^2 / delta^2

    Used by both the up-front design phase (Exercise 4.1) and the
    adaptive/sequential re-estimation phase (Exercise 4.4).
    """
    if mde_absolute <= 0:
        raise ValueError("mde_absolute must be positive")
    z_alpha_half, z_beta = z_critical(alpha, power)
    return math.ceil((z_alpha_half + z_beta) ** 2 * 2 * sigma**2 / mde_absolute**2)


def power_at_n(
    n_per_group: int,
    sigma: float,
    mde_absolute: float,
    alpha: float = ALPHA,
) -> float:
    """Compute achieved power for a given per-group sample size.

    Uses the non-central normal approximation:
        power = 1 - Phi(z_{alpha/2} - ncp) + Phi(-z_{alpha/2} - ncp)
    where ncp = delta / (sigma * sqrt(2/n)).
    """
    z_alpha_half, _ = z_critical(alpha)
    se = sigma * np.sqrt(2 / n_per_group)
    ncp = mde_absolute / se
    return float(
        1 - stats.norm.cdf(z_alpha_half - ncp) + stats.norm.cdf(-z_alpha_half - ncp)
    )


def cohens_d(delta: float, sigma: float) -> float:
    """Cohen's d effect size for two-sample means with pooled sigma."""
    return float(delta / sigma)


# ════════════════════════════════════════════════════════════════════════
# RANDOM NUMBER GENERATOR
# ════════════════════════════════════════════════════════════════════════


def make_rng(seed: int = SEED) -> np.random.Generator:
    """Factory for the per-exercise numpy Generator — deterministic across runs."""
    return np.random.default_rng(seed)


# ════════════════════════════════════════════════════════════════════════
# REPORTING HELPERS
# ════════════════════════════════════════════════════════════════════════


def print_banner(title: str) -> None:
    """Print a uniform 70-char banner around a section title."""
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def summarise_arm(name: str, values: np.ndarray) -> None:
    """Print a one-line summary of an experiment arm."""
    print(
        f"  {name:<10}: n={len(values):>6,}  "
        f"mean={values.mean():.4f}  std={values.std(ddof=1):.4f}"
    )
