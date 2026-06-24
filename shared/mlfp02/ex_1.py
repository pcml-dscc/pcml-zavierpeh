# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP02 Exercise 1 — Probability and Bayesian
Fundamentals.

Contains: HDB data loading (4-ROOM 2020+ slice), prior/posterior math for
the Normal-Normal and Beta-Binomial conjugate families, bootstrap helpers,
and output directory wiring.

Technique-specific narrative (MLE derivation, Bayes scenarios, credible vs
confidence simulation, bootstrap comparison) does NOT belong here — it
lives in the per-technique files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
from scipy import stats

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

# Output directory for plots (HTML) — every technique writes here.
OUTPUT_DIR = Path("outputs") / "mlfp02_ex1_bayes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical prior used across all four technique files. A single source
# of truth means Task 5's prior sweep and Task 4's posterior use the same
# starting point.
PRIOR_MU_0: float = 500_000.0  # SGD — Singapore 4-room HDB market anchor
PRIOR_SIGMA_0: float = 100_000.0  # SGD — moderate uncertainty

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — HDB resale flats (Singapore, data.gov.sg)
# ════════════════════════════════════════════════════════════════════════

_loader = MLFPDataLoader()


def load_hdb_all() -> pl.DataFrame:
    """Full HDB resale dataset filtered to 2020+ transactions.

    Returns a polars DataFrame with columns: month, town, flat_type,
    resale_price, plus any others present in the source parquet. Used
    by Task 1 (truth tables) and Task 8 (expected value by flat type).
    """
    hdb = _loader.load("mlfp01", "hdb_resale.parquet")
    return hdb.filter(pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1))


def load_hdb_4room() -> pl.DataFrame:
    """4-ROOM slice of HDB resale (2020+) — the primary estimation target."""
    return load_hdb_all().filter(pl.col("flat_type") == "4 ROOM")


def load_hdb_prices_4room() -> np.ndarray:
    """Return the 4-room resale_price column as a float64 numpy array.

    This is the primary observation vector for MLE, Normal-Normal, and
    bootstrap tasks.
    """
    return load_hdb_4room()["resale_price"].to_numpy().astype(np.float64)


# ════════════════════════════════════════════════════════════════════════
# NORMAL-NORMAL CONJUGATE
# ════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NormalPosterior:
    """Posterior for μ under a Normal-Normal conjugate with known σ."""

    mean: float
    std: float
    prior_mean: float
    prior_std: float
    n: int
    sigma_known: float

    @property
    def precision_prior(self) -> float:
        return 1.0 / self.prior_std**2

    @property
    def precision_data(self) -> float:
        return self.n / self.sigma_known**2

    @property
    def prior_weight(self) -> float:
        """Fraction of posterior precision contributed by the prior (0..1)."""
        return self.precision_prior / (self.precision_prior + self.precision_data)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        z = stats.norm.ppf(0.5 + level / 2)
        return (self.mean - z * self.std, self.mean + z * self.std)


def normal_normal_posterior(
    data: np.ndarray,
    prior_mean: float,
    prior_std: float,
    sigma_known: float,
) -> NormalPosterior:
    """Closed-form posterior for μ under N(μ₀, σ₀²) prior and known σ.

    Posterior precision = prior precision + n / σ². Posterior mean is the
    precision-weighted average of the prior mean and the sample mean.
    """
    n = len(data)
    xbar = float(np.mean(data))
    prec_prior = 1.0 / prior_std**2
    prec_data = n / sigma_known**2
    prec_post = prec_prior + prec_data
    sigma_post_sq = 1.0 / prec_post
    mu_post = sigma_post_sq * (prior_mean * prec_prior + n * xbar / sigma_known**2)
    return NormalPosterior(
        mean=mu_post,
        std=float(np.sqrt(sigma_post_sq)),
        prior_mean=prior_mean,
        prior_std=prior_std,
        n=n,
        sigma_known=sigma_known,
    )


# ════════════════════════════════════════════════════════════════════════
# BETA-BINOMIAL CONJUGATE
# ════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BetaPosterior:
    """Posterior for p under a Beta-Binomial conjugate."""

    alpha: float
    beta: float
    prior_alpha: float
    prior_beta: float
    k: int
    n: int

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def prior_mean(self) -> float:
        return self.prior_alpha / (self.prior_alpha + self.prior_beta)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        lo = (1 - level) / 2
        hi = 1 - lo
        return tuple(stats.beta.ppf([lo, hi], self.alpha, self.beta).tolist())


def beta_binomial_posterior(
    k: int, n: int, prior_alpha: float, prior_beta: float
) -> BetaPosterior:
    """Closed-form posterior for p under Beta(α, β) prior and k/n Binomial."""
    return BetaPosterior(
        alpha=prior_alpha + k,
        beta=prior_beta + (n - k),
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        k=k,
        n=n,
    )


# ════════════════════════════════════════════════════════════════════════
# MLE + CRAMER-RAO
# ════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NormalMLE:
    mean: float
    mle_std: float  # ddof=0, biased
    unbiased_std: float  # ddof=1, Bessel's correction
    n: int

    @property
    def fisher_information(self) -> float:
        return self.n / self.mle_std**2

    @property
    def cramer_rao_bound(self) -> float:
        return 1.0 / self.fisher_information

    @property
    def standard_error(self) -> float:
        return float(np.sqrt(self.cramer_rao_bound))


def normal_mle(data: np.ndarray) -> NormalMLE:
    """MLE for Normal (μ, σ²) with Cramér-Rao bound bookkeeping."""
    arr = np.asarray(data, dtype=np.float64)
    return NormalMLE(
        mean=float(arr.mean()),
        mle_std=float(arr.std(ddof=0)),
        unbiased_std=float(arr.std(ddof=1)),
        n=len(arr),
    )


# ════════════════════════════════════════════════════════════════════════
# BOOTSTRAP INTERVALS
# ════════════════════════════════════════════════════════════════════════


def bootstrap_mean_distribution(
    data: np.ndarray, n_bootstrap: int = 10_000, seed: int = 42
) -> np.ndarray:
    """Non-parametric bootstrap of the sample mean."""
    rng = np.random.default_rng(seed)
    n = len(data)
    return np.array(
        [rng.choice(data, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    )


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo = (1 - level) / 2 * 100
    hi = (1 + level) / 2 * 100
    return float(np.percentile(draws, lo)), float(np.percentile(draws, hi))


def bca_ci(
    data: np.ndarray, n_bootstrap: int = 10_000, seed: int = 42, level: float = 0.95
) -> tuple[float, float]:
    """Bias-corrected accelerated bootstrap CI for the mean (via scipy)."""
    result = stats.bootstrap(
        (np.asarray(data, dtype=np.float64),),
        statistic=np.mean,
        n_resamples=n_bootstrap,
        confidence_level=level,
        method="BCa",
        random_state=seed,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


# ════════════════════════════════════════════════════════════════════════
# FORMATTING
# ════════════════════════════════════════════════════════════════════════


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def print_interval(label: str, lo: float, hi: float) -> None:
    print(
        f"  {label:<28} [{fmt_money(lo)}, {fmt_money(hi)}]  width={fmt_money(hi - lo)}"
    )
