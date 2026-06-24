# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP02 Exercise 2 — Parameter Estimation and Inference.

Contains: Singapore economic data loading, log-likelihood objectives for
Normal / Student-t / Laplace, profile LR CI helpers, bootstrap utilities,
plot output directory and save helper.

Technique-specific narrative (which scenario, which interpretation) belongs
in the per-technique files in ``modules/mlfp02/solutions/ex_2/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy import stats
from scipy.optimize import minimize

from shared.data_loader import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY
# ════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs") / "mlfp02_ex2_mle_theory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: go.Figure, filename: str) -> Path:
    """Write a Plotly figure to the exercise output directory."""
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    return path


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore economic indicators
# ════════════════════════════════════════════════════════════════════════

SINGAPORE_ECON_DATASET = ("mlfp01", "economic_indicators.csv")


def load_singapore_econ() -> pl.DataFrame:
    """Load Singapore economic indicators (GDP, inflation, unemployment)."""
    loader = MLFPDataLoader()
    return loader.load(*SINGAPORE_ECON_DATASET)


def extract_series(df: pl.DataFrame, column: str) -> np.ndarray:
    """Drop nulls and return a float64 numpy array for a given column."""
    return df[column].drop_nulls().to_numpy().astype(np.float64)


def load_gdp_growth() -> np.ndarray:
    """Convenience: GDP growth (annualised quarterly %) as a 1-D numpy array."""
    return extract_series(load_singapore_econ(), "gdp_growth_pct")


# ════════════════════════════════════════════════════════════════════════
# LIKELIHOOD OBJECTIVES
# ════════════════════════════════════════════════════════════════════════
#
# All objectives are NEGATIVE log-likelihoods so they can be passed
# directly to scipy.optimize.minimize. Location/scale parameters are
# reparameterised where needed (log-sigma) to keep the optimiser in a
# valid region without explicit bounds.


def neg_log_likelihood_normal(params: np.ndarray, x: np.ndarray) -> float:
    """-log L for X ~ N(mu, sigma^2). params = [mu, log_sigma]."""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    if sigma <= 0:
        return np.inf
    return float(-np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma)))


def neg_log_likelihood_t(params: np.ndarray, x: np.ndarray) -> float:
    """-log L for X ~ t(df, mu, scale). params = [df, mu, scale]."""
    df, mu, scale = params
    if df <= 0 or scale <= 0:
        return np.inf
    return float(-np.sum(stats.t.logpdf(x, df=df, loc=mu, scale=scale)))


def fit_normal_mle(x: np.ndarray) -> dict:
    """Fit Normal MLE numerically via L-BFGS-B. Returns mu, sigma, loglik, result."""
    x0 = np.array([float(x.mean()), float(np.log(x.std() + 1e-8))])
    result = minimize(
        neg_log_likelihood_normal,
        x0,
        args=(x,),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return {
        "mu": float(result.x[0]),
        "sigma": float(np.exp(result.x[1])),
        "loglik": float(-result.fun),
        "converged": bool(result.success),
        "result": result,
    }


def fit_student_t_mle(x: np.ndarray) -> dict:
    """Fit Student-t MLE via Nelder-Mead. Returns df, mu, scale, loglik."""
    x0 = np.array([5.0, float(x.mean()), float(x.std() + 1e-8)])
    result = minimize(neg_log_likelihood_t, x0, args=(x,), method="Nelder-Mead")
    return {
        "df": float(result.x[0]),
        "mu": float(result.x[1]),
        "scale": float(result.x[2]),
        "loglik": float(-result.fun),
        "converged": bool(result.success),
        "result": result,
    }


# ════════════════════════════════════════════════════════════════════════
# FISHER INFORMATION + CONFIDENCE INTERVALS
# ════════════════════════════════════════════════════════════════════════
#
# For N(mu, sigma^2):
#   Var(mu_hat)   = sigma^2 / n         =>  SE(mu_hat)  = sigma / sqrt(n)
#   Var(sigma_hat) ~ sigma^2 / (2n)     =>  SE(sigma_hat) = sigma / sqrt(2n)
#
# These come directly from inverting the Fisher information matrix at
# the MLE.


def normal_fisher_standard_errors(sigma: float, n: int) -> tuple[float, float]:
    """Return (SE_mu, SE_sigma) from Fisher information at the Normal MLE."""
    se_mu = sigma / np.sqrt(n)
    se_sigma = sigma / np.sqrt(2 * n)
    return float(se_mu), float(se_sigma)


def wald_ci(estimate: float, se: float, alpha: float = 0.05) -> tuple[float, float]:
    """Symmetric Wald CI: estimate ± z_{1-alpha/2} * SE."""
    z = float(stats.norm.ppf(1 - alpha / 2))
    return (estimate - z * se, estimate + z * se)


def profile_lr_ci_normal_mu(
    x: np.ndarray,
    mu_hat: float,
    sigma_hat: float,
    loglik_at_mle: float,
    alpha: float = 0.05,
    grid_width_in_se: float = 4.0,
    n_grid: int = 500,
) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
    """Profile likelihood 1-alpha CI for the Normal mean.

    The CI is the set of mu where 2*(loglik_at_mle - loglik(mu)) < chi^2_{1-alpha, df=1}.

    Returns (ci, mu_grid, loglik_values) so the caller can plot the profile.
    """
    n = len(x)
    se_mu = sigma_hat / np.sqrt(n)
    threshold = float(stats.chi2.ppf(1 - alpha, df=1)) / 2.0

    mu_grid = np.linspace(
        mu_hat - grid_width_in_se * se_mu,
        mu_hat + grid_width_in_se * se_mu,
        n_grid,
    )
    loglik_values = np.array(
        [-neg_log_likelihood_normal([mu, np.log(sigma_hat)], x) for mu in mu_grid]
    )
    lr_values = loglik_at_mle - loglik_values
    mask = lr_values <= threshold
    if mask.any():
        ci = (float(mu_grid[mask][0]), float(mu_grid[mask][-1]))
    else:
        # Fallback: Wald CI
        ci = wald_ci(mu_hat, se_mu, alpha)
    return ci, mu_grid, loglik_values


# ════════════════════════════════════════════════════════════════════════
# MAP — NORMAL LIKELIHOOD + NORMAL PRIOR ON THE MEAN
# ════════════════════════════════════════════════════════════════════════


def make_map_objective(
    prior_mean: float, prior_std: float
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return a neg_map_objective(params, x) closure for the given Normal prior on mu."""

    def neg_map_objective(params: np.ndarray, x: np.ndarray) -> float:
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        if sigma <= 0:
            return np.inf
        nll = -np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma))
        neg_log_prior = -stats.norm.logpdf(mu, loc=prior_mean, scale=prior_std)
        return float(nll + neg_log_prior)

    return neg_map_objective


def fit_normal_map(x: np.ndarray, prior_mean: float, prior_std: float) -> dict:
    """Fit MAP for Normal likelihood with a Normal(prior_mean, prior_std^2) prior on mu."""
    neg_map = make_map_objective(prior_mean, prior_std)
    x0 = np.array([float(x.mean()), float(np.log(x.std() + 1e-8))])
    result = minimize(
        neg_map,
        x0,
        args=(x,),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return {
        "mu": float(result.x[0]),
        "sigma": float(np.exp(result.x[1])),
        "converged": bool(result.success),
        "result": result,
    }


# ════════════════════════════════════════════════════════════════════════
# BOOTSTRAP UTILITIES
# ════════════════════════════════════════════════════════════════════════


def bootstrap_statistic(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 10_000,
    seed: int | None = 42,
) -> np.ndarray:
    """Nonparametric bootstrap: resample x with replacement, apply statistic."""
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        out[i] = statistic(rng.choice(x, size=n, replace=True))
    return out


def bootstrap_percentile_ci(
    boot_samples: np.ndarray, alpha: float = 0.05
) -> tuple[float, float]:
    lo = float(np.percentile(boot_samples, 100 * alpha / 2))
    hi = float(np.percentile(boot_samples, 100 * (1 - alpha / 2)))
    return lo, hi


# ════════════════════════════════════════════════════════════════════════
# AIC / BIC
# ════════════════════════════════════════════════════════════════════════


def aic(k: int, loglik: float) -> float:
    return 2 * k - 2 * loglik


def bic(k: int, loglik: float, n: int) -> float:
    return k * float(np.log(n)) - 2 * loglik


# ════════════════════════════════════════════════════════════════════════
# DEFAULTS — SAMPLE SIZES, SEEDS, PRIOR VALUES
# ════════════════════════════════════════════════════════════════════════
#
# These constants are referenced by multiple technique files. Change them
# once here and every file picks up the update.

DEFAULT_SEED: int = 42
DEFAULT_N_BOOT: int = 10_000
DEFAULT_N_CLT_REPS: int = 5000

# Singapore prior on quarterly GDP growth — long-run open-economy estimate
# that we use to illustrate MAP shrinkage.
GDP_PRIOR_MEAN: float = 3.5  # %
GDP_PRIOR_STD: float = 1.5  # %
