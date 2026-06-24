# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP02 Exercise 7 — CUPED and Causal Inference.

Contains: experiment data loading, SRM check, naive A/B baseline, CUPED
math helpers, Bayesian decision utilities, mSPRT helpers, DiD scenario
simulators, and plotting helpers. Technique-specific narration and
checkpoints live in the per-technique files.

Importable from any cwd after `uv sync`:

    from shared.mlfp02.ex_7 import (
        load_experiment, compute_srm, naive_ab, single_cov_cuped, ...
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

# Output directory for visualisation artifacts
OUTPUT_DIR = Path("outputs") / "ex7_causal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Experiment data with pre-experiment covariates
# ════════════════════════════════════════════════════════════════════════


def load_experiment() -> pl.DataFrame:
    """Load the MLFP02 experiment dataset.

    Columns (required): experiment_group, revenue, pre_metric_value, timestamp
    Optional: metric_value (additional covariate for multi-CUPED)
    """
    loader = MLFPDataLoader()
    return loader.load("mlfp02", "experiment_data.parquet")


def split_groups(experiment: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (control, treatment) sub-frames by experiment_group column."""
    control = experiment.filter(pl.col("experiment_group") == "control")
    treatment = experiment.filter(pl.col("experiment_group") != "control")
    return control, treatment


def get_revenue_arrays(
    control: pl.DataFrame, treatment: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Extract revenue arrays as float64 numpy arrays."""
    y_c = control["revenue"].to_numpy().astype(np.float64)
    y_t = treatment["revenue"].to_numpy().astype(np.float64)
    return y_c, y_t


def get_covariate_arrays(
    control: pl.DataFrame, treatment: pl.DataFrame, column: str = "pre_metric_value"
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a pre-experiment covariate as float64 numpy arrays."""
    x_c = control[column].to_numpy().astype(np.float64)
    x_t = treatment[column].to_numpy().astype(np.float64)
    return x_c, x_t


# ════════════════════════════════════════════════════════════════════════
# SAMPLE RATIO MISMATCH (SRM)
# ════════════════════════════════════════════════════════════════════════


def compute_srm(n_c: int, n_t: int) -> float:
    """Chi-square SRM test. Returns p-value. p < 0.01 indicates SRM."""
    expected = np.array([n_c + n_t] * 2) / 2
    observed = np.array([n_c, n_t])
    _, srm_p = stats.chisquare(observed, f_exp=expected)
    return float(srm_p)


# ════════════════════════════════════════════════════════════════════════
# STANDARD A/B (BASELINE)
# ════════════════════════════════════════════════════════════════════════


def naive_ab(y_c: np.ndarray, y_t: np.ndarray) -> dict[str, float]:
    """Standard Welch-style lift, SE, 95% CI, z, two-sided p-value."""
    n_c, n_t = len(y_c), len(y_t)
    mean_c, mean_t = y_c.mean(), y_t.mean()
    lift = mean_t - mean_c
    se = float(np.sqrt(y_c.var(ddof=1) / n_c + y_t.var(ddof=1) / n_t))
    ci_lo = lift - 1.96 * se
    ci_hi = lift + 1.96 * se
    z = lift / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "mean_c": float(mean_c),
        "mean_t": float(mean_t),
        "lift": float(lift),
        "se": se,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "z": float(z),
        "p_value": float(p),
    }


# ════════════════════════════════════════════════════════════════════════
# CUPED — SINGLE COVARIATE
# ════════════════════════════════════════════════════════════════════════


def single_cov_cuped(
    y_c: np.ndarray, y_t: np.ndarray, x_c: np.ndarray, x_t: np.ndarray
) -> dict[str, Any]:
    """Single-covariate CUPED: Y_adj = Y - theta*(X - E[X]).

    theta = Cov(Y, X) / Var(X). Returns adjusted arrays, point estimate,
    SE, CI, p-value, theta, rho, and variance reduction.
    """
    x_all = np.concatenate([x_c, x_t])
    y_all = np.concatenate([y_c, y_t])
    var_x = np.var(x_all, ddof=1)
    theta = np.cov(y_all, x_all)[0, 1] / var_x if var_x > 0 else 0.0
    rho = np.corrcoef(y_all, x_all)[0, 1]
    x_mean = x_all.mean()

    y_c_adj = y_c - theta * (x_c - x_mean)
    y_t_adj = y_t - theta * (x_t - x_mean)

    n_c, n_t = len(y_c), len(y_t)
    lift_adj = y_t_adj.mean() - y_c_adj.mean()
    se = float(np.sqrt(y_c_adj.var(ddof=1) / n_c + y_t_adj.var(ddof=1) / n_t))
    ci_lo, ci_hi = lift_adj - 1.96 * se, lift_adj + 1.96 * se
    z = lift_adj / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "theta": float(theta),
        "rho": float(rho),
        "y_c_adj": y_c_adj,
        "y_t_adj": y_t_adj,
        "lift": float(lift_adj),
        "se": se,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "z": float(z),
        "p_value": float(p),
        "theoretical_reduction": float(rho**2),
    }


# ════════════════════════════════════════════════════════════════════════
# CUPED — MULTI-COVARIATE
# ════════════════════════════════════════════════════════════════════════


def multi_cov_cuped(
    y_c: np.ndarray,
    y_t: np.ndarray,
    X_c: np.ndarray,
    X_t: np.ndarray,
) -> dict[str, Any]:
    """Multi-covariate CUPED via OLS on centered covariates.

    theta = (X'X)^-1 X'Y — multivariate regression coefficients.
    """
    y_all = np.concatenate([y_c, y_t])
    X_all = np.vstack([X_c, X_t])
    X_mean = X_all.mean(axis=0)
    X_centered = X_all - X_mean
    theta = np.linalg.lstsq(X_centered, y_all - y_all.mean(), rcond=None)[0]

    y_c_adj = y_c - (X_c - X_mean) @ theta
    y_t_adj = y_t - (X_t - X_mean) @ theta

    n_c, n_t = len(y_c), len(y_t)
    lift = y_t_adj.mean() - y_c_adj.mean()
    se = float(np.sqrt(y_c_adj.var(ddof=1) / n_c + y_t_adj.var(ddof=1) / n_t))
    return {
        "theta": theta,
        "y_c_adj": y_c_adj,
        "y_t_adj": y_t_adj,
        "lift": float(lift),
        "se": se,
        "ci_lo": float(lift - 1.96 * se),
        "ci_hi": float(lift + 1.96 * se),
    }


# ════════════════════════════════════════════════════════════════════════
# CUPED — STRATIFIED
# ════════════════════════════════════════════════════════════════════════


def stratify_by_covariate(
    x_c: np.ndarray, x_t: np.ndarray, percentiles: tuple[int, int] = (33, 67)
) -> dict[str, np.ndarray]:
    """Build Low/Medium/High strata masks over concatenated (x_c, x_t)."""
    x_all = np.concatenate([x_c, x_t])
    q_lo, q_hi = np.percentile(x_all, percentiles)
    return {
        "Low spenders": (x_all <= q_lo),
        "Medium spenders": (x_all > q_lo) & (x_all <= q_hi),
        "High spenders": (x_all > q_hi),
    }


def stratified_cuped(
    y_c: np.ndarray,
    y_t: np.ndarray,
    x_c: np.ndarray,
    x_t: np.ndarray,
    strata: dict[str, np.ndarray],
    min_per_cell: int = 30,
) -> dict[str, dict[str, float]]:
    """Apply CUPED within each stratum. Returns {name: {n_ctrl,n_treat,lift,se,p}}."""
    n_c = len(y_c)
    results: dict[str, dict[str, float]] = {}
    for name, mask in strata.items():
        ctrl_mask, treat_mask = mask[:n_c], mask[n_c:]
        y_c_s, y_t_s = y_c[ctrl_mask], y_t[treat_mask]
        x_c_s, x_t_s = x_c[ctrl_mask], x_t[treat_mask]
        if len(y_c_s) < min_per_cell or len(y_t_s) < min_per_cell:
            continue
        x_all_s = np.concatenate([x_c_s, x_t_s])
        y_all_s = np.concatenate([y_c_s, y_t_s])
        var_x = np.var(x_all_s, ddof=1)
        theta_s = np.cov(y_all_s, x_all_s)[0, 1] / var_x if var_x > 0 else 0.0
        x_mean = x_all_s.mean()
        y_c_adj = y_c_s - theta_s * (x_c_s - x_mean)
        y_t_adj = y_t_s - theta_s * (x_t_s - x_mean)
        lift = y_t_adj.mean() - y_c_adj.mean()
        se = float(
            np.sqrt(y_c_adj.var(ddof=1) / len(y_c_s) + y_t_adj.var(ddof=1) / len(y_t_s))
        )
        z = lift / se if se > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        results[name] = {
            "n_ctrl": len(y_c_s),
            "n_treat": len(y_t_s),
            "lift": float(lift),
            "se": se,
            "p_value": float(p),
        }
    return results


# ════════════════════════════════════════════════════════════════════════
# BAYESIAN A/B
# ════════════════════════════════════════════════════════════════════════


def bayesian_decision(
    y_c_adj: np.ndarray,
    y_t_adj: np.ndarray,
    lift: float,
    practical_threshold: float = 1.0,
) -> dict[str, float]:
    """Bayesian posterior using normal approximation on CUPED-adjusted arrays.

    Returns P(treatment > control), P(treatment > control + threshold),
    expected loss (both directions), and credible interval.
    """
    n_c, n_t = len(y_c_adj), len(y_t_adj)
    se_c = y_c_adj.std(ddof=1) / np.sqrt(n_c)
    se_t = y_t_adj.std(ddof=1) / np.sqrt(n_t)
    se_lift = float(np.sqrt(se_c**2 + se_t**2))

    prob_better = float(1 - stats.norm.cdf(0, loc=lift, scale=se_lift))
    prob_practical = float(
        1 - stats.norm.cdf(practical_threshold, loc=lift, scale=se_lift)
    )

    z = -lift / se_lift if se_lift > 0 else 0.0
    exp_loss_treat = float(se_lift * stats.norm.pdf(z) + lift * stats.norm.cdf(z))
    exp_loss_ctrl = float(se_lift * stats.norm.pdf(-z) - lift * stats.norm.cdf(-z))

    return {
        "prob_treatment_better": prob_better,
        "prob_practical": prob_practical,
        "expected_loss_treatment": max(0.0, exp_loss_treat),
        "expected_loss_control": max(0.0, exp_loss_ctrl),
        "se_lift": se_lift,
        "ci_lo": float(lift - 1.96 * se_lift),
        "ci_hi": float(lift + 1.96 * se_lift),
    }


def bayesian_decision_rule(prob_better: float, exp_loss_treat: float) -> str:
    """Simple ship/continue/hold decision rule."""
    if prob_better > 0.95 and exp_loss_treat < 0.50:
        return "SHIP — high confidence + low expected loss"
    if prob_better > 0.80:
        return "CONTINUE — promising but need more data"
    return "HOLD — insufficient evidence"


# ════════════════════════════════════════════════════════════════════════
# SEQUENTIAL TESTING — mSPRT
# ════════════════════════════════════════════════════════════════════════


def msprt_sequential_pvalues(
    experiment: pl.DataFrame,
    tau_sq: float,
    min_per_group: int = 100,
    skip_first_days: int = 3,
) -> list[dict[str, float]]:
    """Walk the experiment day by day, computing fixed and mSPRT p-values.

    tau_sq is the mSPRT hyperparameter — typically set to se_naive**2.
    """
    if experiment["timestamp"].dtype in [pl.Utf8, pl.String]:
        exp_daily = experiment.with_columns(
            pl.col("timestamp")
            .str.to_datetime("%Y-%m-%d %H:%M:%S")
            .dt.date()
            .alias("day")
        )
    else:
        exp_daily = experiment.with_columns(
            pl.col("timestamp").cast(pl.Date).alias("day")
        )

    days = sorted(exp_daily["day"].unique().to_list())
    results: list[dict[str, float]] = []
    for i, day in enumerate(days):
        if i < skip_first_days:
            continue
        cumulative = exp_daily.filter(pl.col("day") <= day)
        c = (
            cumulative.filter(pl.col("experiment_group") == "control")["revenue"]
            .to_numpy()
            .astype(np.float64)
        )
        t = (
            cumulative.filter(pl.col("experiment_group") != "control")["revenue"]
            .to_numpy()
            .astype(np.float64)
        )
        if len(c) < min_per_group or len(t) < min_per_group:
            continue
        diff = t.mean() - c.mean()
        v_n = c.var(ddof=1) / len(c) + t.var(ddof=1) / len(t)
        se = float(np.sqrt(v_n))
        z = diff / se if se > 0 else 0.0
        p_fixed = float(2 * (1 - stats.norm.cdf(abs(z))))
        # mSPRT always-valid p-value
        lambda_n = np.sqrt(v_n / (v_n + tau_sq)) * np.exp(
            tau_sq * z**2 / (2 * (v_n + tau_sq))
        )
        p_seq = float(min(1.0, 1.0 / lambda_n)) if lambda_n > 0 else 1.0
        results.append(
            {
                "day": i + 1,
                "n": int(len(c) + len(t)),
                "lift": float(diff),
                "p_fixed": p_fixed,
                "p_sequential": p_seq,
            }
        )
    return results


# ════════════════════════════════════════════════════════════════════════
# PEEKING PROBLEM SIMULATION
# ════════════════════════════════════════════════════════════════════════


def simulate_peeking(
    n_sims: int = 1000,
    n_per_sim: int = 2000,
    n_checks: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Simulate A/A experiments (zero effect) with and without peeking.

    Returns dict with false-positive rates for: no-peek, fixed-p peeking,
    plus the theoretical peeking inflation for comparison.
    """
    rng = np.random.default_rng(seed=seed)
    false_pos_fixed = 0
    false_pos_no_peek = 0

    for _ in range(n_sims):
        sim_ctrl = rng.normal(50, 10, size=n_per_sim)
        sim_treat = rng.normal(50, 10, size=n_per_sim)  # no real effect

        # no peeking
        z_end = (sim_treat.mean() - sim_ctrl.mean()) / np.sqrt(
            sim_ctrl.var(ddof=1) / n_per_sim + sim_treat.var(ddof=1) / n_per_sim
        )
        if 2 * (1 - stats.norm.cdf(abs(z_end))) < 0.05:
            false_pos_no_peek += 1

        # peeking at n_checks points, fixed p-values
        peeked_sig = False
        for check_n in np.linspace(100, n_per_sim, n_checks, dtype=int):
            sc = sim_ctrl[:check_n]
            st = sim_treat[:check_n]
            se_p = np.sqrt(sc.var(ddof=1) / check_n + st.var(ddof=1) / check_n)
            z_p = (st.mean() - sc.mean()) / se_p if se_p > 0 else 0
            if 2 * (1 - stats.norm.cdf(abs(z_p))) < 0.05:
                peeked_sig = True
                break
        if peeked_sig:
            false_pos_fixed += 1

    return {
        "n_sims": n_sims,
        "n_checks": n_checks,
        "rate_no_peek": false_pos_no_peek / n_sims,
        "rate_fixed_peek": false_pos_fixed / n_sims,
        "theoretical_inflated_rate": 1 - (1 - 0.05) ** n_checks,
    }


# ════════════════════════════════════════════════════════════════════════
# DIFFERENCE-IN-DIFFERENCES — Singapore HDB cooling measures
# ════════════════════════════════════════════════════════════════════════


def simulate_hdb_cooling_measures(
    n_per_cell: int = 500, seed: int = 99
) -> dict[str, np.ndarray]:
    """Simulate Singapore HDB prices around a stamp-duty cooling measure.

    Treatment group: Central area HDB transactions (hit by the policy).
    Control group: Non-Central area transactions (exempt).

    Returns the four cells as arrays: pre_central, post_central,
    pre_noncentral, post_noncentral.
    """
    rng = np.random.default_rng(seed=seed)
    pre_central = rng.normal(550_000, 80_000, size=n_per_cell)
    pre_noncentral = rng.normal(450_000, 70_000, size=n_per_cell)
    # Policy effect: Central drops $20K; both grow $10K baseline.
    post_central = rng.normal(540_000, 85_000, size=n_per_cell)
    post_noncentral = rng.normal(460_000, 72_000, size=n_per_cell)
    return {
        "pre_central": pre_central,
        "post_central": post_central,
        "pre_noncentral": pre_noncentral,
        "post_noncentral": post_noncentral,
    }


def diff_in_diff(cells: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute DiD estimate, SE, CI, z, p from four cell arrays."""
    y_tp = cells["pre_central"].mean()
    y_tq = cells["post_central"].mean()
    y_cp = cells["pre_noncentral"].mean()
    y_cq = cells["post_noncentral"].mean()

    did = (y_tq - y_tp) - (y_cq - y_cp)
    n = len(cells["pre_central"])
    se = float(
        np.sqrt(
            cells["pre_central"].var(ddof=1) / n
            + cells["post_central"].var(ddof=1) / n
            + cells["pre_noncentral"].var(ddof=1) / n
            + cells["post_noncentral"].var(ddof=1) / n
        )
    )
    z = did / se if se > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "y_treat_pre": float(y_tp),
        "y_treat_post": float(y_tq),
        "y_ctrl_pre": float(y_cp),
        "y_ctrl_post": float(y_cq),
        "did_estimate": float(did),
        "se": se,
        "ci_lo": float(did - 1.96 * se),
        "ci_hi": float(did + 1.96 * se),
        "z": float(z),
        "p_value": p,
    }


# ════════════════════════════════════════════════════════════════════════
# PARALLEL TRENDS TEST
# ════════════════════════════════════════════════════════════════════════


def parallel_trends_test(
    n_periods: int = 6,
    n_per_period: int = 200,
    central_base: float = 530_000,
    noncentral_base: float = 430_000,
    growth_per_period: float = 2000,
    seed: int = 99,
) -> dict[str, Any]:
    """Simulate pre-period trends and run a bootstrap test for parallel slopes.

    Returns pre-period means, slopes, slope difference, bootstrap p-value,
    and a pass/fail flag at alpha=0.05.
    """
    rng = np.random.default_rng(seed=seed)
    pre_central, pre_noncentral = [], []
    for t in range(n_periods):
        c = rng.normal(central_base + t * growth_per_period, 80_000, size=n_per_period)
        nc = rng.normal(
            noncentral_base + t * growth_per_period, 70_000, size=n_per_period
        )
        pre_central.append(c.mean())
        pre_noncentral.append(nc.mean())
    time_points = np.arange(n_periods)
    slope_c = float(np.polyfit(time_points, pre_central, 1)[0])
    slope_nc = float(np.polyfit(time_points, pre_noncentral, 1)[0])
    slope_diff = slope_c - slope_nc

    # Bootstrap
    n_boot = 5000
    boot_diffs = []
    for _ in range(n_boot):
        noise_c = rng.normal(0, 1000, size=n_periods)
        noise_nc = rng.normal(0, 1000, size=n_periods)
        s_c = np.polyfit(time_points, np.array(pre_central) + noise_c, 1)[0]
        s_nc = np.polyfit(time_points, np.array(pre_noncentral) + noise_nc, 1)[0]
        boot_diffs.append(s_c - s_nc)
    boot_p = float(np.mean(np.abs(boot_diffs) >= np.abs(slope_diff)))

    return {
        "pre_central": pre_central,
        "pre_noncentral": pre_noncentral,
        "time_points": time_points,
        "slope_central": slope_c,
        "slope_noncentral": slope_nc,
        "slope_diff": float(slope_diff),
        "bootstrap_p": boot_p,
        "passes": boot_p > 0.05,
    }


# ════════════════════════════════════════════════════════════════════════
# VARIANCE REDUCTION REPORTING
# ════════════════════════════════════════════════════════════════════════


def variance_reduction(se_baseline: float, se_adjusted: float) -> dict[str, float]:
    """Report variance and CI-width reduction from baseline -> adjusted SE."""
    var_red = 1 - (se_adjusted**2) / (se_baseline**2)
    ci_red = 1 - (se_adjusted / se_baseline)
    return {
        "variance_reduction": float(var_red),
        "ci_width_reduction": float(ci_red),
        "effective_sample_multiplier": float(1 / max(1 - var_red, 1e-9)),
    }


def print_banner(title: str) -> None:
    """Consistent section header across technique files."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
