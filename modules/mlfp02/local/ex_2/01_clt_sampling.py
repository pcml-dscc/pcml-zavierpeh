# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 2.1: Central Limit Theorem and Sampling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Profile distribution shape before fitting parametric models
#     (Shapiro-Wilk, skewness, excess kurtosis)
#   - Distinguish population parameters from sample statistics and
#     explain why Bessel's correction (ddof=1) fixes variance bias
#   - Simulate the Central Limit Theorem: observe how the sampling
#     distribution of the mean becomes Normal regardless of population
#   - Visualise how CLT convergence depends on sample size and shape
#
# PREREQUISITES: MLFP02 Exercise 1 (probability, Bayesian thinking)
# ESTIMATED TIME: ~35 minutes
#
# TASKS (5-phase R10):
#   1. Theory — population vs sample, CLT statement
#   2. Build — three synthetic populations (Exponential, Uniform, Bimodal)
#   3. Train — run CLT simulations at different sample sizes
#   4. Visualise — CLT sampling distributions and Bessel's correction
#   5. Apply — MAS quarterly GDP volatility reporting
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from shared.mlfp02.ex_2 import (
    DEFAULT_N_CLT_REPS,
    DEFAULT_SEED,
    OUTPUT_DIR,
    extract_series,
    load_singapore_econ,
    save_figure,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Population vs Sample, and the CLT
# ════════════════════════════════════════════════════════════════════════
#
# POPULATION PARAMETERS vs SAMPLE STATISTICS:
#   Population: mu (true mean), sigma (true std) — fixed but unknown.
#   Sample:     x-bar (sample mean), s (sample std) — random.
#
# KEY INSIGHT: MLE variance (ddof=0) underestimates the population
# variance because the sample mean "uses up" one degree of freedom.
# Bessel's correction divides by (n-1) to give an unbiased estimator.
#
# CENTRAL LIMIT THEOREM:
#   Regardless of the population distribution, the sampling distribution
#   of x-bar approaches N(mu, sigma^2/n) as n -> infinity.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Data Profiling: Distribution Shape Assessment
# ════════════════════════════════════════════════════════════════════════

econ = load_singapore_econ()
gdp_growth = extract_series(econ, "gdp_growth_pct")
inflation = extract_series(econ, "inflation_rate")
unemployment = extract_series(econ, "unemployment_rate")

n_gdp = len(gdp_growth)

print("=" * 70)
print("  MLFP02 Exercise 2.1: CLT and Sampling")
print("=" * 70)
print(f"\n  GDP growth: {n_gdp} observations")
print(f"  Range: {gdp_growth.min():.2f}% to {gdp_growth.max():.2f}%")
print(f"  Sample mean: {gdp_growth.mean():.3f}%")
print(f"  Sample std:  {gdp_growth.std():.3f}%")

# TODO: Run the Shapiro-Wilk normality test on gdp_growth.
# Hint: stats.shapiro(gdp_growth) returns (statistic, p_value)
shapiro_stat, shapiro_p = ____

print(f"\n=== Distribution Shape Assessment ===")
print(f"Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("Cannot reject normality — Normal likelihood is plausible")
else:
    print("Normality rejected — consider heavier-tailed distributions (t, skew-Normal)")

# TODO: Compute the skewness and excess kurtosis of gdp_growth.
# Hint: stats.skew(data) and stats.kurtosis(data)
skew = ____
kurt = ____
print(f"Skewness: {skew:.3f} (|skew|>1 suggests non-Normal)")
print(f"Excess kurtosis: {kurt:.3f} (>0 -> heavier tails than Normal)")

# Descriptive statistics for all three series
for name, arr in [
    ("GDP growth", gdp_growth),
    ("Inflation", inflation),
    ("Unemployment", unemployment),
]:
    print(
        f"\n{name}: mean={arr.mean():.3f}, std={arr.std():.3f}, "
        f"min={arr.min():.3f}, max={arr.max():.3f}, n={len(arr)}"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 <= shapiro_p <= 1, "Shapiro-Wilk p-value must be between 0 and 1"
assert n_gdp > 10, f"Expected substantial GDP history, got {n_gdp} observations"
print("\n--- Checkpoint 1 passed --- data profiled and normality assessed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Bessel's Correction Simulation
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Population vs Sample Variance ===")

rng = np.random.default_rng(seed=DEFAULT_SEED)
true_mu = 3.0
true_sigma = 5.0
n_sim_repeats = DEFAULT_N_CLT_REPS
sample_size_demo = 10

mle_vars = []
unbiased_vars = []

for _ in range(n_sim_repeats):
    sample = rng.normal(true_mu, true_sigma, size=sample_size_demo)
    # TODO: Compute the MLE variance (ddof=0) and unbiased variance (ddof=1)
    # Hint: sample.var(ddof=0) and sample.var(ddof=1)
    mle_vars.append(____)
    unbiased_vars.append(____)

mle_var_mean = np.mean(mle_vars)
unbiased_var_mean = np.mean(unbiased_vars)
true_var = true_sigma**2

print(f"True sigma^2 = {true_var:.2f}")
print(f"n per sample: {sample_size_demo}")
print(f"Simulations: {n_sim_repeats:,}")
print(
    f"E[MLE var] (ddof=0):     {mle_var_mean:.2f} (bias = {mle_var_mean - true_var:+.2f})"
)
print(
    f"E[unbiased s^2] (ddof=1): {unbiased_var_mean:.2f} (bias = {unbiased_var_mean - true_var:+.2f})"
)
print(
    f"Ratio MLE/true: {mle_var_mean / true_var:.4f} (expected: {(sample_size_demo-1)/sample_size_demo:.4f})"
)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert mle_var_mean < true_var, "MLE variance should underestimate on average"
assert (
    abs(unbiased_var_mean - true_var) < true_var * 0.05
), "Unbiased variance should be within 5% of true variance"
print("\n--- Checkpoint 2 passed --- Bessel's correction demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Central Limit Theorem Simulation
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Central Limit Theorem Simulation ===")

n_clt_samples = DEFAULT_N_CLT_REPS
sample_sizes_clt = [5, 15, 30, 100]

# TODO: Create three synthetic populations:
# 1. Exponential(scale=1.0) with 100,000 samples
# 2. Uniform(0, 10) with 100,000 samples
# 3. Bimodal: concatenate N(2, 0.5) with 50K and N(8, 0.5) with 50K
# Hint: rng.exponential(), rng.uniform(), rng.normal(), np.concatenate()
population_exp = ____
population_unif = ____
pop_bimodal = np.concatenate(
    [
        ____,
        ____,
    ]
)

populations = {
    "Exponential(1)": population_exp,
    "Uniform(0,10)": population_unif,
    "Bimodal N(2,0.5)+N(8,0.5)": pop_bimodal,
}

for pop_name, pop in populations.items():
    print(f"\n--- {pop_name} ---")
    print(
        f"Population: mean={pop.mean():.3f}, std={pop.std():.3f}, skew={stats.skew(pop):.3f}"
    )

    for n_s in sample_sizes_clt:
        # TODO: Draw n_clt_samples sets of size n_s (with replacement),
        # compute the mean of each set, store in sample_means.
        # Hint: rng.choice(pop, size=n_s, replace=True).mean()
        sample_means = np.array([____ for _ in range(n_clt_samples)])
        sm_mean = sample_means.mean()
        sm_std = sample_means.std()
        sm_skew = stats.skew(sample_means)
        theoretical_se = pop.std() / np.sqrt(n_s)

        sw_stat, sw_p = stats.shapiro(
            rng.choice(sample_means, size=min(500, n_clt_samples), replace=False)
        )

        print(
            f"  n={n_s:>3}: mean(x-bar)={sm_mean:.3f}, "
            f"SE(x-bar)={sm_std:.3f} (theory: {theoretical_se:.3f}), "
            f"skew={sm_skew:.3f}, Shapiro p={sw_p:.3f}"
        )

# ── Checkpoint 3 ─────────────────────────────────────────────────────
exp_means_100 = np.array(
    [rng.choice(population_exp, size=100, replace=True).mean() for _ in range(2000)]
)
_, sw_p_100 = stats.shapiro(rng.choice(exp_means_100, size=200, replace=False))
assert (
    sw_p_100 > 0.01
), f"CLT: sampling distribution of mean should be ~Normal for n=100, Shapiro p={sw_p_100}"
print("\n--- Checkpoint 3 passed --- CLT demonstrated across three population shapes\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: CLT Convergence
# ════════════════════════════════════════════════════════════════════════

fig1 = make_subplots(rows=1, cols=3, subplot_titles=["n=5", "n=30", "n=100"])
for i, ns in enumerate([5, 30, 100]):
    sample_means = [rng.choice(population_exp, size=ns).mean() for _ in range(3000)]
    fig1.add_trace(
        go.Histogram(
            x=sample_means, nbinsx=40, name=f"n={ns}", opacity=0.7, showlegend=True
        ),
        row=1,
        col=i + 1,
    )
fig1.update_layout(
    title="CLT: Sampling Distribution of Mean from Exponential(1)", height=350
)
save_figure(fig1, "ex2_01_clt_demonstration.html")
print("Saved: ex2_01_clt_demonstration.html")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n--- Checkpoint 4 passed --- CLT visualisation saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS Quarterly GDP Volatility Reporting
# ════════════════════════════════════════════════════════════════════════
# MAS publishes quarterly GDP statistics. The standard error determines
# how precisely we know the average growth rate. Bessel correction
# matters with only ~40-60 quarterly observations.

print(f"\n=== APPLY: MAS GDP Volatility Reporting ===")

# TODO: Compute the MLE sigma (ddof=0) and unbiased sigma (ddof=1)
# for the real GDP growth data. Then compute SE = sigma / sqrt(n).
# Hint: gdp_growth.std(ddof=0) and gdp_growth.std(ddof=1)
sigma_mle = ____
sigma_unbiased = ____
se_mle = sigma_mle / np.sqrt(n_gdp)
se_unbiased = sigma_unbiased / np.sqrt(n_gdp)

print(f"GDP observations: {n_gdp} quarters")
print(f"MLE sigma: {sigma_mle:.3f}%   ->  SE(mean) = {se_mle:.3f}%")
print(f"Unbiased sigma: {sigma_unbiased:.3f}%  ->  SE(mean) = {se_unbiased:.3f}%")
print(
    f"Difference: {(sigma_unbiased - sigma_mle):.4f}% "
    f"({(sigma_unbiased - sigma_mle) / sigma_mle * 100:.2f}% wider)"
)
print(
    f"\nFor MAS fan chart: 95% CI for mean growth = "
    f"[{gdp_growth.mean() - 1.96 * se_unbiased:.2f}%, "
    f"{gdp_growth.mean() + 1.96 * se_unbiased:.2f}%]"
)
print("Using the biased MLE would produce a CI that is too narrow,")
print("giving policymakers false confidence in the growth forecast.")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert sigma_unbiased > sigma_mle, "Unbiased sigma must exceed MLE sigma"
print("\n--- Checkpoint 5 passed --- MAS application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - Population vs sample: parameter (mu, sigma) vs statistic (x-bar, s)
  - Bessel's correction: MLE variance is biased low by factor (n-1)/n
  - Central Limit Theorem: x-bar -> Normal regardless of population
    shape, demonstrated for Exponential, Uniform, and Bimodal
  - Shapiro-Wilk test to assess normality before fitting models
  - Real-world impact: MAS GDP reporting with correct volatility bands

  NEXT: In 02_mle_fisher.py, you'll write and optimise a log-likelihood
  function using scipy.optimize, then compute standard errors from the
  Fisher information matrix and compare Wald vs profile likelihood CIs.
"""
)

print("--- Exercise 2.1 complete --- CLT and Sampling")
