# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 2.4: MLE Failure Modes
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Diagnose MLE failure case 1: small n — biased, high-variance sigma
#   - Diagnose MLE failure case 2: multimodal data — mean between modes
#   - Diagnose MLE failure case 3: misspecified likelihood — tail risk
#   - Know when MLE is trustworthy and when alternatives are needed
#   - Connect each failure to its remedy (Bessel, GMM, model selection)
#
# PREREQUISITES: 03_map_estimation.py (MAP as a remedy for small n)
# ESTIMATED TIME: ~35 minutes
#
# TASKS (5-phase R10):
#   1. Theory — three ways MLE can mislead
#   2. Build — simulations for each failure mode
#   3. Train — quantify bias, bimodality, and tail underestimation
#   4. Visualise — bimodal histogram + tail comparison table
#   5. Apply — MAS stress testing: why Normal underestimates crises
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from kailash_ml import ModelVisualizer
from scipy import stats

from shared.mlfp02.ex_2 import (
    DEFAULT_SEED,
    OUTPUT_DIR,
    extract_series,
    fit_normal_mle,
    fit_student_t_mle,
    load_singapore_econ,
    save_figure,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Three MLE Failure Modes
# ════════════════════════════════════════════════════════════════════════
#
# FAILURE 1 — SMALL n: MLE sigma biased low by (n-1)/n. CIs too narrow.
# FAILURE 2 — MULTIMODAL: single Normal puts mean in the "valley".
# FAILURE 3 — MISSPECIFIED: Normal underestimates heavy tails.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Small n Simulation
# ════════════════════════════════════════════════════════════════════════

econ = load_singapore_econ()
gdp_growth = extract_series(econ, "gdp_growth_pct")
n_gdp = len(gdp_growth)

rng = np.random.default_rng(seed=DEFAULT_SEED)

print(f"\n=== MLE Failure Case 1: Small n ===")

small_ns = [3, 5, 10]
n_trials_per = 2000

for n_s in small_ns:
    mle_mus = []
    mle_sigmas = []
    for _ in range(n_trials_per):
        # TODO: Draw a random subsample of size n_s from gdp_growth
        # (without replacement), then compute the sample mean and
        # sample std (ddof=0).
        # Hint: rng.choice(gdp_growth, size=n_s, replace=False)
        sample = ____
        mle_mus.append(____)
        mle_sigmas.append(____)

    mu_bias = np.mean(mle_mus) - gdp_growth.mean()
    sigma_bias = np.mean(mle_sigmas) - gdp_growth.std(ddof=0)
    mu_var = np.var(mle_mus)
    sigma_var = np.var(mle_sigmas)

    print(f"\nn={n_s}: (over {n_trials_per} random subsets)")
    print(f"  mu_hat: bias={mu_bias:+.4f}%, variance={mu_var:.4f}")
    print(f"  sigma_hat: bias={sigma_bias:+.4f}%, variance={sigma_var:.4f}")
    print(f"  sigma_hat negative bias confirms MLE underestimates sigma for small n")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
mle_sigmas_3 = [
    rng.choice(gdp_growth, size=3, replace=False).std(ddof=0) for _ in range(1000)
]
assert np.mean(mle_sigmas_3) < gdp_growth.std(
    ddof=0
), "MLE sigma_hat should be biased low for n=3"
print("\n--- Checkpoint 1 passed --- small-n bias demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3a — TRAIN: Multimodal Data Failure
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== MLE Failure Case 2: Multimodal Data ===")

# Simulate two economic regimes: pre-COVID vs COVID
pre_covid = rng.normal(loc=4.0, scale=1.2, size=30)
covid_shock = rng.normal(loc=-5.0, scale=3.0, size=10)
bimodal_data = np.concatenate([pre_covid, covid_shock])

# TODO: Compute the MLE mean and sigma of the bimodal data.
# Hint: bimodal_data.mean() and bimodal_data.std(ddof=0)
bimodal_mle_mu = ____
bimodal_mle_sigma = ____

# Bimodality coefficient
bimodality_coeff = (stats.skew(bimodal_data) ** 2 + 1) / (
    stats.kurtosis(bimodal_data, fisher=True)
    + 3
    * (
        (len(bimodal_data) - 1) ** 2
        / ((len(bimodal_data) - 2) * (len(bimodal_data) - 3))
    )
)

print(f"Bimodal data: {len(bimodal_data)} observations (two economic regimes)")
print(f"Mode 1 (pre-COVID): mean=4.0%, n=30")
print(f"Mode 2 (COVID shock): mean=-5.0%, n=10")
print(f"\nSingle Normal MLE: mu={bimodal_mle_mu:.2f}%, sigma={bimodal_mle_sigma:.2f}%")
print(f"Bimodality coefficient: {bimodality_coeff:.3f} (>0.555 -> bimodal)")
print(f"Problem: MLE estimate {bimodal_mle_mu:.2f}% lies BETWEEN the two modes!")
print(f"  -> No actual observation is near the MLE mean")
print(f"  -> The model assigns low probability to BOTH clusters of real data")
print(f"\nRemedy: Gaussian Mixture Model (GMM) or regime-switching model")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (
    bimodal_mle_mu > -5.0 and bimodal_mle_mu < 4.0
), "MLE mean should fall between the two modes"
print("\n--- Checkpoint 2 passed --- multimodal failure demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3b — TRAIN: Misspecified Likelihood
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== MLE Failure Case 3: Misspecified Likelihood ===")

rng_t = np.random.default_rng(seed=77)
shock_data = rng_t.standard_t(df=3, size=100) * 2.0 + 2.5

# Fit Normal MLE
normal_mle_mu = shock_data.mean()
normal_mle_sigma = shock_data.std(ddof=0)

# TODO: Fit a Student-t distribution to the shock data using
# fit_student_t_mle(). It returns a dict with keys: df, mu, scale.
# Hint: fit_student_t_mle(shock_data)
t_result = ____
t_df = t_result["df"]
t_mu = t_result["mu"]
t_scale = t_result["scale"]

# Compare tail risk at different percentiles
print(
    f"{'Percentile':<12} {'Normal':>10} {'t-dist':>10} {'Empirical':>10} {'Normal Error':>12}"
)
print("-" * 60)
for pctile in [90, 95, 99, 99.5]:
    normal_q = stats.norm.ppf(pctile / 100, loc=normal_mle_mu, scale=normal_mle_sigma)
    t_q = stats.t.ppf(pctile / 100, df=t_df, loc=t_mu, scale=t_scale)
    emp_q = np.percentile(shock_data, pctile)
    error = abs(normal_q - emp_q)
    print(
        f"{pctile:>10}th  {normal_q:>10.2f} {t_q:>10.2f} {emp_q:>10.2f} {error:>12.2f}"
    )

print(f"\nt-distribution df = {t_df:.1f} (lower -> heavier tails)")
print(f"Normal systematically underestimates extreme events")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
normal_99 = stats.norm.ppf(0.99, loc=normal_mle_mu, scale=normal_mle_sigma)
t_99 = stats.t.ppf(0.99, df=t_df, loc=t_mu, scale=t_scale)
assert t_99 > normal_99, "t-distribution 99th percentile should exceed Normal's"
print("\n--- Checkpoint 3 passed --- misspecification and tail risk demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Bimodal Failure
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

bimodal_df = pl.DataFrame({"gdp_growth_pct": bimodal_data})
fig = viz.histogram(
    bimodal_df,
    column="gdp_growth_pct",
    bins=40,
    title="Bimodal GDP Growth (Pre-COVID + COVID Shock)",
)
fig.add_vline(x=bimodal_mle_mu, line_dash="dash", annotation_text="MLE mean")
save_figure(fig, "ex2_04_bimodal_failure.html")
print("Saved: ex2_04_bimodal_failure.html")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n--- Checkpoint 4 passed --- bimodal failure visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS Stress Testing — Why Normal Underestimates Crises
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== APPLY: MAS Stress Testing ===")

threshold = -5.0
# TODO: Compute the probability of GDP growth < threshold under
# Normal and t-distribution models.
# Hint: stats.norm.cdf(threshold, loc=..., scale=...)
#       stats.t.cdf(threshold, df=..., loc=..., scale=...)
prob_normal = ____
prob_t = ____
empirical_fraction = np.mean(shock_data < threshold)

print(f"Stress threshold: GDP growth < {threshold}%")
print(f"Normal model probability:  {prob_normal:.6f} ({prob_normal*100:.4f}%)")
print(f"t-distribution probability: {prob_t:.6f} ({prob_t*100:.4f}%)")
print(
    f"Empirical frequency:       {empirical_fraction:.4f} ({empirical_fraction*100:.2f}%)"
)
print(
    f"\nNormal UNDERESTIMATES crisis probability by {prob_t/max(prob_normal, 1e-12):.1f}x"
)
print(
    f"If MAS sets capital requirements using the Normal model, banks"
    f"\nwill hold insufficient reserves for tail events."
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert prob_t > prob_normal, "t-dist assigns more probability to tail events"
print("\n--- Checkpoint 5 passed --- MAS stress testing application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - MLE failure 1: small n -> biased sigma (Bessel/MAP remedies)
  - MLE failure 2: multimodal -> mean between modes (GMM remedy)
  - MLE failure 3: misspecification -> wrong tails (model selection)
  - Always visualise before fitting — histograms reveal bimodality
  - Heavy-tailed distributions (Student-t) for financial risk
  - Real-world impact: stress testing with correct tail probabilities

  NEXT: In 05_model_selection.py, you'll use AIC/BIC to formally
  compare distribution families and bootstrap for non-standard
  statistics (median, trimmed mean, IQR).
"""
)

print("--- Exercise 2.4 complete --- MLE Failure Modes")
