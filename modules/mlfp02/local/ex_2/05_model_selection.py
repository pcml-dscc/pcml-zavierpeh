# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 2.5: Model Selection and Bootstrap
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Use AIC/BIC to compare distribution families (Normal, Student-t,
#     Skew-Normal, Laplace) on the same dataset
#   - Interpret AIC vs BIC disagreement (complexity-accuracy trade-off)
#   - Implement bootstrap for non-standard statistics (median, trimmed
#     mean, IQR) where Fisher information has no formula
#   - Compare bootstrap SE to theoretical SE for the mean (validation)
#   - Visualise bootstrap distributions and confidence intervals
#
# PREREQUISITES: 04_mle_failures.py (when MLE goes wrong)
# ESTIMATED TIME: ~35 minutes
#
# TASKS (5-phase R10):
#   1. Theory — AIC/BIC derivation, bootstrap principle
#   2. Build — fit four distribution families, compute AIC/BIC
#   3. Train — bootstrap for median, trimmed mean, IQR
#   4. Visualise — bootstrap distributions and CI comparison
#   5. Apply — OCBC portfolio risk: which tail model to use?
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from shared.mlfp02.ex_2 import (
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    OUTPUT_DIR,
    aic,
    bic,
    bootstrap_percentile_ci,
    bootstrap_statistic,
    extract_series,
    fit_normal_mle,
    fit_student_t_mle,
    load_singapore_econ,
    save_figure,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Model Selection and Bootstrap
# ════════════════════════════════════════════════════════════════════════
#
# AIC = 2k - 2 l(theta_hat)   (k = number of parameters, lower is better)
# BIC = k*log(n) - 2 l(theta_hat)  (penalises more for large n)
#
# When AIC and BIC agree, the evidence is strong. When they disagree,
# prefer BIC for prediction, AIC for explanation.
#
# BOOTSTRAP: resample with replacement, compute statistic, repeat.
# Works for ANY statistic — no analytical formulas needed.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: AIC/BIC Model Comparison
# ════════════════════════════════════════════════════════════════════════

econ = load_singapore_econ()
gdp_growth = extract_series(econ, "gdp_growth_pct")
n_gdp = len(gdp_growth)
rng = np.random.default_rng(seed=DEFAULT_SEED)

print(f"\n=== AIC/BIC Model Comparison ===")

fits: dict[str, dict] = {}

# TODO: Fit Normal MLE and get log-likelihood.
# Hint: fit_normal_mle(gdp_growth) returns dict with "loglik" key.
normal_result = ____
ll_normal = normal_result["loglik"]
fits["Normal"] = {"k": 2, "ll": ll_normal}

# TODO: Fit Student-t MLE and get log-likelihood.
# Hint: fit_student_t_mle(gdp_growth) returns dict with "loglik" key.
t_result = ____
ll_t = t_result["loglik"]
fits["Student-t"] = {"k": 3, "ll": ll_t}

# Skew-Normal (k=3: loc, scale, shape)
sn_params = stats.skewnorm.fit(gdp_growth)
ll_sn = float(np.sum(stats.skewnorm.logpdf(gdp_growth, *sn_params)))
fits["Skew-Normal"] = {"k": 3, "ll": ll_sn}

# Laplace (k=2: loc, scale)
lap_loc, lap_scale = stats.laplace.fit(gdp_growth)
ll_lap = float(np.sum(stats.laplace.logpdf(gdp_growth, loc=lap_loc, scale=lap_scale)))
fits["Laplace"] = {"k": 2, "ll": ll_lap}

print(f"{'Distribution':<15} {'k':>3} {'Log-lik':>12} {'AIC':>10} {'BIC':>10}")
print("-" * 55)
for name, f in fits.items():
    # TODO: Compute AIC and BIC using the shared helpers.
    # Hint: aic(k, loglik) and bic(k, loglik, n)
    f["aic"] = ____
    f["bic"] = ____
    print(
        f"{name:<15} {f['k']:>3} {f['ll']:>12.2f} {f['aic']:>10.2f} {f['bic']:>10.2f}"
    )

best_aic = min(fits.items(), key=lambda x: x[1]["aic"])
best_bic = min(fits.items(), key=lambda x: x[1]["bic"])
print(f"\nBest by AIC: {best_aic[0]} (AIC={best_aic[1]['aic']:.2f})")
print(f"Best by BIC: {best_bic[0]} (BIC={best_bic[1]['bic']:.2f})")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert all(
    f["aic"] is not None for f in fits.values()
), "All AIC values must be computed"
print("\n--- Checkpoint 1 passed --- AIC/BIC model selection completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Bootstrap for Non-Standard Statistics
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Bootstrap for Non-Standard Statistics ===")

# TODO: Bootstrap the median using bootstrap_statistic().
# Hint: bootstrap_statistic(data, statistic_func, n_boot=...) returns
# an array of bootstrapped values. np.median is the statistic function.
boot_medians = ____
median_ci = bootstrap_percentile_ci(boot_medians)

# TODO: Bootstrap the 10% trimmed mean.
# Hint: use lambda x: float(stats.trim_mean(x, 0.1)) as the statistic
boot_trimmed = ____
trimmed_ci = bootstrap_percentile_ci(boot_trimmed)

# Bootstrap the IQR
boot_iqrs = bootstrap_statistic(
    gdp_growth,
    lambda x: float(np.subtract(*np.percentile(x, [75, 25]))),
    n_boot=DEFAULT_N_BOOT,
)
iqr_ci = bootstrap_percentile_ci(boot_iqrs)

# Bootstrap the mean for reference
boot_means = bootstrap_statistic(
    gdp_growth,
    lambda x: float(x.mean()),
    n_boot=DEFAULT_N_BOOT,
)
mean_ci = bootstrap_percentile_ci(boot_means)

print(f"{'Statistic':<18} {'Estimate':>10} {'Boot SE':>10} {'95% CI':>25}")
print("-" * 65)
for name, est, boots in [
    ("Mean", float(gdp_growth.mean()), boot_means),
    ("Median", float(np.median(gdp_growth)), boot_medians),
    ("10% Trimmed Mean", float(stats.trim_mean(gdp_growth, 0.1)), boot_trimmed),
    ("IQR", float(np.subtract(*np.percentile(gdp_growth, [75, 25]))), boot_iqrs),
]:
    ci = bootstrap_percentile_ci(boots)
    print(
        f"{name:<18} {est:>10.4f} {boots.std():>10.4f} [{ci[0]:>10.4f}, {ci[1]:>10.4f}]"
    )

# Effect of sample size on bootstrap SE
print(f"\n--- Bootstrap SE by Sample Size ---")
for n_sub in [10, 20, 50, n_gdp]:
    sub = gdp_growth[: min(n_sub, n_gdp)]
    boot_m = bootstrap_statistic(
        sub, lambda x: float(x.mean()), n_boot=5000, seed=DEFAULT_SEED
    )
    print(
        f"  n={n_sub:>3}: Boot SE(mean) = {boot_m.std():.4f}, "
        f"Theory SE = {sub.std(ddof=1)/np.sqrt(len(sub)):.4f}"
    )

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (
    median_ci[0] < np.median(gdp_growth) < median_ci[1]
), "Median must be within its bootstrap CI"
assert boot_medians.std() > 0, "Bootstrap SE of median must be positive"
print("\n--- Checkpoint 2 passed --- bootstrap for non-standard statistics\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Bootstrap Distributions
# ════════════════════════════════════════════════════════════════════════

fig = make_subplots(
    rows=1, cols=2, subplot_titles=["Bootstrap Means", "Bootstrap Medians"]
)
fig.add_trace(go.Histogram(x=boot_means, nbinsx=50, name="Means"), row=1, col=1)
fig.add_trace(go.Histogram(x=boot_medians, nbinsx=50, name="Medians"), row=1, col=2)
fig.update_layout(title="Bootstrap Distributions: Mean vs Median", height=350)
save_figure(fig, "ex2_05_bootstrap_comparison.html")
print("Saved: ex2_05_bootstrap_comparison.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n--- Checkpoint 3 passed --- bootstrap visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: OCBC Portfolio Risk — Which Tail Model?
# ════════════════════════════════════════════════════════════════════════
# OCBC manages a fixed-income portfolio. VaR at 99% confidence
# determines the capital reserve. Wrong distribution = wrong capital.

print(f"\n=== APPLY: OCBC Portfolio Risk — Distribution Choice ===")

mu_normal = normal_result["mu"]
sigma_normal = normal_result["sigma"]
t_df_fit = t_result["df"]
t_mu_fit = t_result["mu"]
t_scale_fit = t_result["scale"]

portfolio_value = 10_000  # SGD millions
print(f"Portfolio value: SGD {portfolio_value:,}M")
print(
    f"\n{'Confidence':>12} {'VaR (Normal)':>15} {'VaR (t-dist)':>15} {'Shortfall':>12}"
)
print("-" * 60)
for alpha in [0.95, 0.99, 0.995]:
    # TODO: Compute VaR under Normal and t-distribution models.
    # VaR = negative of the (1-alpha) quantile.
    # Hint: stats.norm.ppf(1-alpha, loc=..., scale=...)
    #       stats.t.ppf(1-alpha, df=..., loc=..., scale=...)
    var_normal = ____
    var_t = ____
    shortfall = (var_t - var_normal) * portfolio_value / 100
    print(
        f"{alpha*100:>10.1f}%  {var_normal:>12.3f}%  {var_t:>12.3f}%  "
        f"SGD {shortfall:>7.0f}M"
    )

if best_aic[0] == best_bic[0]:
    print(f"\nAIC and BIC agree: use {best_aic[0]} for portfolio risk.")
else:
    print(f"\nAIC prefers {best_aic[0]}, BIC prefers {best_bic[0]}.")
    print("For risk management (tail events matter), prefer the heavier-tailed model.")

print(
    "\nBottom line: choosing the wrong distribution model can leave"
    "\nhundreds of millions in unprovisioned tail risk."
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n--- Checkpoint 4 passed --- OCBC portfolio risk application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - AIC/BIC: penalised likelihood for distribution comparison
  - When AIC and BIC agree, the evidence is strong
  - Bootstrap for any statistic: median, trimmed mean, IQR
  - Bootstrap SE matches theoretical SE for the mean (validation)
  - Bootstrap is the ONLY practical option for non-standard statistics
  - Real-world impact: VaR capital reserves under different models

  SUMMARY — Exercise 2 Complete:
    2.1  CLT and Sampling — why x-bar is Normal regardless of population
    2.2  MLE and Fisher — optimise log-likelihood, standard errors, CIs
    2.3  MAP Estimation — MLE + prior = Bayesian regularisation
    2.4  MLE Failures — small n, multimodal, misspecification
    2.5  Model Selection — AIC/BIC + bootstrap for robust inference

  NEXT: In Exercise 3, you'll move from estimation to decision-making.
  You'll formulate null and alternative hypotheses, run power analysis,
  apply multiple testing corrections (Bonferroni and BH-FDR),
  implement a permutation test, and simulate false discovery rates
  — all on A/B test data.
"""
)

print("--- Exercise 2.5 complete --- Model Selection and Bootstrap")
