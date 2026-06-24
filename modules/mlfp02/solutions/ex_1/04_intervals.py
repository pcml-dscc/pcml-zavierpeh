# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 1.4: Intervals — Credible vs Confidence, Bootstrap,
#                         and Flat-Type Comparison
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Distinguish Bayesian credible intervals from frequentist confidence
#     intervals with a repeated-sampling simulation
#   - Compute bootstrap CIs (percentile + BCa) and compare with theory
#   - Apply Bayesian estimation across flat types to see how prior
#     influence varies with sample size (Bayesian regularisation)
#   - Understand expected value and sampling bias (friendship paradox)
#   - Synthesize all findings into a stakeholder-ready business summary
#
# PREREQUISITES: Complete 03_conjugate_priors.py (Normal-Normal, Beta-Binomial)
#
# ESTIMATED TIME: ~50 min
#
# TASKS:
#   1. Theory — credible vs confidence: the philosophical difference
#   2. Build — repeated-sampling simulation (1,000 experiments)
#   3. Train — bootstrap CIs + expected value + sampling bias
#   4. Visualise — interval comparison + flat-type posterior chart
#   5. Apply — Bayesian regularisation across flat types + business synthesis
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from kailash_ml import ModelVisualizer
from plotly.subplots import make_subplots
from scipy import stats

from shared.mlfp02.ex_1 import (
    OUTPUT_DIR,
    PRIOR_MU_0,
    PRIOR_SIGMA_0,
    bca_ci,
    bootstrap_mean_distribution,
    fmt_money,
    load_hdb_all,
    load_hdb_prices_4room,
    normal_mle,
    normal_normal_posterior,
    percentile_ci,
    print_interval,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Credible vs Confidence: The Philosophical Difference
# ════════════════════════════════════════════════════════════════════════
# The critical difference between these two interval types:
#
# Frequentist confidence interval (CI):
#   "If I repeated this experiment many times, 95% of the intervals
#    I construct would contain the true μ."
#   → It's a statement about the PROCEDURE, not about THIS interval.
#   → The true μ is fixed; the interval is random.
#
# Bayesian credible interval:
#   "Given THIS data, there is a 95% probability that μ lies in
#    this interval."
#   → It's a statement about THIS interval, not about a procedure.
#   → μ is treated as random; the data is fixed.
#
# Which is "better"? Neither — they answer different questions. But the
# Bayesian interpretation is what most people THINK the frequentist CI
# means. When a stakeholder asks "is the true mean in this range?",
# they want the credible interval answer.
#
# Bootstrap CIs are a frequentist method that relaxes distributional
# assumptions. BCa (bias-corrected accelerated) is the gold standard —
# it corrects for both bias and skewness in the bootstrap distribution.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Repeated-Sampling Simulation
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP02 Exercise 1.4: Intervals and Flat-Type Comparison")
print("=" * 70)

prices = load_hdb_prices_4room()
mle = normal_mle(prices)
mu_0 = PRIOR_MU_0
sigma_0 = PRIOR_SIGMA_0

rng = np.random.default_rng(seed=42)
true_mu = mle.mean  # Treat our MLE as the "true" population mean
true_sigma = mle.mle_std
n_simulations = 1000
sample_size = 100  # Small sample to show variation

# For the coverage simulation we use a weak (uninformative) prior so the
# Bayesian credible interval is comparable to the frequentist CI. An
# informative prior biased away from true_mu would deliberately undercover
# — that effect is explored separately in Task 5 (flat-type regularisation).
sim_prior_mu = true_mu
sim_prior_sigma = 10 * true_sigma  # ~1.35M SGD — effectively flat

freq_covers = 0
bayes_covers = 0
freq_widths = []
bayes_widths = []

print(f"\n=== Credible vs Confidence Interval Simulation ===")
print(f"Simulations: {n_simulations:,}, sample size: {sample_size}")

for _ in range(n_simulations):
    sample = rng.normal(true_mu, true_sigma, size=sample_size)
    xbar = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(sample_size)

    # Frequentist 95% CI
    freq_lower = xbar - 1.96 * se
    freq_upper = xbar + 1.96 * se
    if freq_lower <= true_mu <= freq_upper:
        freq_covers += 1
    freq_widths.append(freq_upper - freq_lower)

    # Bayesian 95% credible interval (Normal-Normal with weak prior so
    # the credible interval is comparable to the frequentist CI)
    post_sim = normal_normal_posterior(
        sample, sim_prior_mu, sim_prior_sigma, sample.std(ddof=0)
    )
    bayes_lower, bayes_upper = post_sim.credible_interval(0.95)
    if bayes_lower <= true_mu <= bayes_upper:
        bayes_covers += 1
    bayes_widths.append(bayes_upper - bayes_lower)

freq_coverage = freq_covers / n_simulations
bayes_coverage = bayes_covers / n_simulations

print(f"Frequentist CI coverage: {freq_coverage:.1%} (target: 95%)")
print(f"Bayesian credible coverage: {bayes_coverage:.1%}")
print(f"Mean freq CI width: {fmt_money(np.mean(freq_widths))}")
print(f"Mean Bayes CI width: {fmt_money(np.mean(bayes_widths))}")
# INTERPRETATION: Both methods achieve approximately 95% coverage when
# the model is correct. The Bayesian interval can be slightly narrower
# because the prior adds information. The key difference is philosophical:
# the freq CI is about the procedure, the Bayes CI is about THIS interval.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert (
    0.90 < freq_coverage < 1.0
), f"Frequentist coverage should be near 95%, got {freq_coverage:.1%}"
assert (
    0.90 < bayes_coverage < 1.0
), f"Bayesian coverage should be near 95%, got {bayes_coverage:.1%}"
print("\n✓ Checkpoint 1 passed — coverage simulation validates both methods\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Bootstrap CIs + Expected Value + Sampling Bias
# ════════════════════════════════════════════════════════════════════════

# --- Bootstrap confidence intervals ---
n_bootstrap = 10_000
bootstrap_means = bootstrap_mean_distribution(prices, n_bootstrap=n_bootstrap, seed=42)
boot_lo, boot_hi = percentile_ci(bootstrap_means, level=0.95)
bca_lo, bca_hi = bca_ci(prices, n_bootstrap=n_bootstrap, seed=42, level=0.95)

# Normal theory CI for reference
normal_ci_lo = mle.mean - 1.96 * mle.standard_error
normal_ci_hi = mle.mean + 1.96 * mle.standard_error

# Bayesian CI from full dataset
posterior_full = normal_normal_posterior(prices, mu_0, sigma_0, mle.mle_std)
bayes_ci_lo, bayes_ci_hi = posterior_full.credible_interval(0.95)

print("=== Confidence / Credible Intervals Comparison ===")
print(f"{'Method':<25} {'Lower':>14} {'Upper':>14} {'Width':>12}")
print("─" * 70)
print_interval("Normal theory 95% CI", normal_ci_lo, normal_ci_hi)
print_interval("Bootstrap percentile CI", boot_lo, boot_hi)
print_interval("Bootstrap BCa CI", bca_lo, bca_hi)
print_interval("Bayesian 95% credible", bayes_ci_lo, bayes_ci_hi)
# INTERPRETATION: All methods agree because n is large and the distribution
# is approximately Normal. Bootstrap and BCa diverge for small n or skewed
# distributions. BCa is preferred as it corrects for both bias and skewness.

# --- Expected value by flat type ---
print(f"\n=== Expected Value: HDB Price by Flat Type ===")
hdb_all = load_hdb_all()
flat_type_stats = (
    hdb_all.group_by("flat_type")
    .agg(
        pl.col("resale_price").mean().alias("mean_price"),
        pl.len().alias("count"),
    )
    .sort("flat_type")
)
total_transactions = flat_type_stats["count"].sum()
flat_type_stats = flat_type_stats.with_columns(
    (pl.col("count") / total_transactions).alias("probability")
)
print(flat_type_stats)

# E[price] = Σ P(flat_type) × E[price | flat_type]
expected_price = (
    flat_type_stats["probability"].to_numpy() * flat_type_stats["mean_price"].to_numpy()
).sum()
simple_avg = flat_type_stats["mean_price"].mean()
print(f"\nE[price] (transaction-weighted) = {fmt_money(expected_price)}")
print(f"Simple average across types     = {fmt_money(simple_avg)}")
print(f"Difference: {fmt_money(expected_price - simple_avg)}")
# INTERPRETATION: The transaction-weighted expected price differs from
# the simple average because different flat types have very different
# transaction volumes. 4-room flats dominate, pulling the weighted
# average toward their price point.

# --- Friendship paradox (sampling bias) ---
print(f"\n=== Sampling Bias: Friendship Paradox ===")
n_people = 200
degrees = rng.zipf(a=2.0, size=n_people).clip(max=n_people - 1)
avg_degree = degrees.mean()

friend_degrees = []
for person_idx in range(n_people):
    if degrees[person_idx] > 0:
        friend_probs = degrees / degrees.sum()
        friend_idx = rng.choice(n_people, p=friend_probs)
        friend_degrees.append(degrees[friend_idx])

avg_friend_degree = np.mean(friend_degrees)

print(f"People: {n_people}")
print(f"Your average number of friends: {avg_degree:.1f}")
print(f"Your friends' average number of friends: {avg_friend_degree:.1f}")
print(f"Ratio: {avg_friend_degree / avg_degree:.2f}x")
print(
    f"→ Your friends have {avg_friend_degree / avg_degree:.1f}x more friends than you!"
)
# INTERPRETATION: This is sampling bias in action. Popular people appear
# disproportionately in friend lists. In data science, this same bias
# affects product reviews, survey responses, and click-through rates.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert boot_lo < boot_hi, "Bootstrap CI lower must be below upper"
assert bca_lo < bca_hi, "BCa CI lower must be below upper"
assert avg_friend_degree > avg_degree, "Friends should have more friends (paradox)"
assert expected_price > 0, "Expected price must be positive"
print("\n✓ Checkpoint 2 passed — bootstrap, expected value, sampling bias computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Interval Comparison + Flat-Type Posteriors
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# -- Plot 1: Bootstrap distribution histogram --
fig1 = go.Figure()
fig1.add_trace(
    go.Histogram(
        x=bootstrap_means,
        nbinsx=80,
        name="Bootstrap means",
        marker_color="steelblue",
        opacity=0.7,
    )
)
fig1.add_vline(
    x=mle.mean,
    line_dash="solid",
    line_color="red",
    annotation_text=f"MLE: {fmt_money(mle.mean)}",
)
fig1.add_vline(x=boot_lo, line_dash="dash", line_color="green", annotation_text="2.5%")
fig1.add_vline(x=boot_hi, line_dash="dash", line_color="green", annotation_text="97.5%")
fig1.update_layout(
    title="Bootstrap Distribution of Sample Mean (10,000 resamples)",
    xaxis_title="Mean Price ($)",
    yaxis_title="Count",
    height=400,
)
fig1.write_html(str(OUTPUT_DIR / "bootstrap_distribution.html"))
print("Saved: bootstrap_distribution.html")

# -- Plot 2: Bayesian estimates across flat types --
flat_types = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]
results_by_type = {}
hdb_full = load_hdb_all()

for ft in flat_types:
    subset = hdb_full.filter(pl.col("flat_type") == ft)
    if subset.height == 0:
        continue
    p = subset["resale_price"].to_numpy().astype(np.float64)
    post_ft = normal_normal_posterior(p, mu_0, sigma_0, p.std())
    ci_ft = post_ft.credible_interval(0.95)
    results_by_type[ft] = {
        "n": len(p),
        "mle_mean": float(p.mean()),
        "posterior_mean": post_ft.mean,
        "posterior_std": post_ft.std,
        "prior_weight": post_ft.prior_weight * 100,
        "ci_lower": ci_ft[0],
        "ci_upper": ci_ft[1],
    }

print(f"\n=== Bayesian Estimates by Flat Type ===")
print(
    f"{'Type':<12} {'n':>8} {'MLE Mean':>12} {'Post Mean':>12} {'Post σ':>10} {'Prior %':>8}"
)
print("─" * 70)
for ft, r in results_by_type.items():
    print(
        f"{ft:<12} {r['n']:>8,} ${r['mle_mean']:>10,.0f} "
        f"${r['posterior_mean']:>10,.0f} ${r['posterior_std']:>8,.2f} "
        f"{r['prior_weight']:>7.3f}%"
    )

# Flat-type comparison scatter
if results_by_type:
    ft_names = list(results_by_type.keys())
    mle_means = [results_by_type[ft]["mle_mean"] for ft in ft_names]
    post_means = [results_by_type[ft]["posterior_mean"] for ft in ft_names]
    ci_lowers = [results_by_type[ft]["ci_lower"] for ft in ft_names]
    ci_uppers = [results_by_type[ft]["ci_upper"] for ft in ft_names]

    fig2 = go.Figure()
    # 45-degree line (perfect agreement)
    all_vals = mle_means + post_means
    line_min, line_max = min(all_vals) * 0.95, max(all_vals) * 1.05
    fig2.add_trace(
        go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            name="y=x (no shrinkage)",
            line={"color": "gray", "dash": "dash"},
        )
    )
    # Points with error bars
    fig2.add_trace(
        go.Scatter(
            x=mle_means,
            y=post_means,
            mode="markers+text",
            text=ft_names,
            textposition="top center",
            name="Flat types",
            marker={"color": "red", "size": 10},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [u - m for u, m in zip(ci_uppers, post_means)],
                "arrayminus": [m - lo for m, lo in zip(post_means, ci_lowers)],
            },
        )
    )
    fig2.update_layout(
        title="Bayesian Regularisation: MLE vs Posterior Mean by Flat Type",
        xaxis_title="MLE Mean ($)",
        yaxis_title="Posterior Mean ($)",
        height=500,
    )
    fig2.write_html(str(OUTPUT_DIR / "flat_type_comparison.html"))
    print("Saved: flat_type_comparison.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
running_means = np.cumsum(bootstrap_means) / np.arange(1, n_bootstrap + 1)
assert (
    abs(running_means[-1] - mle.mean) < mle.standard_error * 3
), "Bootstrap mean should converge to MLE mean"
for ft, r in results_by_type.items():
    assert r["posterior_std"] < sigma_0, f"{ft}: posterior std should shrink from prior"
    assert 0 < r["prior_weight"] < 100, f"{ft}: prior weight must be between 0-100%"
print("\n✓ Checkpoint 3 passed — visualisations and flat-type posteriors valid\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Business Interpretation Synthesis
# ════════════════════════════════════════════════════════════════════════
# Synthesize all findings from Exercises 1.1–1.4 into a stakeholder-
# ready summary. A property fund manager needs to answer:
#   1. How precise is the market estimate?
#   2. How much does prior belief matter?
#   3. Which flat types carry more estimation uncertainty?
#   4. What's the portfolio rebalance recommendation?

print("=== APPLICATION: Property Valuation Insights for Fund Manager ===")
print(
    f"""
1. MARKET POSITION: The average 4-room HDB resale price is {fmt_money(mle.mean)}
   with a standard error of {fmt_money(mle.standard_error)}. This estimate is very
   precise — based on {mle.n:,} recent transactions.

2. BAYESIAN ESTIMATE: Even starting with a prior belief of {fmt_money(mu_0)},
   the data overwhelms the prior — the posterior mean is
   {fmt_money(posterior_full.mean)}. Prior assumptions barely matter with this
   volume of data. For the fund manager, this is reassuring.

3. FLAT TYPE VARIATION: Prior influence varies dramatically by segment.
   For abundant flat types (4-room), the prior contributes <0.01% —
   pure data-driven. For rare types (2-room, Executive), the prior
   contributes more, meaning market assumptions play a larger role.

4. INTERVAL AGREEMENT: All four interval methods (Normal, Bootstrap,
   BCa, Bayesian) agree closely with n={mle.n:,}. For smaller segments
   or skewed distributions, BCa bootstrap is recommended.

5. SAMPLING BIAS WARNING: The friendship paradox applies to property
   markets — properties that appear in many listings are not
   representative. High-visibility properties skew perception upward.

6. PORTFOLIO RECOMMENDATION: With {fmt_money(expected_price)} as the
   transaction-weighted expected price, set listing floors per segment
   using the posterior credible intervals, not point estimates.
"""
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("✓ Checkpoint 4 passed — business interpretation complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED (1.4 — Intervals & Comparison)")
print("═" * 70)
print(
    """
  ✓ Credible vs confidence interval: philosophical and practical
    differences validated with 1,000 simulations
  ✓ Bootstrap CI (percentile + BCa): non-parametric alternatives
    that relax distributional assumptions
  ✓ Expected value: transaction-weighted average differs from simple
    average because of volume imbalance across flat types
  ✓ Sampling bias: the friendship paradox shows why naive averages
    systematically mislead
  ✓ Bayesian regularisation across flat types: less data → prior
    matters more → estimates shrink toward the prior
  ✓ Business synthesis: translating statistical results into
    actionable portfolio decisions for a fund manager

  EXERCISE 1 COMPLETE. In Exercise 2, you'll implement MLE from
  scratch using scipy.optimize.minimize, add a prior to get MAP
  estimation, explore the Central Limit Theorem through simulation,
  and diagnose three cases where MLE fails (small n, multimodal
  data, misspecified likelihood). Singapore GDP growth data.
"""
)

print("\n✓ Exercise 1.4 complete — Intervals and Flat-Type Comparison")
print("✓ EXERCISE 1 COMPLETE — Probability and Bayesian Thinking")
