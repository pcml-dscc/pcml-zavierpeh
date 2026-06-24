# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 1.3: Conjugate Priors — Normal-Normal and
#                         Beta-Binomial
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement the Normal-Normal conjugate: prior + likelihood → posterior
#     analytically (no MCMC needed)
#   - Understand precision weighting — how data overwhelms the prior
#   - Run prior sensitivity analysis: sweep μ₀ and σ₀ to show robustness
#   - Implement the Beta-Binomial conjugate for proportions
#   - Compare weak vs strong priors and see both converge with large n
#
# PREREQUISITES: Complete 02_bayes_theorem.py (Bayes' theorem intuition)
#
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — conjugate families and why they matter
#   2. Build — Normal-Normal posterior for 4-room HDB prices
#   3. Train — prior sensitivity sweep (μ₀ and σ₀ grids)
#   4. Visualise — prior vs posterior density + sensitivity heatmap
#   5. Apply — Beta-Binomial for HDB transaction success rates
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from shared.mlfp02.ex_1 import (
    OUTPUT_DIR,
    PRIOR_MU_0,
    PRIOR_SIGMA_0,
    beta_binomial_posterior,
    fmt_money,
    load_hdb_prices_4room,
    normal_mle,
    normal_normal_posterior,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Conjugate Families and Why They Matter
# ════════════════════════════════════════════════════════════════════════
# A conjugate prior is a prior distribution that, when combined with a
# specific likelihood, yields a posterior in the SAME family as the prior.
# This means the posterior has a closed-form solution — no sampling needed.
#
# Normal-Normal conjugate:
#   Prior:      μ ~ N(μ₀, σ₀²)          — what we believe before data
#   Likelihood: xᵢ | μ ~ N(μ, σ²)       — how data is generated
#   Posterior:  μ | data ~ N(μₙ, σₙ²)   — updated belief
#
#   The posterior precision = prior precision + data precision:
#     1/σₙ² = 1/σ₀² + n/σ²
#   The posterior mean is a precision-weighted average:
#     μₙ = σₙ² × (μ₀/σ₀² + n×x̄/σ²)
#
# Beta-Binomial conjugate:
#   Prior:      p ~ Beta(α, β)           — belief about a proportion
#   Likelihood: k | p ~ Binomial(n, p)   — count of successes
#   Posterior:  p | data ~ Beta(α+k, β+n-k)
#
# Why this matters: in practice, you rarely have infinite data. Conjugate
# priors let you combine domain knowledge with limited data in a
# principled, computationally free way. No MCMC, no GPU, no tuning.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Normal-Normal Posterior for 4-Room HDB Prices
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP02 Exercise 1.3: Conjugate Priors")
print("=" * 70)

prices = load_hdb_prices_4room()
mle = normal_mle(prices)

# Use the canonical prior from shared module
mu_0 = PRIOR_MU_0
sigma_0 = PRIOR_SIGMA_0

# Compute posterior using the shared helper
posterior = normal_normal_posterior(
    data=prices,
    prior_mean=mu_0,
    prior_std=sigma_0,
    sigma_known=mle.mle_std,
)

print(f"\n=== Normal-Normal Conjugate Posterior ===")
print(f"Prior: μ ~ N(μ₀={fmt_money(mu_0)}, σ₀={fmt_money(sigma_0)})")
print(f"Likelihood: X|μ ~ N(μ, σ²) with σ={fmt_money(mle.mle_std)} (plug-in)")
print(
    f"\nPosterior: μ|data ~ N(μₙ={fmt_money(posterior.mean)}, σₙ={posterior.std:,.2f})"
)
print(f"Prior precision:     {posterior.precision_prior:.2e}")
print(f"Data precision:      {posterior.precision_data:.2e}")
print(
    f"Posterior precision:  {posterior.precision_prior + posterior.precision_data:.2e}"
)
print(
    f"Data-to-prior precision ratio: "
    f"{posterior.precision_data / posterior.precision_prior:.0f}x"
)
print(
    f"  → Posterior is dominated by "
    f"{'data' if posterior.precision_data > posterior.precision_prior else 'prior'}"
)

# 95% credible interval
ci_lo, ci_hi = posterior.credible_interval(0.95)
print(f"\n95% Bayesian credible interval: [{fmt_money(ci_lo)}, {fmt_money(ci_hi)}]")
# INTERPRETATION: A credible interval has a direct probability statement:
# "Given the data, there is a 95% probability that the true mean price
# lies in this range." This is STRONGER than the frequentist CI, which
# says "95% of intervals constructed this way would contain the true mean."

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert posterior.precision_data > 0, "Data precision must be positive"
assert posterior.std < sigma_0, "Posterior std should be narrower than prior"
assert ci_lo < posterior.mean < ci_hi, "Posterior mean must be within its own CI"
assert (
    abs(posterior.mean - mle.mean) < sigma_0
), "Posterior should be near MLE with large n"
print("\n✓ Checkpoint 1 passed — Normal-Normal posterior computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Prior Sensitivity Sweep
# ════════════════════════════════════════════════════════════════════════
# How does the posterior change as we vary the prior?
# Sweep prior mean from $300K to $700K and prior std from $20K to $200K.
# This shows when the prior matters and when data overwhelms it.

print("=== Prior Sensitivity Analysis ===")

# Sweep prior mean
print(f"\n--- Varying prior mean (σ₀ = {fmt_money(sigma_0)} fixed) ---")
print(
    f"{'Prior μ₀':>12} {'Posterior μₙ':>15} {'Shift from MLE':>16} {'Prior Weight':>13}"
)
print("─" * 60)
mu_sweep_values = [300_000, 400_000, 500_000, 600_000, 700_000]
for mu_sweep in mu_sweep_values:
    post_sweep = normal_normal_posterior(prices, mu_sweep, sigma_0, mle.mle_std)
    print(
        f"${mu_sweep:>10,.0f}  ${post_sweep.mean:>13,.0f}  "
        f"{post_sweep.mean - mle.mean:>+14,.0f}  "
        f"{post_sweep.prior_weight * 100:>11.4f}%"
    )

# Sweep prior std
print(f"\n--- Varying prior std (μ₀ = {fmt_money(mu_0)} fixed) ---")
print(f"{'Prior σ₀':>12} {'Posterior μₙ':>15} {'Prior Weight':>13}")
print("─" * 45)
sigma_sweep_values = [20_000, 50_000, 100_000, 200_000, 500_000]
for sigma_sweep in sigma_sweep_values:
    post_sweep = normal_normal_posterior(prices, mu_0, sigma_sweep, mle.mle_std)
    print(
        f"${sigma_sweep:>10,.0f}  ${post_sweep.mean:>13,.0f}  "
        f"{post_sweep.prior_weight * 100:>11.4f}%"
    )
# INTERPRETATION: Even a very opinionated prior (σ₀=$20K) gets overwhelmed
# by the data when n is large. But with small n, the prior choice matters.
# A sensitivity analysis like this should accompany any Bayesian report
# to show stakeholders that conclusions are robust to prior assumptions.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
# Verify that varying the prior mean within ±$200K barely moves posterior
for mu_test in [300_000, 700_000]:
    post_test = normal_normal_posterior(prices, mu_test, sigma_0, mle.mle_std)
    assert (
        abs(post_test.mean - mle.mean) < 5000
    ), "With large n, posterior should be near MLE regardless of prior mean"
print("\n✓ Checkpoint 2 passed — prior sensitivity demonstrates data dominance\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Prior vs Posterior + Sensitivity Heatmap
# ════════════════════════════════════════════════════════════════════════

# -- Plot 1: Prior vs Posterior distributions --
x_prior = np.linspace(mu_0 - 3 * sigma_0, mu_0 + 3 * sigma_0, 500)
prior_pdf = stats.norm.pdf(x_prior, mu_0, sigma_0)

x_post = np.linspace(
    posterior.mean - 5 * posterior.std, posterior.mean + 5 * posterior.std, 500
)
posterior_pdf = stats.norm.pdf(x_post, posterior.mean, posterior.std)

fig1 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Prior Distribution", "Posterior Distribution"],
)
fig1.add_trace(
    go.Scatter(
        x=x_prior,
        y=prior_pdf,
        name=f"Prior N({fmt_money(mu_0)}, {fmt_money(sigma_0)}²)",
        line={"color": "blue"},
    ),
    row=1,
    col=1,
)
fig1.add_trace(
    go.Scatter(
        x=x_post,
        y=posterior_pdf,
        name=f"Posterior N({fmt_money(posterior.mean)}, {posterior.std:,.0f}²)",
        line={"color": "red"},
    ),
    row=1,
    col=2,
)
fig1.update_layout(
    title="Normal-Normal Conjugate: Prior vs Posterior for 4-Room HDB Mean",
    height=400,
)
fig1.write_html(str(OUTPUT_DIR / "prior_vs_posterior.html"))
print("Saved: prior_vs_posterior.html")

# -- Plot 2: Sensitivity heatmap (μ₀ × σ₀ → posterior mean) --
mu_grid = np.linspace(300_000, 700_000, 20)
sigma_grid = np.linspace(20_000, 300_000, 20)
z_sensitivity = np.zeros((len(sigma_grid), len(mu_grid)))

for i, sg in enumerate(sigma_grid):
    for j, mg in enumerate(mu_grid):
        post_g = normal_normal_posterior(prices, mg, sg, mle.mle_std)
        z_sensitivity[i, j] = post_g.mean

fig2 = go.Figure(
    data=go.Heatmap(
        z=z_sensitivity,
        x=mu_grid,
        y=sigma_grid,
        colorscale="RdBu_r",
        colorbar={"title": "Posterior μₙ ($)"},
    )
)
fig2.update_layout(
    title="Prior Sensitivity: Posterior Mean as f(μ₀, σ₀)",
    xaxis_title="Prior Mean μ₀ ($)",
    yaxis_title="Prior Std σ₀ ($)",
    height=500,
)
fig2.write_html(str(OUTPUT_DIR / "sensitivity_heatmap.html"))
print("Saved: sensitivity_heatmap.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert z_sensitivity.shape == (20, 20), "Heatmap should be 20×20"
print("\n✓ Checkpoint 3 passed — visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Beta-Binomial for HDB Transaction Success Rates
# ════════════════════════════════════════════════════════════════════════
# Different conjugate family: Beta prior for a proportion.
# Use case: what fraction of 4-room HDB transactions close above $500K?
#
# Prior: p ~ Beta(α, β) where E[p] = α/(α+β)
# Likelihood: k successes in n trials ~ Binomial(n, p)
# Posterior: p | data ~ Beta(α + k, β + n - k)
#
# Business context: a property fund needs to know the "success rate"
# of 4-room flats closing above $500K to set portfolio thresholds.
# If the rate drops below 40%, the fund reallocates to 3-room inventory.

print("=== Beta-Binomial Conjugate: Transaction Success Rates ===")

threshold = 500_000
k_success = int((prices > threshold).sum())
n_trials = len(prices)
empirical_rate = k_success / n_trials

print(f"Threshold: {fmt_money(threshold)}")
print(
    f"Successes (price > threshold): {k_success:,} / {n_trials:,} = {empirical_rate:.2%}"
)

# Weak prior: Beta(2, 2) — slight preference for 50%
post_weak = beta_binomial_posterior(
    k_success, n_trials, prior_alpha=2.0, prior_beta=2.0
)

# Strong prior: Beta(20, 80) — 20% success rate expected
post_strong = beta_binomial_posterior(
    k_success, n_trials, prior_alpha=20.0, prior_beta=80.0
)

ci_weak = post_weak.credible_interval(0.95)
ci_strong = post_strong.credible_interval(0.95)

print(f"\n--- Weak Prior: Beta(2, 2), E[p]={post_weak.prior_mean:.2f} ---")
print(f"Posterior: Beta({post_weak.alpha:.0f}, {post_weak.beta:.0f})")
print(f"Posterior mean: {post_weak.mean:.4f} ({post_weak.mean:.2%})")
print(f"95% CI: [{ci_weak[0]:.4f}, {ci_weak[1]:.4f}]")

print(f"\n--- Strong Prior: Beta(20, 80), E[p]={post_strong.prior_mean:.2f} ---")
print(f"Posterior: Beta({post_strong.alpha:.0f}, {post_strong.beta:.0f})")
print(f"Posterior mean: {post_strong.mean:.4f} ({post_strong.mean:.2%})")
print(f"95% CI: [{ci_strong[0]:.4f}, {ci_strong[1]:.4f}]")

print(f"\nEmpirical rate: {empirical_rate:.4f}")
print(
    f"Both posteriors converge toward {empirical_rate:.2%} because n={n_trials:,} is large."
)
# INTERPRETATION: The Beta-Binomial is the natural Bayesian model for
# proportions. Even a strong prior (Beta(20,80) suggesting 20%) gets
# overwhelmed by thousands of observations.

# Beta-Binomial visualisation
x_beta = np.linspace(0, 1, 500)
beta_prior_pdf = stats.beta.pdf(x_beta, 2.0, 2.0)
beta_post_pdf = stats.beta.pdf(x_beta, post_weak.alpha, post_weak.beta)
beta_strong_pdf = stats.beta.pdf(x_beta, post_strong.alpha, post_strong.beta)

fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=x_beta,
        y=beta_prior_pdf,
        name="Prior Beta(2,2)",
        line={"color": "blue", "dash": "dash"},
    )
)
fig3.add_trace(
    go.Scatter(
        x=x_beta,
        y=beta_post_pdf,
        name="Posterior (weak prior)",
        line={"color": "red"},
    )
)
fig3.add_trace(
    go.Scatter(
        x=x_beta,
        y=beta_strong_pdf,
        name="Posterior (strong prior)",
        line={"color": "orange", "dash": "dot"},
    )
)
fig3.add_vline(
    x=empirical_rate,
    line_dash="dot",
    annotation_text=f"Empirical: {empirical_rate:.2%}",
)
fig3.update_layout(
    title="Beta-Binomial: P(price > $500K) — Prior vs Posterior",
    xaxis_title="Proportion",
    yaxis_title="Density",
    height=450,
)
fig3.write_html(str(OUTPUT_DIR / "beta_binomial.html"))
print("Saved: beta_binomial.html")

# Dollar impact for property fund
fund_threshold = 0.40
print(f"\n--- Fund Decision: rebalance if rate < {fund_threshold:.0%} ---")
p_below_threshold = stats.beta.cdf(fund_threshold, post_weak.alpha, post_weak.beta)
print(f"P(rate < {fund_threshold:.0%}) = {p_below_threshold:.6f}")
if p_below_threshold < 0.05:
    print("→ Very unlikely the rate is below 40%. Keep 4-room allocation.")
else:
    print("→ Non-trivial probability of rate < 40%. Consider rebalancing.")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert post_weak.alpha == 2.0 + k_success, "Posterior alpha update incorrect"
assert post_weak.beta == 2.0 + (n_trials - k_success), "Posterior beta update incorrect"
assert ci_weak[0] < post_weak.mean < ci_weak[1], "Posterior mean must be within CI"
assert (
    abs(post_weak.mean - empirical_rate) < 0.01
), "With weak prior and large n, posterior mean should be near empirical rate"
print("\n✓ Checkpoint 4 passed — Beta-Binomial conjugate and fund decision complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED (1.3 — Conjugate Priors)")
print("═" * 70)
print(
    """
  ✓ Normal-Normal conjugate: prior + likelihood → closed-form posterior
  ✓ Precision weighting: posterior precision = prior + data precision
  ✓ Prior sensitivity: sweeping μ₀ and σ₀ proves conclusions are
    robust — with large n, prior barely moves the posterior
  ✓ Beta-Binomial conjugate: natural model for proportions, same
    "prior + data → posterior" logic for success rates
  ✓ Weak vs strong priors: both converge to the empirical rate when
    n is large — data overwhelms belief
  ✓ Business framing: property fund threshold decision using the
    posterior CDF — "what is P(rate < 40%)?"

  NEXT: In 04_intervals.py, you'll compare Bayesian credible intervals
  with frequentist confidence intervals, run bootstrap CIs, and apply
  Bayesian estimation across all flat types to see how prior influence
  varies with sample size.
"""
)

print("\n✓ Exercise 1.3 complete — Conjugate Priors")
