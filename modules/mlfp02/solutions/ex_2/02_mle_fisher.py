# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 2.2: MLE and Fisher Information
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Write and optimise a log-likelihood function using scipy.optimize
#   - Reparameterise (log sigma) to enforce positivity without bounds
#   - Derive standard errors from the Fisher information matrix
#   - Construct Wald CIs and profile likelihood CIs
#   - Interpret profile likelihood — invariant to reparameterisation
#
# PREREQUISITES: 01_clt_sampling.py (data profiling, CLT intuition)
# ESTIMATED TIME: ~30 minutes
#
# TASKS (5-phase R10):
#   1. Theory — log-likelihood, Fisher information, reparameterisation
#   2. Build — neg-log-likelihood function with log(sigma) trick
#   3. Train — fit Normal MLE via L-BFGS-B, compare to analytical
#   4. Visualise — profile log-likelihood surface with CI bands
#   5. Apply — DBS risk-weighted capital adequacy SE estimation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from shared.mlfp02.ex_2 import (
    OUTPUT_DIR,
    extract_series,
    fit_normal_mle,
    load_singapore_econ,
    neg_log_likelihood_normal,
    normal_fisher_standard_errors,
    profile_lr_ci_normal_mu,
    save_figure,
    wald_ci,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Log-Likelihood and Fisher Information
# ════════════════════════════════════════════════════════════════════════
#
# For X ~ N(mu, sigma^2), the log-likelihood is:
#   l(mu, sigma | x) = -n/2 * log(2 pi) - n * log(sigma) - Sum(xi - mu)^2 / (2 sigma^2)
#
# REPARAMETERISATION:
#   We optimise over [mu, log(sigma)] instead of [mu, sigma] so that
#   exp(log(sigma)) > 0 always — no explicit bounds needed. This is
#   standard practice for scale parameters in MLE.
#
# FISHER INFORMATION:
#   The Fisher information matrix I(theta) tells you how much information
#   the data carries about each parameter. For N(mu, sigma^2):
#     I(mu)    = n / sigma^2      =>  SE(mu)    = sigma / sqrt(n)
#     I(sigma) = 2n / sigma^2     =>  SE(sigma) = sigma / sqrt(2n)
#
#   These come from inverting the Fisher information matrix at the MLE.
#   Larger n => smaller SE => narrower CIs. Larger sigma => more noise =>
#   wider CIs. The Cramer-Rao bound says no unbiased estimator can beat
#   these SEs.
#
# TWO TYPES OF CONFIDENCE INTERVAL:
#   WALD CI:    mu_hat +/- z * SE(mu_hat)
#     - Simple, uses Fisher SE, assumes quadratic log-likelihood
#     - Good for large n; can be inaccurate for small n
#   PROFILE LR CI:  { mu : 2 * (l(mu_hat) - l(mu)) < chi2(0.95, 1) }
#     - Invariant to reparameterisation
#     - More accurate for small n
#     - Requires evaluating log-likelihood on a grid


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Log-Likelihood and MLE
# ════════════════════════════════════════════════════════════════════════

econ = load_singapore_econ()
gdp_growth = extract_series(econ, "gdp_growth_pct")
n_gdp = len(gdp_growth)

# Analytical MLE (reference)
mle_mu_analytic = gdp_growth.mean()
mle_sigma_analytic = gdp_growth.std(ddof=0)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Numerical MLE via L-BFGS-B
# ════════════════════════════════════════════════════════════════════════

mle_result = fit_normal_mle(gdp_growth)
mle_mu = mle_result["mu"]
mle_sigma = mle_result["sigma"]
loglik_at_mle = mle_result["loglik"]

print(f"\n=== MLE Results: GDP Growth Rate ===")
print(
    f"Analytical MLE:  mu = {mle_mu_analytic:.4f}%, sigma = {mle_sigma_analytic:.4f}%"
)
print(f"Numerical MLE:   mu = {mle_mu:.4f}%, sigma = {mle_sigma:.4f}%")
print(f"Optimizer converged: {mle_result['converged']}")
print(f"Log-likelihood at MLE: {loglik_at_mle:.4f}")

# INTERPRETATION: Analytical and numerical MLE agree closely. The
# numerical approach scales to distributions with no closed-form
# solution (e.g., mixtures, generalised distributions). The
# reparameterisation log(sigma) ensures sigma stays positive without
# requiring explicit bounds.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert mle_result["converged"], "MLE optimizer should converge"
assert abs(mle_mu - mle_mu_analytic) < 0.01, "Numerical and analytical MLE should agree"
print("\n--- Checkpoint 1 passed --- MLE computed analytically and numerically\n")


# ════════════════════════════════════════════════════════════════════════
# Standard Errors and Confidence Intervals
# ════════════════════════════════════════════════════════════════════════

se_mu, se_sigma = normal_fisher_standard_errors(mle_sigma, n_gdp)

print(f"\n=== Standard Errors and Confidence Intervals ===")
print(f"SE(mu_hat) = {se_mu:.4f}%")
print(f"SE(sigma_hat) = {se_sigma:.4f}%")

# 95% Wald confidence interval
mu_wald_ci = wald_ci(mle_mu, se_mu)
print(f"\n95% Wald CI for mu: [{mu_wald_ci[0]:.3f}%, {mu_wald_ci[1]:.3f}%]")

# Profile likelihood CI — more accurate for small n
profile_ci, mu_grid, ll_values = profile_lr_ci_normal_mu(
    gdp_growth, mle_mu, mle_sigma, loglik_at_mle
)
print(f"Profile LR 95% CI for mu: [{profile_ci[0]:.3f}%, {profile_ci[1]:.3f}%]")
print(f"Wald CI width:    {mu_wald_ci[1] - mu_wald_ci[0]:.4f}%")
print(f"Profile CI width: {profile_ci[1] - profile_ci[0]:.4f}%")

# INTERPRETATION: The profile LR CI is invariant to reparameterisation
# and better for small n. The Wald CI assumes the likelihood is quadratic
# at the MLE — a good approximation for large n but not always for small.
# When they agree closely, the Normal approximation is adequate.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert mu_wald_ci[0] < mle_mu < mu_wald_ci[1], "MLE mean must be inside Wald CI"
assert se_mu > 0, "Standard error must be positive"
print("\n--- Checkpoint 2 passed --- SEs and CIs computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Profile Log-Likelihood
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
fig.add_trace(go.Scatter(x=mu_grid, y=ll_values, name="Profile log-likelihood"))
fig.add_vline(x=mle_mu, line_dash="dash", annotation_text="MLE")
fig.add_vline(x=profile_ci[0], line_dash="dot", line_color="red")
fig.add_vline(x=profile_ci[1], line_dash="dot", line_color="red")
fig.update_layout(
    title="Profile Log-Likelihood: GDP Growth Rate",
    xaxis_title="mu (GDP growth %)",
    yaxis_title="Log-likelihood",
)
save_figure(fig, "ex2_02_profile_loglikelihood.html")
print("Saved: ex2_02_profile_loglikelihood.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n--- Checkpoint 3 passed --- profile log-likelihood visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Risk-Weighted Capital Adequacy
# ════════════════════════════════════════════════════════════════════════
# Singapore banks (DBS, OCBC, UOB) compute capital adequacy ratios
# under MAS Notice 637, which requires estimating expected loss rates.
# The standard error of the loss rate estimate determines how much
# capital buffer the bank must hold above the point estimate.
#
# Consider: DBS estimates its probability of default (PD) for SME loans.
# The PD estimate has uncertainty — captured by SE(mu_hat). Wider SE
# means the regulator may demand a larger capital buffer.

print(f"\n=== APPLY: DBS Capital Adequacy SE ===")

# Treat GDP growth as a proxy for credit cycle indicator
# The question: how precisely do we know the average economic condition?
print(f"Point estimate (GDP growth mean): {mle_mu:.3f}%")
print(f"Standard error: {se_mu:.3f}%")
print(f"95% CI: [{mu_wald_ci[0]:.3f}%, {mu_wald_ci[1]:.3f}%]")
print(
    f"\nImplication: If MAS requires capital provisioning against the"
    f"\nlower bound of the CI ({mu_wald_ci[0]:.3f}%), the bank must"
    f"\nhold {mle_mu - mu_wald_ci[0]:.3f}% more buffer than the point estimate."
)
print(
    f"With {n_gdp} quarters of data, SE = {se_mu:.3f}%. If we had twice"
    f"\nthe data, SE would shrink to {se_mu / np.sqrt(2):.3f}% — the value"
    f"\nof longer data histories for regulatory precision."
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert se_mu > 0, "SE must be positive for capital buffer calculation"
print("\n--- Checkpoint 4 passed --- DBS capital application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - Log-likelihood optimisation with scipy.optimize.minimize
  - L-BFGS-B reparameterisation (log sigma enforces sigma > 0)
  - Wald SE from Fisher information: SE(mu_hat) = sigma / sqrt(n)
  - Profile likelihood CI — invariant to reparameterisation
  - When Wald and profile CIs agree: Normal approximation is adequate
  - Real-world impact: capital buffer sizing under regulatory SEs

  NEXT: In 03_map_estimation.py, you'll add a prior to the MLE
  objective, creating MAP estimation, and observe how the prior
  "shrinks" the estimate — the Bayesian regularisation effect.
"""
)

print("--- Exercise 2.2 complete --- MLE and Fisher Information")
