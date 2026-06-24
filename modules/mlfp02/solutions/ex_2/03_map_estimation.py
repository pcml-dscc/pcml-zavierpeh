# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 2.3: MAP Estimation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Distinguish MLE (no prior) from MAP estimation (MLE + prior)
#   - Implement MAP as neg-log-likelihood + neg-log-prior
#   - Observe shrinkage: MAP pulls the estimate toward the prior
#   - Quantify how shrinkage varies with sample size (n)
#   - Connect MAP to L2 regularisation (Ridge regression)
#
# PREREQUISITES: 02_mle_fisher.py (MLE, Fisher information)
# ESTIMATED TIME: ~25 minutes
#
# TASKS (5-phase R10):
#   1. Theory — MAP = MLE + log-prior, shrinkage mechanics
#   2. Build — MAP objective with Normal prior on mu
#   3. Train — fit MAP at different sample sizes, measure shrinkage
#   4. Visualise — MLE vs MAP vs Prior across sample sizes
#   5. Apply — GIC Singapore SME lending with informative prior
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

from shared.mlfp02.ex_2 import (
    GDP_PRIOR_MEAN,
    GDP_PRIOR_STD,
    OUTPUT_DIR,
    extract_series,
    fit_normal_map,
    fit_normal_mle,
    load_singapore_econ,
    neg_log_likelihood_normal,
    save_figure,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — MAP Estimation
# ════════════════════════════════════════════════════════════════════════
#
# MAXIMUM A POSTERIORI (MAP):
#   MAP = argmax p(theta | x) = argmax [ l(theta | x) + log p(theta) ]
#
# In words: find the parameter value that maximises the POSTERIOR, which
# is the product of likelihood and prior. Since we maximise on log scale:
#   neg_map = neg_log_likelihood + neg_log_prior
#
# KEY INSIGHT — SHRINKAGE:
#   The prior "pulls" the MAP estimate away from the MLE toward the prior
#   mean. The strength of this pull depends on:
#     - Prior precision (1/sigma_prior^2): tighter prior => more pull
#     - Data precision (n/sigma^2): more data => less pull
#
#   For Normal likelihood + Normal prior on mu:
#     mu_MAP = (n * x_bar / sigma^2 + mu_prior / sigma_prior^2)
#              / (n / sigma^2 + 1 / sigma_prior^2)
#
#   This is a PRECISION-WEIGHTED AVERAGE of the MLE and the prior mean.
#
# CONNECTION TO RIDGE REGRESSION:
#   MAP with a Gaussian prior on coefficients is EXACTLY Ridge regression.
#   The penalty lambda = sigma^2 / sigma_prior^2. So:
#     Large lambda <=> tight prior <=> strong regularisation
#     Small lambda <=> wide prior  <=> close to OLS/MLE


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: MAP Objective
# ════════════════════════════════════════════════════════════════════════

econ = load_singapore_econ()
gdp_growth = extract_series(econ, "gdp_growth_pct")
n_gdp = len(gdp_growth)

# Prior: mu ~ N(3.5%, 1.5^2%) — typical growth for a small open economy
print(f"\n=== MAP Estimation Setup ===")
print(f"Prior: mu ~ N({GDP_PRIOR_MEAN}, {GDP_PRIOR_STD}^2)")
print(f"Data: {n_gdp} quarterly GDP growth observations")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Fit MAP and Compare with MLE
# ════════════════════════════════════════════════════════════════════════

mle_result = fit_normal_mle(gdp_growth)
map_result = fit_normal_map(gdp_growth, GDP_PRIOR_MEAN, GDP_PRIOR_STD)

mle_mu = mle_result["mu"]
mle_sigma = mle_result["sigma"]
map_mu = map_result["mu"]
map_sigma = map_result["sigma"]

print(f"\n=== MAP vs MLE Comparison ===")
print(f"Prior: mu ~ N({GDP_PRIOR_MEAN}, {GDP_PRIOR_STD}^2)")
print(f"n = {n_gdp} observations")
print(f"MLE: mu_hat = {mle_mu:.4f}%, sigma_hat = {mle_sigma:.4f}%")
print(f"MAP: mu_hat = {map_mu:.4f}%, sigma_hat = {map_sigma:.4f}%")
print(f"MAP shrinkage toward prior: {map_mu - mle_mu:+.4f}%")

# Show shrinkage at different sample sizes
print(f"\n--- MAP Shrinkage by Sample Size ---")
shrinkage_data: list[dict[str, float]] = []

for n_small in [3, 5, 10, 20, 50, n_gdp]:
    sample = gdp_growth[: min(n_small, n_gdp)]
    r_mle = fit_normal_mle(sample)
    r_map = fit_normal_map(sample, GDP_PRIOR_MEAN, GDP_PRIOR_STD)
    mle_s = r_mle["mu"]
    map_s = r_map["mu"]
    shrinkage_pct = abs(map_s - mle_s) / (abs(GDP_PRIOR_MEAN - mle_s) + 1e-10) * 100
    shrinkage_data.append(
        {"n": n_small, "mle": mle_s, "map": map_s, "shrinkage_pct": shrinkage_pct}
    )
    print(
        f"  n={n_small:>3}: MLE={mle_s:>7.3f}%, MAP={map_s:>7.3f}%, "
        f"shrinkage={map_s - mle_s:+.3f}% ({shrinkage_pct:.1f}% toward prior)"
    )

# INTERPRETATION: MAP shrinks toward the prior. The shrinkage is
# proportional to the prior precision relative to data precision. With
# n=3, the prior has ~30% influence. With n=50+, the prior barely
# matters. MAP is equivalent to L2 regularisation (Ridge) in regression.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert map_result["converged"], "MAP optimizer should converge"
assert (
    abs(map_mu - mle_mu) <= abs(GDP_PRIOR_MEAN - mle_mu) + 0.1
), "MAP should lie between MLE and prior (cannot overshoot)"
print("\n--- Checkpoint 1 passed --- MAP estimation and shrinkage demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: MLE vs MAP vs Prior
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
ns = [d["n"] for d in shrinkage_data]
fig.add_trace(
    go.Scatter(
        x=ns,
        y=[d["mle"] for d in shrinkage_data],
        mode="lines+markers",
        name="MLE",
    )
)
fig.add_trace(
    go.Scatter(
        x=ns,
        y=[d["map"] for d in shrinkage_data],
        mode="lines+markers",
        name="MAP",
    )
)
fig.add_hline(
    y=GDP_PRIOR_MEAN,
    line_dash="dash",
    line_color="green",
    annotation_text=f"Prior mean ({GDP_PRIOR_MEAN}%)",
)
fig.update_layout(
    title="MAP Shrinkage: Convergence to MLE as n Grows",
    xaxis_title="Sample size (n)",
    yaxis_title="Estimated mu (%)",
    xaxis_type="log",
)
save_figure(fig, "ex2_03_map_shrinkage.html")
print("Saved: ex2_03_map_shrinkage.html")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
print("\n--- Checkpoint 2 passed --- MAP shrinkage visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: GIC Singapore SME Lending Prior
# ════════════════════════════════════════════════════════════════════════
# GIC (Singapore's sovereign wealth fund) backstops the SME lending
# ecosystem. When a new lending programme launches, there are few data
# points — perhaps only 3-5 quarters of default history.
#
# An MLE-only approach would produce wildly unstable estimates of the
# default rate. MAP with an informative prior (based on historical
# default rates across similar programmes) stabilises the estimate.
#
# Scenario: A new green-bond SME programme has 5 quarters of data
# showing GDP growth of [2.1, 3.8, 1.5, 4.2, 2.9]. The prior from
# 20 years of SME lending experience says growth ~ N(3.5, 1.5^2).

print(f"\n=== APPLY: GIC SME Lending — MAP with Informative Prior ===")

new_programme_data = np.array([2.1, 3.8, 1.5, 4.2, 2.9])
n_prog = len(new_programme_data)

mle_prog = fit_normal_mle(new_programme_data)
map_prog = fit_normal_map(new_programme_data, GDP_PRIOR_MEAN, GDP_PRIOR_STD)

print(f"New programme: {n_prog} quarters of data")
print(f"Data: {new_programme_data}")
print(f"Prior: mu ~ N({GDP_PRIOR_MEAN}, {GDP_PRIOR_STD}^2) (20-year experience)")
print(f"\nMLE estimate: {mle_prog['mu']:.3f}%")
print(f"MAP estimate: {map_prog['mu']:.3f}%")
print(f"Shrinkage: {map_prog['mu'] - mle_prog['mu']:+.3f}% toward prior")
print(
    f"\nWith only {n_prog} observations, MLE is highly variable."
    f"\nMAP incorporates institutional knowledge (the prior) to produce"
    f"\na more stable estimate. As more quarters accumulate, the MAP"
    f"\nwill converge to the MLE — the data eventually overwhelms the prior."
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert map_prog["converged"], "MAP should converge on small dataset"
print("\n--- Checkpoint 3 passed --- GIC SME lending application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - MAP = MLE + log-prior; maximises the posterior, not just likelihood
  - Shrinkage: MAP pulls estimates toward the prior mean
  - Precision-weighted average: data precision vs prior precision
  - As n grows, MAP converges to MLE (data overwhelms prior)
  - Connection to Ridge regression (L2 regularisation)
  - Real-world impact: stabilising estimates with few data points

  NEXT: In 04_mle_failures.py, you'll explore three scenarios where
  MLE gives misleading results: small samples, multimodal data, and
  misspecified likelihood (wrong distribution family).
"""
)

print("--- Exercise 2.3 complete --- MAP Estimation")
