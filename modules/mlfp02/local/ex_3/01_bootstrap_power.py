# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 3.1: Bootstrap CIs & Power Analysis
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement bootstrap resampling from scratch (with replacement)
#   - Compute percentile, normal, and BCa confidence intervals
#   - Understand why BCa is the gold standard for bootstrap CIs
#   - Calculate minimum detectable effect (MDE) for a given sample size
#   - Generate power curves — the trade-off between n, effect size, and power
#
# PREREQUISITES: Exercise 2 (MLE, confidence intervals)
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Load A/B test data + SRM sanity check
#   2. Bootstrap resampling from scratch — 10K resamples
#   3. Three CI methods: percentile, normal, BCa
#   4. Power analysis — minimum detectable effect
#   5. Power curves — effect size and sample size
#   6. Visualise bootstrap distribution + power curves
#
# THEORY:
#   Bootstrap: resample WITH REPLACEMENT, compute statistic, repeat.
#   The bootstrap distribution approximates the sampling distribution.
#     - Percentile CI: [q_{alpha/2}, q_{1-alpha/2}] of boot distribution
#     - Normal CI: x_bar +/- z * SE_boot (assumes symmetry)
#     - BCa: bias-corrected and accelerated (gold standard)
#   MDE = (z_{alpha/2} + z_beta) * SE  (smallest reliably detectable effect)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy import stats

from shared.mlfp02.ex_3 import (
    ALPHA,
    POWER_TARGET,
    N_BOOTSTRAP,
    RANDOM_SEED,
    OUTPUT_DIR,
    load_experiment,
    split_groups,
    conversion_arrays,
    srm_check,
    print_header,
)

print_header("MLFP02 Exercise 3.1: Bootstrap CIs & Power Analysis")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data + SRM sanity check
# ════════════════════════════════════════════════════════════════════════
# Before ANY analysis, verify the experiment split is balanced.
# SRM (Sample Ratio Mismatch) detects randomisation bugs, bot traffic,
# or pipeline issues. If SRM fires, do NOT trust downstream results.

df = load_experiment()
control, treatment = split_groups(df)
n_control = control.height
n_treatment = treatment.height
n_total = df.height

print(f"\nData loaded: {n_total:,} users")
print(f"  Control:   {n_control:,}")
print(f"  Treatment: {n_treatment:,}")

# TODO: Call srm_check(n_control, n_treatment) to verify the sample split.
# Hint: srm_check returns a dict with 'chi2', 'p_value', and 'verdict'.
srm = ____

print(f"\nSRM check: chi2={srm['chi2']:.4f}, p={srm['p_value']:.6f}")
print(f"  Verdict: {srm['verdict']}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert srm["p_value"] >= 0, "SRM p-value must be non-negative"
assert n_control + n_treatment == n_total, "Groups must sum to total"
print("\n>>> Checkpoint 1 passed -- SRM check completed\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Bootstrap?
# ════════════════════════════════════════════════════════════════════════
# Classical CIs assume the sampling distribution is Normal (CLT). This
# works for means with large n, but fails for:
#   - Medians, ratios, percentiles (no closed-form SE)
#   - Small samples where CLT hasn't kicked in
#   - Skewed distributions (revenue, time-on-site)
#
# Bootstrap solves this by SIMULATING the sampling distribution:
#   1. Draw n samples WITH REPLACEMENT from your data
#   2. Compute the statistic on the resample
#   3. Repeat 10,000 times
#   4. The distribution of resampled statistics approximates
#      the true sampling distribution
#
# Analogy: You have one bag of 1,000 marbles (your data). You can't
# get more bags from the factory. But you CAN repeatedly grab handfuls
# WITH REPLACEMENT (putting each marble back), record the colour mix,
# and build up a picture of how variable each handful is.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Bootstrap resampling from scratch
# ════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(seed=RANDOM_SEED)
ctrl_conv, treat_conv = conversion_arrays(df)

boot_diffs = np.zeros(N_BOOTSTRAP)
boot_ctrl_rates = np.zeros(N_BOOTSTRAP)
boot_treat_rates = np.zeros(N_BOOTSTRAP)

for i in range(N_BOOTSTRAP):
    # TODO: Resample control WITH REPLACEMENT using rng.choice.
    # Hint: rng.choice(ctrl_conv, size=n_control, replace=True)
    boot_ctrl = ____
    # TODO: Resample treatment WITH REPLACEMENT.
    boot_treat = ____

    boot_ctrl_rates[i] = boot_ctrl.mean()
    boot_treat_rates[i] = boot_treat.mean()
    boot_diffs[i] = boot_treat.mean() - boot_ctrl.mean()

observed_diff = treat_conv.mean() - ctrl_conv.mean()
boot_se = boot_diffs.std()

print(f"Observed conversion diff: {observed_diff:+.6f}")
print(f"Bootstrap SE: {boot_se:.6f}")
print(f"Resamples: {N_BOOTSTRAP:,}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Three CI methods: percentile, normal, BCa
# ════════════════════════════════════════════════════════════════════════

# Method 1: Percentile CI — simplest
# TODO: Use np.percentile on boot_diffs to get [2.5, 97.5] percentiles.
pctile_ci = ____

# Method 2: Normal bootstrap CI — assumes symmetric distribution
# TODO: Compute (observed_diff - 1.96 * boot_se, observed_diff + 1.96 * boot_se)
normal_boot_ci = ____

# Method 3: BCa — bias-corrected and accelerated (gold standard)
# TODO: Use stats.bootstrap with method="BCa", confidence_level=0.95.
# Hint: stats.bootstrap((treat_conv, ctrl_conv),
#         statistic=lambda t, c: t.mean() - c.mean(),
#         n_resamples=N_BOOTSTRAP, confidence_level=0.95, method="BCa",
#         random_state=RANDOM_SEED, batch=100)
# NOTE: batch=100 caps peak memory — without it SciPy builds one
# (n_resamples × 500K) array and the process is OOM-killed.
bca_result = ____
bca_ci = (bca_result.confidence_interval.low, bca_result.confidence_interval.high)

print(f"\n=== Bootstrap CIs for Conversion Rate Difference ===")
print(f"{'Method':<25} {'Lower':>12} {'Upper':>12} {'Width':>12}")
print("-" * 65)
print(
    f"{'Percentile CI':<25} {pctile_ci[0]:>12.6f} {pctile_ci[1]:>12.6f} "
    f"{pctile_ci[1]-pctile_ci[0]:>12.6f}"
)
print(
    f"{'Normal Boot CI':<25} {normal_boot_ci[0]:>12.6f} {normal_boot_ci[1]:>12.6f} "
    f"{normal_boot_ci[1]-normal_boot_ci[0]:>12.6f}"
)
print(
    f"{'BCa CI':<25} {bca_ci[0]:>12.6f} {bca_ci[1]:>12.6f} "
    f"{bca_ci[1]-bca_ci[0]:>12.6f}"
)

# INTERPRETATION: BCa is the gold standard because it corrects for both
# bias (the bootstrap distribution may not be centred at the observed
# statistic) and acceleration (the SE may vary with the parameter value).
# For symmetric, well-behaved statistics like the mean, all three agree.
# For medians, ratios, or skewed data, BCa gives narrower and more
# accurate intervals.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(boot_diffs) == N_BOOTSTRAP, "Should have N_BOOTSTRAP resamples"
assert pctile_ci[0] < pctile_ci[1], "CI lower must be below upper"
assert boot_se > 0, "Bootstrap SE must be positive"
print("\n>>> Checkpoint 2 passed -- bootstrap CIs computed from scratch\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Power analysis: minimum detectable effect
# ════════════════════════════════════════════════════════════════════════
# Given our sample size, what's the smallest effect we can reliably detect
# at alpha=0.05 with 80% power?
#
# MDE = (z_{alpha/2} + z_beta) * sqrt(p(1-p)(1/n1 + 1/n2))

p_control = ctrl_conv.mean()
z_alpha_half = stats.norm.ppf(1 - ALPHA / 2)
z_beta = stats.norm.ppf(POWER_TARGET)

# TODO: Compute pooled standard error for proportions.
# Hint: np.sqrt(p_control * (1 - p_control) * (1/n_control + 1/n_treatment))
pooled_se = ____

# TODO: Compute MDE = (z_alpha_half + z_beta) * pooled_se
mde = ____

print(f"=== Power Analysis ===")
print(f"Baseline conversion rate: {p_control:.4f} ({p_control:.2%})")
print(f"alpha = {ALPHA}, Power = {POWER_TARGET:.0%}")
print(f"z_{{alpha/2}} = {z_alpha_half:.3f}, z_beta = {z_beta:.3f}")
print(f"Minimum Detectable Effect (MDE): {mde:.6f} ({mde:.4%} absolute)")
print(f"Relative MDE: {mde / p_control:.2%} of baseline")

# INTERPRETATION: An MDE of {mde:.4%} absolute means we can reliably
# detect a treatment that changes conversion by at least that many
# percentage points. Smaller effects may exist but are invisible to
# this experiment at 80% power. To detect smaller effects: get more data.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert 0 < mde < 1, "MDE must be a valid proportion"
assert 0 < p_control < 1, "Baseline must be a valid proportion"
print("\n>>> Checkpoint 3 passed -- MDE computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Power curves
# ════════════════════════════════════════════════════════════════════════

# Power vs effect size (fixed n)
effect_sizes = np.linspace(0, mde * 3, 100)
powers_by_effect = []
for delta in effect_sizes:
    # TODO: Compute non-centrality parameter ncp = delta / pooled_se,
    # then power = 1 - norm.cdf(z_alpha_half - ncp) + norm.cdf(-z_alpha_half - ncp)
    ncp = ____
    power_val = ____
    powers_by_effect.append(power_val)

print(f"=== Power Curves ===")
print(f"\n--- Power by Effect Size (n={n_total:,}) ---")
for mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
    idx = min(int(mult / 3 * 99), 99)
    print(f"  Effect = {effect_sizes[idx]:.4%}: Power = {powers_by_effect[idx]:.1%}")

# Power vs sample size (fixed effect = MDE)
sample_sizes_power = np.arange(500, n_total * 2, 500)
powers_by_n = []
for n_per in sample_sizes_power:
    # TODO: Compute SE for this sample size: sqrt(p*(1-p)*2/n_per)
    # then ncp = mde / se_n, then power as above.
    se_n = ____
    ncp = ____
    power_val = ____
    powers_by_n.append(power_val)

print(f"\n--- Power by Sample Size (effect = MDE = {mde:.4%}) ---")
for frac in [0.25, 0.5, 1.0, 1.5, 2.0]:
    idx = min(int(frac * n_total / 500) - 1, len(sample_sizes_power) - 1)
    idx = max(0, idx)
    print(f"  n = {sample_sizes_power[idx]:>8,}: Power = {powers_by_n[idx]:.1%}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert (
    powers_by_effect[-1] > powers_by_effect[0]
), "Power should increase with effect size"
assert len(powers_by_effect) == len(effect_sizes), "One power per effect size"
print("\n>>> Checkpoint 4 passed -- power curves computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise: bootstrap distribution + power curves
# ════════════════════════════════════════════════════════════════════════

from kailash_ml import ModelVisualizer

viz = ModelVisualizer()

# Plot 1: Bootstrap distribution of conversion rate difference
fig1 = viz.histogram(
    pl.DataFrame({"Treatment - Control": boot_diffs}),
    "Treatment - Control",
    title="Bootstrap Distribution: Conversion Rate Difference",
)
fig1.add_vline(x=observed_diff, line_dash="dash", annotation_text="Observed")
fig1.add_vline(x=0, line_dash="dot", line_color="red", annotation_text="H0: no effect")
out_1 = OUTPUT_DIR / "bootstrap_distribution.html"
fig1.write_html(str(out_1))
print(f"Saved: {out_1}")

# Plot 2: Power vs effect size
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=effect_sizes, y=powers_by_effect, name="Power"))
fig2.add_hline(y=0.8, line_dash="dash", annotation_text="80% power target")
fig2.add_vline(x=mde, line_dash="dot", annotation_text=f"MDE={mde:.4f}")
fig2.update_layout(
    title="Statistical Power vs Effect Size",
    xaxis_title="Effect Size (absolute)",
    yaxis_title="Power",
)
out_2 = OUTPUT_DIR / "power_curve.html"
fig2.write_html(str(out_2))
print(f"Saved: {out_2}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore e-commerce experiment planning
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore e-commerce platform (Shopee-scale, ~500K daily
# users) wants to test a new recommendation algorithm. The product team
# asks: "How long do we need to run the experiment?"
#
# With the current baseline conversion of ~10% and an MDE of ~2pp,
# the experiment needs ~{n_total:,} users per group. At 500K daily
# users split 50/50, that's {n_total // 250_000} days minimum.
#
# BUSINESS IMPACT: Without power analysis, teams either:
#   - Stop experiments too early (underpowered) -> miss real effects
#   - Run experiments too long (overpowered) -> waste traffic + delay launches
# A proper MDE calculation saves weeks of experimentation time and
# prevents false negatives worth S$100K+ in unrealised revenue.

print(f"\n--- Business Application: Experiment Planning ---")
daily_users = 500_000
days_needed = max(1, (n_total * 2) // daily_users)
print(f"At {daily_users:,} daily users (50/50 split):")
print(f"  MDE = {mde:.4%} -> need ~{n_total * 2:,} users -> ~{days_needed} days")
small_effect_n = int(
    ((z_alpha_half + z_beta) / (mde / 2)) ** 2 * p_control * (1 - p_control) * 2
)
days_small = max(1, small_effect_n // daily_users)
print(f"  Half the MDE -> need ~{small_effect_n:,} users -> ~{days_small} days")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] SRM: chi-squared test detects broken randomisation before trusting results
  [x] Bootstrap: resample with replacement, compute any statistic's CI
  [x] Three CI methods: percentile (simple), normal (symmetric), BCa (gold standard)
  [x] MDE: smallest detectable effect at given n, alpha, and power
  [x] Power curves: visualise trade-off between n, effect size, and power

  NEXT: In 02_hypothesis_testing.py you'll use these power calculations
  to run the actual hypothesis test and compute effect sizes.
"""
)

print(">>> Exercise 3.1 complete -- Bootstrap CIs & Power Analysis")
