# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 3.4: Permutation Test
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement a permutation test from scratch
#   - Understand why permutation tests require no distributional assumptions
#   - Compare permutation p-values with parametric p-values
#   - Apply permutation tests to both conversion and revenue metrics
#   - Visualise the permutation null distribution
#
# PREREQUISITES: Exercise 3.2 (hypothesis testing)
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Load data and compute observed differences
#   2. Permutation test for conversion rate (10K permutations)
#   3. Permutation test for revenue (non-Normal metric)
#   4. Compare with parametric results
#   5. Visualise permutation null distributions
#
# THEORY:
#   Permutation test algorithm:
#     1. Pool ALL observations (control + treatment)
#     2. Randomly assign to "control" and "treatment" (preserving sizes)
#     3. Compute test statistic on permuted data
#     4. Repeat 10K times -> null distribution
#     5. p-value = proportion of |permuted stat| >= |observed stat|
#
#   Key advantage: NO assumptions about the shape of the distribution.
#   If the parametric test assumes Normality and the data is skewed,
#   the permutation test gives a more trustworthy p-value.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy import stats

from shared.mlfp02.ex_3 import (
    ALPHA,
    N_PERMUTATIONS,
    RANDOM_SEED,
    OUTPUT_DIR,
    load_experiment,
    split_groups,
    conversion_arrays,
    revenue_arrays,
    two_proportion_ztest,
    print_header,
)

print_header("MLFP02 Exercise 3.4: Permutation Test")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and compute observed differences
# ════════════════════════════════════════════════════════════════════════

df = load_experiment()
control, treatment = split_groups(df)
n_control = control.height
n_treatment = treatment.height

ctrl_conv, treat_conv = conversion_arrays(df)
ctrl_rev, treat_rev = revenue_arrays(df)

observed_conv_diff = treat_conv.mean() - ctrl_conv.mean()
observed_rev_diff = treat_rev.mean() - ctrl_rev.mean()

print(f"Data loaded: {df.height:,} users")
print(f"  Observed conversion diff: {observed_conv_diff:+.6f}")
print(f"  Observed revenue diff:    ${observed_rev_diff:+.2f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert n_control > 0 and n_treatment > 0, "Both groups must have data"
print("\n>>> Checkpoint 1 passed -- data loaded\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why permutation tests?
# ════════════════════════════════════════════════════════════════════════
# The z-test assumes the sampling distribution of the difference is
# Normal. For conversion rates with large n, the CLT guarantees this.
# But for revenue (highly skewed, heavy-tailed), the CLT approximation
# may be poor even at n=5000.
#
# The permutation test asks a simpler question: "If the treatment had
# NO effect, how likely is it that we'd see a difference this large
# just by random assignment?"
#
# We answer this by ACTUALLY doing the random assignment 10,000 times:
#   - Pool all users together (erasing group labels)
#   - Randomly assign n_control to "control", rest to "treatment"
#   - Compute the difference in the permuted groups
#   - Count how often the permuted difference is as extreme as observed
#
# Analogy: Imagine you're told a coin is biased. You flip it 100 times
# and get 58 heads. Is that enough evidence? You could compute the
# binomial test (parametric). OR you could shuffle a deck of 100 cards
# (58 red, 42 black), deal them into two piles, and count how often
# the difference is as extreme as 58 vs 42. No math needed -- just
# repeated shuffling.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Permutation test for conversion rate
# ════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(seed=RANDOM_SEED)
all_converted = df["converted"].to_numpy()

perm_conv_diffs = np.zeros(N_PERMUTATIONS)
for i in range(N_PERMUTATIONS):
    perm = rng.permutation(all_converted)
    perm_ctrl_rate = perm[:n_control].mean()
    perm_treat_rate = perm[n_control:].mean()
    perm_conv_diffs[i] = perm_treat_rate - perm_ctrl_rate

perm_p_conversion = np.mean(np.abs(perm_conv_diffs) >= np.abs(observed_conv_diff))

print(f"=== Permutation Test (conversion rate) ===")
print(f"Observed difference: {observed_conv_diff:+.6f}")
print(
    f"Permutation null: mean={perm_conv_diffs.mean():.6f}, "
    f"std={perm_conv_diffs.std():.6f}"
)
print(f"Permutation p-value: {perm_p_conversion:.6f}")

# Compare with parametric
_, parametric_p = two_proportion_ztest(
    ctrl_conv.mean(), treat_conv.mean(), n_control, n_treatment
)
print(f"Parametric p-value:  {parametric_p:.6f}")
agreement = (perm_p_conversion < ALPHA) == (parametric_p < ALPHA)
print(f"Agreement: {'YES' if agreement else 'NO'}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert 0 <= perm_p_conversion <= 1, "Permutation p-value must be valid"
assert len(perm_conv_diffs) == N_PERMUTATIONS, "Should have N_PERMUTATIONS samples"
assert (
    abs(perm_conv_diffs.mean()) < 0.01
), "Permutation null should be centred near zero"
print("\n>>> Checkpoint 2 passed -- conversion permutation test completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Permutation test for revenue (non-Normal)
# ════════════════════════════════════════════════════════════════════════
# Revenue is typically right-skewed (many small purchases, few large).
# The parametric Mann-Whitney U test is robust to this, but the
# permutation test is even more direct -- no assumptions at all.

all_revenue = df["revenue"].to_numpy().astype(np.float64)

perm_rev_diffs = np.zeros(N_PERMUTATIONS)
for i in range(N_PERMUTATIONS):
    perm = rng.permutation(all_revenue)
    perm_rev_diffs[i] = perm[n_control:].mean() - perm[:n_control].mean()

perm_p_revenue = np.mean(np.abs(perm_rev_diffs) >= np.abs(observed_rev_diff))

print(f"=== Permutation Test (revenue) ===")
print(f"Observed revenue diff: ${observed_rev_diff:+.2f}")
print(
    f"Permutation null: mean=${perm_rev_diffs.mean():.2f}, "
    f"std=${perm_rev_diffs.std():.2f}"
)
print(f"Permutation p-value: {perm_p_revenue:.6f}")

# Compare with Mann-Whitney U
_, mwu_p = stats.mannwhitneyu(treat_rev, ctrl_rev, alternative="two-sided")
print(f"Mann-Whitney p-value: {mwu_p:.6f}")
rev_agreement = (perm_p_revenue < ALPHA) == (mwu_p < ALPHA)
print(f"Agreement: {'YES' if rev_agreement else 'NO'}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert 0 <= perm_p_revenue <= 1, "Revenue permutation p-value must be valid"
assert len(perm_rev_diffs) == N_PERMUTATIONS, "Should have N_PERMUTATIONS samples"
print("\n>>> Checkpoint 3 passed -- revenue permutation test completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Comparison summary
# ════════════════════════════════════════════════════════════════════════

print(f"=== Parametric vs Permutation Comparison ===")
print(f"{'Metric':<20} {'Parametric p':>14} {'Permutation p':>15} {'Agree':>8}")
print("-" * 60)
print(
    f"{'Conversion':<20} {parametric_p:>14.6f} {perm_p_conversion:>15.6f} "
    f"{'YES' if agreement else 'NO':>8}"
)
print(
    f"{'Revenue':<20} {mwu_p:>14.6f} {perm_p_revenue:>15.6f} "
    f"{'YES' if rev_agreement else 'NO':>8}"
)

# INTERPRETATION: When parametric and permutation tests agree, both
# assumptions hold and you can trust either. When they disagree, the
# permutation test is more trustworthy because it makes fewer
# assumptions. In practice, for conversion rates (binary, large n),
# they almost always agree. For revenue (skewed, heavy-tailed),
# disagreement is more common.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n>>> Checkpoint 4 passed -- comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise permutation null distributions
# ════════════════════════════════════════════════════════════════════════

from kailash_ml import ModelVisualizer

viz = ModelVisualizer()

# Conversion permutation null
conv_df = pl.DataFrame({"permuted_difference": perm_conv_diffs})
fig1 = viz.histogram(
    conv_df,
    column="permuted_difference",
    bins=50,
    title="Permutation Null: Conversion Rate Difference",
)
fig1.add_vline(
    x=observed_conv_diff,
    line_dash="dash",
    annotation_text=f"Observed ({observed_conv_diff:+.4f})",
)
fig1.add_vline(
    x=-abs(observed_conv_diff),
    line_dash="dash",
    line_color="red",
    annotation_text="Mirror",
)
out_1 = OUTPUT_DIR / "permutation_null_conversion.html"
fig1.write_html(str(out_1))
print(f"Saved: {out_1}")

# Revenue permutation null
rev_df = pl.DataFrame({"permuted_revenue_difference": perm_rev_diffs})
fig2 = viz.histogram(
    rev_df,
    column="permuted_revenue_difference",
    bins=50,
    title="Permutation Null: Revenue Difference",
)
fig2.add_vline(
    x=observed_rev_diff,
    line_dash="dash",
    annotation_text=f"Observed (${observed_rev_diff:+.2f})",
)
fig2.add_vline(
    x=-abs(observed_rev_diff),
    line_dash="dash",
    line_color="red",
    annotation_text="Mirror",
)
out_2 = OUTPUT_DIR / "permutation_null_revenue.html"
fig2.write_html(str(out_2))
print(f"Saved: {out_2}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Distribution-free testing for a Singapore ride-hailing platform
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore ride-hailing platform tests a new surge pricing
# algorithm. The key metric is revenue per ride, which is highly skewed:
# most rides cost S$8-15, but airport rides, peak-hour rides, and
# luxury rides create a heavy right tail (S$50-200).
#
# A t-test on this data is questionable -- the distribution is nowhere
# near Normal, and even with n=10,000 the CLT approximation for the
# mean is unreliable because the variance is dominated by the tail.
#
# The permutation test handles this naturally:
#   - No distributional assumptions
#   - Exact p-value (up to permutation count)
#   - Works for ANY test statistic (mean, median, trimmed mean, ratio)
#
# The team runs a permutation test with 10K permutations and finds:
#   - Parametric t-test: p=0.032 (significant)
#   - Permutation test:  p=0.071 (not significant)
#
# The disagreement reveals that the t-test was fooled by a few extreme
# rides in the treatment group. The permutation test, which doesn't
# assume Normality, gives the correct answer: the treatment effect is
# not reliable.
#
# BUSINESS IMPACT: Without the permutation test, the team would have
# shipped a pricing algorithm that appeared to increase revenue but
# actually just happened to catch a few high-value airport rides in the
# treatment group. Avoiding this false positive saves ~S$200K in
# engineering effort and prevents a price increase that would have
# reduced rider satisfaction with no real revenue benefit.

print(f"\n--- Business Application: Distribution-Free Testing ---")
print("Revenue distributions are rarely Normal.")
print("Permutation tests give reliable p-values regardless of distribution shape.")
print("When parametric and permutation tests disagree, trust the permutation test.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Permutation test: distribution-free, no parametric assumptions
  [x] Permutation null distribution: what "no effect" actually looks like
  [x] Conversion + revenue permutation tests from scratch
  [x] Parametric vs permutation comparison: when they agree and disagree

  EXERCISE 3 COMPLETE: You now have the full A/B testing toolkit:
    3.1: Bootstrap CIs + power analysis (how precise, how sensitive)
    3.2: Hypothesis testing + effect sizes (is it real, is it big enough)
    3.3: Multiple testing corrections (controlling false discoveries)
    3.4: Permutation tests (no assumptions, trustworthy p-values)

  NEXT: In Exercise 4 you'll design a complete experiment from scratch --
  writing hypotheses before data, computing required sample sizes,
  and creating a structured data collection plan.
"""
)

print(">>> Exercise 3.4 complete -- Permutation Test")
