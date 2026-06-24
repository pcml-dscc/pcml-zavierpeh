# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 3.2: Hypothesis Testing & Effect Sizes
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Run a two-proportion z-test for conversion rates
#   - Run Mann-Whitney U for skewed continuous metrics (revenue)
#   - Run Welch's t-test for continuous metrics (AOV, engagement)
#   - Compute and interpret Cohen's h (proportions) and Cohen's d (means)
#   - Distinguish statistical significance from practical significance
#   - Make a data-driven business decision
#
# PREREQUISITES: Exercise 3.1 (bootstrap CIs, power analysis)
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Load data and extract conversion/revenue arrays
#   2. Primary test: two-proportion z-test on conversion rate
#   3. Multiple metrics: revenue, AOV, engagement
#   4. Effect size interpretation (Cohen's h and d)
#   5. Visualise p-values and effect sizes
#   6. Business decision summary
#
# THEORY:
#   - Two-proportion z-test: H0: p_treatment = p_control
#     z = (p_t - p_c) / sqrt(p_pool * (1-p_pool) * (1/n1 + 1/n2))
#   - p-value = P(data | H0 true) -- NOT P(H0 is true)
#   - Cohen's h = 2(arcsin(sqrt(p2)) - arcsin(sqrt(p1)))
#   - Cohen's d = (mean2 - mean1) / s_pooled
#   - Convention: |effect| < 0.2 negligible, < 0.5 small, < 0.8 medium, else large
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from scipy import stats

from shared.mlfp02.ex_3 import (
    ALPHA,
    OUTPUT_DIR,
    load_experiment,
    split_groups,
    conversion_arrays,
    revenue_arrays,
    two_proportion_ztest,
    cohens_h,
    cohens_d,
    interpret_magnitude,
    print_header,
)

print_header("MLFP02 Exercise 3.2: Hypothesis Testing & Effect Sizes")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and extract arrays
# ════════════════════════════════════════════════════════════════════════

df = load_experiment()
control, treatment = split_groups(df)
n_control = control.height
n_treatment = treatment.height
n_total = df.height

ctrl_conv, treat_conv = conversion_arrays(df)
ctrl_rev, treat_rev = revenue_arrays(df)

p_control = ctrl_conv.mean()
p_treatment = treat_conv.mean()

print(f"Data loaded: {n_total:,} users")
print(f"  Control conversion:   {p_control:.4f} ({p_control:.2%})")
print(f"  Treatment conversion: {p_treatment:.4f} ({p_treatment:.2%})")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 < p_control < 1, "Baseline must be a valid proportion"
assert 0 < p_treatment < 1, "Treatment must be a valid proportion"
print("\n>>> Checkpoint 1 passed -- data loaded\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — What a p-value actually means
# ════════════════════════════════════════════════════════════════════════
# p-value = P(observing data this extreme OR more | H0 is true)
#
# It is NOT:
#   - P(H0 is true)
#   - P(the treatment works)
#   - The probability you are wrong
#
# A small p-value means the data is unlikely under the null hypothesis.
# A large effect with small n can have a large p-value (underpowered).
# A tiny effect with huge n can have a tiny p-value (overpowered).
#
# ALWAYS report both p-value AND effect size. p-value answers "is there
# a signal?" Effect size answers "is the signal large enough to matter?"


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Primary test: two-proportion z-test
# ════════════════════════════════════════════════════════════════════════

z_stat, p_value_conversion = two_proportion_ztest(
    p_control, p_treatment, n_control, n_treatment
)

h_conversion = cohens_h(p_control, p_treatment)
h_magnitude = interpret_magnitude(abs(h_conversion))

print(f"=== Primary Metric: Conversion Rate ===")
print(f"Control:   {p_control:.4f} ({p_control:.2%})")
print(f"Treatment: {p_treatment:.4f} ({p_treatment:.2%})")
print(f"Absolute lift: {p_treatment - p_control:+.4f}")
print(f"Relative lift: {(p_treatment - p_control) / p_control:+.2%}")
print(f"z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value_conversion:.6f}")
print(f"Cohen's h: {h_conversion:.4f} ({h_magnitude})")
sig_label = "SIGNIFICANT" if p_value_conversion < ALPHA else "NOT significant"
print(f"{sig_label} at alpha={ALPHA}")

# INTERPRETATION: A small p-value means the observed difference is
# unlikely under H0 (no effect). It does NOT mean the effect is large
# or important -- a tiny effect with huge n can be highly significant.
# ALWAYS report effect size (Cohen's h) alongside the p-value.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert 0 <= p_value_conversion <= 1, "p-value must be between 0 and 1"
print("\n>>> Checkpoint 2 passed -- primary test completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Multiple metrics: revenue, AOV, engagement
# ════════════════════════════════════════════════════════════════════════
# Testing multiple metrics inflates Type I error.
# P(at least one false positive) = 1 - (1 - alpha)^m
# We collect ALL results here; corrections are in 03_multiple_testing.py.

metrics_results = {}

# Metric 1: Conversion (already computed)
metrics_results["conversion_rate"] = {
    "control": p_control,
    "treatment": p_treatment,
    "stat": z_stat,
    "p_value": p_value_conversion,
    "test": "z-test",
}

# Metric 2: Revenue per user (Mann-Whitney U for skewed data)
u_stat, p_value_revenue = stats.mannwhitneyu(
    treat_rev, ctrl_rev, alternative="two-sided"
)
metrics_results["revenue_per_user"] = {
    "control": ctrl_rev.mean(),
    "treatment": treat_rev.mean(),
    "stat": u_stat,
    "p_value": p_value_revenue,
    "test": "Mann-Whitney U",
}

# Metric 3: Average order value (converters only, Welch's t-test)
aov_ctrl = (
    control.filter(pl.col("converted") == 1)["revenue"].to_numpy().astype(np.float64)
)
aov_treat = (
    treatment.filter(pl.col("converted") == 1)["revenue"].to_numpy().astype(np.float64)
)
if len(aov_ctrl) > 1 and len(aov_treat) > 1:
    t_aov, p_aov = stats.ttest_ind(aov_treat, aov_ctrl, equal_var=False)
    metrics_results["avg_order_value"] = {
        "control": aov_ctrl.mean(),
        "treatment": aov_treat.mean(),
        "stat": t_aov,
        "p_value": p_aov,
        "test": "Welch's t-test",
    }

# Metric 4: Primary engagement metric (metric_value)
mv_ctrl = control["metric_value"].to_numpy().astype(np.float64)
mv_treat = treatment["metric_value"].to_numpy().astype(np.float64)
t_mv, p_mv = stats.ttest_ind(mv_treat, mv_ctrl, equal_var=False)
metrics_results["metric_value"] = {
    "control": mv_ctrl.mean(),
    "treatment": mv_treat.mean(),
    "stat": t_mv,
    "p_value": p_mv,
    "test": "Welch's t-test",
}

# Metric 5: Pages viewed (if available)
if "pages_viewed" in df.columns:
    pg_ctrl = control["pages_viewed"].to_numpy().astype(np.float64)
    pg_treat = treatment["pages_viewed"].to_numpy().astype(np.float64)
    t_pg, p_pg = stats.ttest_ind(pg_treat, pg_ctrl, equal_var=False)
    metrics_results["pages_viewed"] = {
        "control": pg_ctrl.mean(),
        "treatment": pg_treat.mean(),
        "stat": t_pg,
        "p_value": p_pg,
        "test": "Welch's t-test",
    }

m = len(metrics_results)
fwer = 1 - (1 - ALPHA) ** m

print(f"=== Multiple Metric Results (unadjusted) ===")
print(
    f"{'Metric':<20} {'Control':>12} {'Treatment':>12} "
    f"{'p-value':>10} {'Test':<18} {'Sig':>4}"
)
print("-" * 80)
for name, r in metrics_results.items():
    sig = (
        "***"
        if r["p_value"] < 0.001
        else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
    )
    print(
        f"{name:<20} {r['control']:>12.4f} {r['treatment']:>12.4f} "
        f"{r['p_value']:>10.6f} {r['test']:<18} {sig:>4}"
    )

print(f"\nNumber of tests (m): {m}")
print(f"Family-wise error rate (uncorrected): {fwer:.4f} ({fwer:.1%})")

# INTERPRETATION: With m tests at alpha=0.05, the probability of at
# least one false positive exceeds the single-test alpha. Multiple
# testing corrections (next file) are essential.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert m >= 3, "Should test at least 3 metrics"
assert fwer > ALPHA, "FWER must exceed single-test alpha"
print("\n>>> Checkpoint 3 passed -- multiple metrics tested\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Effect size interpretation
# ════════════════════════════════════════════════════════════════════════
# Statistical significance != practical significance.
# Cohen's h (proportions): small=0.2, medium=0.5, large=0.8
# Cohen's d (means): small=0.2, medium=0.5, large=0.8

print(f"=== Effect Size Interpretation ===")

for name, r in metrics_results.items():
    ctrl_val = r["control"]
    treat_val = r["treatment"]

    if name == "conversion_rate":
        effect = cohens_h(ctrl_val, treat_val)
        metric_label = "Cohen's h"
    else:
        if name == "revenue_per_user":
            effect = cohens_d(ctrl_rev, treat_rev)
        elif name == "metric_value":
            effect = cohens_d(mv_ctrl, mv_treat)
        elif name == "avg_order_value" and len(aov_ctrl) > 1:
            effect = cohens_d(aov_ctrl, aov_treat)
        else:
            effect = 0.0
        metric_label = "Cohen's d"

    magnitude = interpret_magnitude(abs(effect))
    sig_str = "sig" if r["p_value"] < ALPHA else "ns"
    print(
        f"  {name:<20}: {metric_label}={effect:+.4f} ({magnitude}) "
        f"| p={r['p_value']:.4f} ({sig_str})"
    )

print(f"\n--- Key Insight ---")
print(f"A metric can be statistically significant but practically negligible.")
print(
    f"A metric can be practically important but statistically "
    f"non-significant (underpowered)."
)
print(f"Always report BOTH p-value AND effect size for informed decision-making.")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert h_conversion is not None, "Cohen's h must be computed"
print("\n>>> Checkpoint 4 passed -- effect sizes computed and interpreted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise: p-value bar chart
# ════════════════════════════════════════════════════════════════════════

metric_names = list(metrics_results.keys())
p_values = np.array([r["p_value"] for r in metrics_results.values()])
bonferroni_alpha = ALPHA / m

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=metric_names,
        y=p_values,
        name="Raw p-value",
        marker_color="lightblue",
    )
)
fig.add_hline(y=ALPHA, line_dash="dash", annotation_text=f"alpha={ALPHA}")
fig.add_hline(
    y=bonferroni_alpha,
    line_dash="dot",
    line_color="red",
    annotation_text=f"Bonferroni alpha={bonferroni_alpha:.4f}",
)
fig.update_layout(
    title="P-values with Multiple Testing Thresholds",
    yaxis_title="p-value",
    yaxis_type="log",
)
out_path = OUTPUT_DIR / "pvalue_comparison.html"
fig.write_html(str(out_path))
print(f"Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Business decision for Singapore e-commerce
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: The data science team at a Singapore e-commerce company has
# run a 2-week A/B test on a new recommendation algorithm. The product
# manager asks: "Should we ship the treatment?"
#
# The answer depends on BOTH the p-value and the effect size:
#   - p < 0.05 AND |Cohen's h| > 0.1 -> Ship with confidence
#   - p < 0.05 AND |Cohen's h| < 0.1 -> Statistically significant but
#     practically negligible. Ship only if even tiny gains matter.
#   - p > 0.05 AND |Cohen's h| > 0.1 -> Likely underpowered.
#     Run longer or with more traffic.
#   - p > 0.05 AND |Cohen's h| < 0.1 -> No evidence of meaningful effect.
#
# BUSINESS IMPACT: For a platform processing S$10M daily GMV, a 2pp
# conversion lift means ~S$200K additional daily revenue. Even a
# "small" Cohen's h of 0.05 translates to real money at scale.

print(f"\n=== Business Decision Summary ===")
decision = (
    "Ship the treatment"
    if p_value_conversion < ALPHA and abs(h_conversion) > 0.1
    else (
        "Need more data"
        if abs(h_conversion) > 0.05
        else "No meaningful effect detected"
    )
)
print(
    f"""
Experiment: e-commerce recommendation algorithm A/B test
Sample: {n_total:,} users ({n_control:,} control, {n_treatment:,} treatment)

Primary metric (conversion):
  Lift: {p_treatment - p_control:+.4f} ({(p_treatment - p_control)/p_control:+.1%} relative)
  p-value: {p_value_conversion:.6f} | Cohen's h: {h_conversion:.4f} ({h_magnitude})

Multiple metrics tested: {m} (corrections needed -- see 03_multiple_testing.py)
  FWER without correction: {fwer:.1%}

Recommendation: {decision}
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Two-proportion z-test for conversion rates
  [x] Mann-Whitney U for skewed continuous metrics (revenue)
  [x] Welch's t-test for continuous metrics (AOV, engagement)
  [x] p-value: P(data | H0 true) -- NOT P(H0 is true)
  [x] Cohen's h and d -- practical vs statistical significance
  [x] Business decision framework: both p-value AND effect size

  NEXT: In 03_multiple_testing.py you'll apply Bonferroni and BH-FDR
  corrections to control error rates across multiple metrics.
"""
)

print(">>> Exercise 3.2 complete -- Hypothesis Testing & Effect Sizes")
