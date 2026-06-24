# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 3.3: Multiple Testing Corrections
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why testing multiple metrics inflates false positive rates
#   - Apply Bonferroni correction (FWER control)
#   - Apply Benjamini-Hochberg correction (FDR control)
#   - Compute q-values (FDR-adjusted p-values)
#   - Simulate false discovery rates to see corrections in action
#   - Choose between FWER and FDR control for different settings
#
# PREREQUISITES: Exercise 3.2 (hypothesis testing, multiple metrics)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Load data and run multiple metric tests
#   2. Bonferroni correction (FWER control)
#   3. Benjamini-Hochberg correction (FDR control) with q-values
#   4. Correction comparison table
#   5. False discovery rate simulation (100 tests, 500 rounds)
#   6. Visualise FDR simulation results
#
# THEORY:
#   - FWER = P(at least one false positive among m tests)
#   - Bonferroni: reject if p < alpha/m. Controls FWER. Very conservative.
#   - FDR = E[false discoveries / total discoveries]
#   - BH procedure: sort p-values, reject if p(k) <= (k/m) * alpha
#   - q-value: the minimum FDR at which a test would be declared significant
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
    RANDOM_SEED,
    OUTPUT_DIR,
    load_experiment,
    split_groups,
    conversion_arrays,
    revenue_arrays,
    two_proportion_ztest,
    print_header,
)

print_header("MLFP02 Exercise 3.3: Multiple Testing Corrections")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and run multiple metric tests
# ════════════════════════════════════════════════════════════════════════

df = load_experiment()
control, treatment = split_groups(df)
n_control = control.height
n_treatment = treatment.height

ctrl_conv, treat_conv = conversion_arrays(df)
ctrl_rev, treat_rev = revenue_arrays(df)
p_control = ctrl_conv.mean()
p_treatment = treat_conv.mean()

# Collect all metric p-values
metrics_results = {}

# Conversion
z_stat, p_conv = two_proportion_ztest(p_control, p_treatment, n_control, n_treatment)
metrics_results["conversion_rate"] = {"p_value": p_conv, "test": "z-test"}

# Revenue per user
_, p_rev = stats.mannwhitneyu(treat_rev, ctrl_rev, alternative="two-sided")
metrics_results["revenue_per_user"] = {"p_value": p_rev, "test": "Mann-Whitney U"}

# AOV (converters only)
aov_ctrl = (
    control.filter(pl.col("converted") == 1)["revenue"].to_numpy().astype(np.float64)
)
aov_treat = (
    treatment.filter(pl.col("converted") == 1)["revenue"].to_numpy().astype(np.float64)
)
if len(aov_ctrl) > 1 and len(aov_treat) > 1:
    _, p_aov = stats.ttest_ind(aov_treat, aov_ctrl, equal_var=False)
    metrics_results["avg_order_value"] = {"p_value": p_aov, "test": "Welch's t-test"}

# Engagement (metric_value)
mv_ctrl = control["metric_value"].to_numpy().astype(np.float64)
mv_treat = treatment["metric_value"].to_numpy().astype(np.float64)
_, p_mv = stats.ttest_ind(mv_treat, mv_ctrl, equal_var=False)
metrics_results["metric_value"] = {"p_value": p_mv, "test": "Welch's t-test"}

# Pages viewed (if available)
if "pages_viewed" in df.columns:
    pg_ctrl = control["pages_viewed"].to_numpy().astype(np.float64)
    pg_treat = treatment["pages_viewed"].to_numpy().astype(np.float64)
    _, p_pg = stats.ttest_ind(pg_treat, pg_ctrl, equal_var=False)
    metrics_results["pages_viewed"] = {"p_value": p_pg, "test": "Welch's t-test"}

metric_names = list(metrics_results.keys())
p_values = np.array([r["p_value"] for r in metrics_results.values()])
m = len(metrics_results)

print(f"Metrics tested: {m}")
for name, r in metrics_results.items():
    print(f"  {name}: p={r['p_value']:.6f} ({r['test']})")

fwer = 1 - (1 - ALPHA) ** m
print(f"\nUncorrected FWER: {fwer:.4f} ({fwer:.1%})")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert m >= 3, "Should test at least 3 metrics"
assert fwer > ALPHA, "FWER must exceed single-test alpha"
print("\n>>> Checkpoint 1 passed -- metrics collected\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why multiple testing is dangerous
# ════════════════════════════════════════════════════════════════════════
# Imagine rolling a 20-sided die. The chance of rolling a 1 is 5%.
# But if you roll it 20 times, the chance of AT LEAST ONE 1 is:
#   1 - (19/20)^20 = 64%
#
# Same with p-values. Each test at alpha=0.05 has a 5% false positive
# rate. But 20 tests give you a 64% chance of at least one false
# positive. The two corrections below address this:
#
# Bonferroni (FWER): "I want at most a 5% chance of ANY false positive"
#   -> Very strict, fewer discoveries, but high confidence in each one
#
# BH-FDR: "I accept that 5% of my discoveries may be false"
#   -> Less strict, more discoveries, suitable for exploration


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Bonferroni correction (FWER control)
# ════════════════════════════════════════════════════════════════════════
# Most conservative: alpha_adj = alpha / m
# Controls FWER -- P(ANY false positive among m tests)

bonferroni_alpha = ALPHA / m
bonferroni_significant = p_values < bonferroni_alpha

print(f"=== Bonferroni Correction (FWER control) ===")
print(f"Adjusted alpha = {ALPHA}/{m} = {bonferroni_alpha:.4f}")
for i, name in enumerate(metric_names):
    sig = "SIGNIFICANT" if bonferroni_significant[i] else "not significant"
    print(f"  {name}: p={p_values[i]:.6f} -> {sig}")

print(f"\nBonferroni rejects: {sum(bonferroni_significant)}/{m}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert bonferroni_alpha < ALPHA, "Bonferroni alpha must be smaller than original alpha"
print("\n>>> Checkpoint 2 passed -- Bonferroni correction applied\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Benjamini-Hochberg correction (FDR control)
# ════════════════════════════════════════════════════════════════════════
# Less conservative: controls False Discovery Rate (expected proportion
# of false discoveries among all discoveries).
# Procedure: sort p-values, find largest k where p(k) <= (k/m) * alpha

sorted_indices = np.argsort(p_values)
sorted_p = p_values[sorted_indices]
bh_thresholds = np.array([(k + 1) / m * ALPHA for k in range(m)])

# Find the largest k where p(k) <= threshold
max_k = -1
for k in range(m):
    if sorted_p[k] <= bh_thresholds[k]:
        max_k = k

bh_significant_mask = np.zeros(m, dtype=bool)
if max_k >= 0:
    bh_significant_mask[sorted_indices[: max_k + 1]] = True

# Compute q-values (adjusted p-values)
q_values = np.zeros(m)
sorted_q = np.zeros(m)
sorted_q[m - 1] = sorted_p[m - 1]
for k in range(m - 2, -1, -1):
    sorted_q[k] = min(sorted_p[k] * m / (k + 1), sorted_q[k + 1])
q_values[sorted_indices] = sorted_q

print(f"=== Benjamini-Hochberg Correction (FDR control) ===")
print(f"Target FDR: {ALPHA}")
print(
    f"{'Rank':>4} {'Metric':<20} {'p-value':>10} {'Threshold':>10} "
    f"{'q-value':>10} {'Sig':>6}"
)
print("-" * 65)
for k in range(m):
    orig_idx = sorted_indices[k]
    name = metric_names[orig_idx]
    sig = "YES" if bh_significant_mask[orig_idx] else "no"
    print(
        f"{k+1:>4} {name:<20} {sorted_p[k]:>10.6f} {bh_thresholds[k]:>10.6f} "
        f"{sorted_q[k]:>10.6f} {sig:>6}"
    )

print(f"\nBH-FDR rejects: {sum(bh_significant_mask)}/{m}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert all(0 <= q <= 1 for q in q_values), "All q-values must be valid"
assert all(q_values >= p_values - 1e-10), "q-values must be >= raw p-values"
print("\n>>> Checkpoint 3 passed -- BH-FDR correction applied\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Correction comparison
# ════════════════════════════════════════════════════════════════════════

print(f"=== Correction Comparison ===")
print(f"{'Metric':<20} {'Raw p':>10} {'Bonferroni':>12} {'BH-FDR':>12}")
print("-" * 60)
for i, name in enumerate(metric_names):
    bonf = "sig" if bonferroni_significant[i] else "ns"
    bh = "sig" if bh_significant_mask[i] else "ns"
    print(f"{name:<20} {p_values[i]:>10.6f} {bonf:>12} {bh:>12}")

# INTERPRETATION: BH-FDR allows more discoveries than Bonferroni. In
# exploratory settings (many metrics, screening), BH-FDR is preferred.
# In confirmatory settings (one primary metric, regulatory), Bonferroni
# is safer. Most A/B test platforms use BH-FDR for guardrail metrics.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert sum(bh_significant_mask) >= sum(
    bonferroni_significant
), "BH-FDR should reject at least as many as Bonferroni"
print("\n>>> Checkpoint 4 passed -- correction comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — False discovery rate simulation
# ════════════════════════════════════════════════════════════════════════
# Simulate many tests where we KNOW most have no real effect.
# Show that Bonferroni and BH-FDR control error rates as advertised.

print(f"=== False Discovery Rate Simulation ===")

rng = np.random.default_rng(seed=RANDOM_SEED)
n_sim_tests = 100
n_sim_rounds = 500
n_truly_different = 10  # Only 10 out of 100 have real effects

bonf_false_discoveries = []
bh_false_discoveries = []
bonf_true_discoveries = []
bh_true_discoveries = []
raw_false_discoveries = []

for _ in range(n_sim_rounds):
    sim_p_values = np.zeros(n_sim_tests)
    is_true_null = np.ones(n_sim_tests, dtype=bool)

    for j in range(n_sim_tests):
        n_per = 500
        ctrl_sim = rng.binomial(1, 0.10, size=n_per)
        if j < n_truly_different:
            # Real effect: treatment rate is 12% (vs 10% control)
            treat_sim = rng.binomial(1, 0.12, size=n_per)
            is_true_null[j] = False
        else:
            # No effect
            treat_sim = rng.binomial(1, 0.10, size=n_per)

        # z-test
        p_c = ctrl_sim.mean()
        p_t = treat_sim.mean()
        p_pool = (ctrl_sim.sum() + treat_sim.sum()) / (2 * n_per)
        se = np.sqrt(p_pool * (1 - p_pool) * 2 / n_per) if 0 < p_pool < 1 else 1e-10
        z = (p_t - p_c) / se if se > 0 else 0
        sim_p_values[j] = 2 * (1 - stats.norm.cdf(abs(z)))

    # Raw (no correction)
    raw_sig = sim_p_values < ALPHA
    raw_false = raw_sig & is_true_null
    raw_false_discoveries.append(raw_false.sum())

    # Bonferroni
    bonf_sig = sim_p_values < (ALPHA / n_sim_tests)
    bonf_false = bonf_sig & is_true_null
    bonf_true = bonf_sig & ~is_true_null
    bonf_false_discoveries.append(bonf_false.sum())
    bonf_true_discoveries.append(bonf_true.sum())

    # BH-FDR
    sorted_idx = np.argsort(sim_p_values)
    sorted_pv = sim_p_values[sorted_idx]
    bh_thresh = np.array([(k + 1) / n_sim_tests * ALPHA for k in range(n_sim_tests)])
    mk = -1
    for k in range(n_sim_tests):
        if sorted_pv[k] <= bh_thresh[k]:
            mk = k
    bh_sig = np.zeros(n_sim_tests, dtype=bool)
    if mk >= 0:
        bh_sig[sorted_idx[: mk + 1]] = True
    bh_false = bh_sig & is_true_null
    bh_true = bh_sig & ~is_true_null
    bh_false_discoveries.append(bh_false.sum())
    bh_true_discoveries.append(bh_true.sum())

print(f"Simulation: {n_sim_tests} tests per round, {n_truly_different} truly different")
print(f"Rounds: {n_sim_rounds}")
print(f"\n{'Method':<15} {'Avg False Disc':>15} {'Avg True Disc':>15} {'FDR':>8}")
print("-" * 55)
avg_raw_fd = np.mean(raw_false_discoveries)
print(f"{'Raw (alpha=.05)':<15} {avg_raw_fd:>15.1f} {'--':>15} {'--':>8}")
avg_bonf_fd = np.mean(bonf_false_discoveries)
avg_bonf_td = np.mean(bonf_true_discoveries)
bonf_fdr = (
    avg_bonf_fd / (avg_bonf_fd + avg_bonf_td) if (avg_bonf_fd + avg_bonf_td) > 0 else 0
)
print(f"{'Bonferroni':<15} {avg_bonf_fd:>15.2f} {avg_bonf_td:>15.2f} {bonf_fdr:>8.3f}")
avg_bh_fd = np.mean(bh_false_discoveries)
avg_bh_td = np.mean(bh_true_discoveries)
bh_fdr = avg_bh_fd / (avg_bh_fd + avg_bh_td) if (avg_bh_fd + avg_bh_td) > 0 else 0
print(f"{'BH-FDR':<15} {avg_bh_fd:>15.2f} {avg_bh_td:>15.2f} {bh_fdr:>8.3f}")

print(
    f"\nRaw: ~{avg_raw_fd:.0f} false positives per round "
    f"(out of {n_sim_tests - n_truly_different} null tests)"
)
print(
    f"Bonferroni: very conservative -- few false positives but "
    f"also misses real effects"
)
print(f"BH-FDR: controls FDR at ~{ALPHA:.0%} while finding more real effects")

# INTERPRETATION: Without correction, we get ~4-5 false positives per
# round. Bonferroni eliminates nearly all false positives but also
# misses many real effects (low power). BH-FDR controls the proportion
# of false discoveries while maintaining better power.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert (
    avg_raw_fd > avg_bonf_fd
), "Bonferroni should have fewer false discoveries than raw"
assert avg_bh_td >= avg_bonf_td, "BH-FDR should find at least as many true effects"
print("\n>>> Checkpoint 5 passed -- FDR simulation validates corrections\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise FDR simulation
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
fig.add_trace(go.Box(y=raw_false_discoveries, name="Raw", marker_color="salmon"))
fig.add_trace(
    go.Box(y=bonf_false_discoveries, name="Bonferroni", marker_color="lightblue")
)
fig.add_trace(go.Box(y=bh_false_discoveries, name="BH-FDR", marker_color="lightgreen"))
fig.update_layout(
    title=f"False Discoveries per Round ({n_sim_tests} tests, "
    f"{n_truly_different} true effects, {n_sim_rounds} rounds)",
    yaxis_title="Number of False Discoveries",
)
out_path = OUTPUT_DIR / "fdr_simulation.html"
fig.write_html(str(out_path))
print(f"Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — A/B test platform at a Singapore fintech
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore fintech company runs experiments on their payment
# app. Each experiment tests 8 metrics: conversion, revenue, time-to-
# complete, error rate, retry rate, customer satisfaction, support
# tickets, and churn risk.
#
# Without correction (8 tests at alpha=0.05):
#   FWER = 1 - (1-0.05)^8 = 33.7%
#   One in three experiments will show a "significant" result that is
#   actually a false positive. Over 50 experiments per quarter, that's
#   ~17 false positives shipped to production.
#
# With BH-FDR correction:
#   At most 5% of declared wins are expected to be false.
#   Over 50 experiments, the team ships ~1-2 false positives instead of 17.
#
# BUSINESS IMPACT: Each false positive shipped means wasted engineering
# effort (building, deploying, monitoring a change that does nothing)
# plus the opportunity cost of not running the next experiment. At an
# estimated S$50K per shipped feature, 15 false positives = S$750K/year
# wasted. BH-FDR correction pays for itself immediately.

fwer_8 = 1 - (1 - ALPHA) ** 8
print(f"\n--- Business Application: Multi-Metric A/B Platform ---")
print(f"8 metrics per experiment, alpha=0.05")
print(f"  Uncorrected FWER: {fwer_8:.1%}")
print(f"  50 experiments/quarter: ~{int(50 * fwer_8 * 0.5)} false positives shipped")
print(f"  With BH-FDR: ~{int(50 * 0.05 * 0.5)} false positives shipped")
print(f"  Annual savings: ~S$750K in wasted engineering effort")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] FWER with Bonferroni: P(ANY false positive) controlled at alpha
  [x] FDR with BH: expected proportion of false discoveries controlled
  [x] q-values: FDR-adjusted p-values for each test
  [x] FDR simulation: empirically verified both corrections work
  [x] When to use which: confirmatory (Bonferroni) vs exploratory (BH-FDR)

  NEXT: In 04_permutation_test.py you'll implement a distribution-free
  alternative that makes no parametric assumptions.
"""
)

print(">>> Exercise 3.3 complete -- Multiple Testing Corrections")
