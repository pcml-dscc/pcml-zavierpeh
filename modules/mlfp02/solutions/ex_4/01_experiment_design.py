# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 4.1: Experiment Design & Power Analysis
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Design a complete A/B experiment with pre-registered hypotheses
#   - Compute required sample sizes via power analysis (normal approx.)
#   - Build a power curve showing how detection probability scales with n
#   - Connect sample-size planning to real Singapore e-commerce decisions
#
# PREREQUISITES:
#   - MLFP02 Exercise 3 (hypothesis testing, p-values, power)
#
# ESTIMATED TIME: ~40 minutes
#
# TASKS (5-phase R10):
#   1. Theory — pre-registration prevents p-hacking
#   2. Build — power analysis: required n for a target MDE
#   3. Train — power curve: power vs sample size
#   4. Visualise — interactive power curve with required-n marker
#   5. Apply — Shopee Singapore checkout-flow A/B design
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from shared.mlfp02.ex_4 import (
    ALPHA,
    DESIGN_MDE_PCT,
    OUTPUT_DIR,
    POWER_TARGET,
    TwoArmAB,
    cohens_d,
    load_experiment,
    power_at_n,
    print_banner,
    required_n_per_group,
    summarise_arm,
    z_critical,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Pre-Registration Prevents p-Hacking
# ════════════════════════════════════════════════════════════════════════
# An experiment whose hypothesis is decided AFTER looking at the data is
# not an experiment — it is HARKing (Hypothesising After Results are
# Known).  Pre-registration commits to:
#
#   1. The hypothesis  (H0 vs H1)
#   2. The primary metric
#   3. The significance level and power
#   4. The stopping rule (when to analyse)
#   5. Any planned corrections for multiplicity
#
# The point is simple: you cannot shoot the arrow then paint the target.
#
# SINGAPORE CONTEXT:
#   PDPA (Personal Data Protection Act) requires purpose limitation —
#   data collected for an experiment must be pre-declared.
#   Pre-registration satisfies this by documenting intent before
#   collection.

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — LOAD experiment data & formulate hypothesis
# ════════════════════════════════════════════════════════════════════════

print_banner("Exercise 4.1 — Experiment Design & Power Analysis")

data: TwoArmAB = load_experiment()

print(f"\nData loaded: experiment_data.parquet")
print(f"Shape: {data.experiment.shape}")
print(f"Columns: {data.experiment.columns}")
print(data.experiment.head(5))

# Group structure
group_counts = (
    data.experiment["experiment_group"].value_counts().sort("experiment_group")
)
print(f"\n--- Group Allocation ---")
print(group_counts)

# Two-arm summary
print(f"\n=== Two-Arm A/B Subset ===")
summarise_arm("Control", data.ctrl_values)
summarise_arm("Treatment", data.treat_values)
print(f"  Total n: {data.n_total:,}")

# Pre-registration document
print(
    """
PRE-REGISTRATION DOCUMENT:
═══════════════════════════
1. Primary hypothesis:
   H0: mu_treatment = mu_control  (no effect on engagement)
   H1: mu_treatment != mu_control (two-sided)

2. Primary metric: metric_value (engagement score)

3. Design parameters:
   - Significance level (alpha): 0.05 (5% false positive rate)
   - Power (1-beta): 0.80 (80% chance of detecting a real effect)
   - Randomisation unit: user
   - Allocation: equal (50/50)

4. Stopping rule: analyse only after target n is reached.
   No peeking (see Exercise 4.4 for sequential testing).

5. Pre-registered corrections: Bonferroni for 3 secondary metrics.

Key principles:
  - Randomisation eliminates confounders
  - Equal allocation maximises power for given total n
  - Pre-registration prevents p-hacking and HARKing
  - Single primary metric reduces multiple testing burden
"""
)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Power Analysis (Required Sample Size)
# ════════════════════════════════════════════════════════════════════════
# The sample-size formula (normal approximation, two-sample) is:
#
#   n = (z_{alpha/2} + z_beta)^2 * 2 * sigma^2 / delta^2
#
# where delta is the minimum detectable effect (MDE) in absolute units.
# Smaller MDE => need more data.  Higher power => need more data.

print_banner("Power Analysis — Required Sample Size")

sigma_pooled = data.ctrl_values.std(ddof=1)
ctrl_mean = data.ctrl_values.mean()

z_a, z_b = z_critical(ALPHA, POWER_TARGET)
print(f"z_{{alpha/2}} = {z_a:.4f},  z_beta = {z_b:.4f}")
print(f"Baseline mean: {ctrl_mean:.4f},  sigma_pooled: {sigma_pooled:.4f}")

# Compute required n for several MDE levels
print(
    f"\n{'Relative MDE':>14} {'Absolute MDE':>14} {'n per group':>14} {'n total':>10}"
)
print("-" * 56)

n_required_results: dict[float, dict] = {}
for rel_mde_pct in [1.0, 2.0, 3.0, 5.0, 10.0]:
    mde_abs = ctrl_mean * (rel_mde_pct / 100)
    d = cohens_d(mde_abs, sigma_pooled)
    n_per = required_n_per_group(sigma_pooled, mde_abs, ALPHA, POWER_TARGET)
    n_required_results[rel_mde_pct] = {"mde": mde_abs, "n_per": n_per, "d": d}
    print(f"{rel_mde_pct:>12.1f}%  {mde_abs:>14.4f}  {n_per:>14,}  {2 * n_per:>10,}")

# Design target: 2% relative MDE
mde_absolute = n_required_results[DESIGN_MDE_PCT]["mde"]
n_required_per = n_required_results[DESIGN_MDE_PCT]["n_per"]
d_design = n_required_results[DESIGN_MDE_PCT]["d"]

print(f"\nDesign target: {DESIGN_MDE_PCT}% relative MDE = {mde_absolute:.4f} absolute")
print(
    f"Cohen's d: {d_design:.4f} "
    f"({'small' if d_design < 0.2 else 'small-medium' if d_design < 0.5 else 'medium'})"
)
print(f"Required: {n_required_per:,} per group, {2 * n_required_per:,} total")
print(f"Actual: {data.n_total:,} ({data.n_total / (2 * n_required_per):.1f}x required)")
if data.n_total >= 2 * n_required_per:
    print("-> ADEQUATELY POWERED")
else:
    print("-> UNDERPOWERED — need more observations or accept larger MDE")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert n_required_per > 0, "Required sample size must be positive"
assert d_design > 0, "Cohen's d must be positive"
print("\n>>> Checkpoint 1 passed — power analysis completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Power Curve (Power vs Sample Size)
# ════════════════════════════════════════════════════════════════════════
# How does detection probability change as we increase sample size?

print_banner("Power Curve")

sample_sizes = np.arange(500, n_required_per * 3, max(500, n_required_per // 20))
power_values = [
    power_at_n(int(ns), sigma_pooled, mde_absolute, ALPHA) for ns in sample_sizes
]

print(f"Power at selected sample sizes (per group, MDE={mde_absolute:.4f}):")
for frac in [0.25, 0.5, 1.0, 1.5, 2.0]:
    idx = int(np.argmin(np.abs(sample_sizes - n_required_per * frac)))
    print(f"  n = {sample_sizes[idx]:>8,}: power = {power_values[idx]:.1%}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert power_values[-1] > power_values[0], "Power should increase with n"
print("\n>>> Checkpoint 2 passed — power curve computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Interactive Power Curve
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=sample_sizes.tolist(),
        y=power_values,
        mode="lines",
        name="Power",
        line={"color": "#2196F3", "width": 2},
    )
)
fig.add_hline(y=0.8, line_dash="dash", annotation_text="80% target")
fig.add_vline(
    x=n_required_per,
    line_dash="dot",
    annotation_text=f"Required n={n_required_per:,}",
)
fig.update_layout(
    title=f"Power Curve (MDE={mde_absolute:.2f}, alpha={ALPHA})",
    xaxis_title="Sample Size per Group",
    yaxis_title="Power (probability of detecting true effect)",
    template="plotly_white",
)
out_path = OUTPUT_DIR / "power_curve.html"
fig.write_html(str(out_path))
print(f"Saved: {out_path}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert out_path.exists(), "Power curve plot must be saved"
print("\n>>> Checkpoint 3 passed — power curve visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Shopee Singapore Checkout Flow A/B
# ════════════════════════════════════════════════════════════════════════
# Shopee wants to test a simplified checkout flow.
# Current conversion rate: 8.2%.  Target MDE: +0.5 pp (absolute).
# Use our power analysis primitives to size the experiment.

print_banner("Applied — Shopee Singapore Checkout A/B")

baseline_rate = 0.082
target_mde_pp = 0.005  # 0.5 percentage point
# For proportions: sigma ~ sqrt(p * (1-p))
sigma_prop = math.sqrt(baseline_rate * (1 - baseline_rate))
n_shopee = required_n_per_group(sigma_prop, target_mde_pp, ALPHA, POWER_TARGET)

print(f"Baseline conversion: {baseline_rate:.1%}")
print(f"Target MDE: {target_mde_pp:.1%} (absolute)")
print(f"sigma (Bernoulli): {sigma_prop:.4f}")
print(f"Required: {n_shopee:,} per group, {2 * n_shopee:,} total")
print(f"\nAt 50,000 daily users, this takes ~{2 * n_shopee / 50_000:.0f} days.")
print(
    f"Business decision: if 0.5 pp lift = +${0.005 * 50_000 * 365 * 15:.0f}/yr revenue"
)
print(f"at $15 AOV, the experiment cost ({2 * n_shopee / 50_000:.0f} days of risk)")
print(f"is worth it.")
# INTERPRETATION: Even a tiny absolute lift in conversion rates has
# enormous dollar-value impact at Shopee's scale. The power analysis
# tells you exactly how long to wait before drawing conclusions.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert n_shopee > 0, "Shopee required n must be positive"
print("\n>>> Checkpoint 4 passed — applied scenario completed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - Pre-registration: hypotheses, metrics, stopping rules BEFORE data
  - Power analysis formula: n = f(alpha, power, MDE, sigma)
  - Power curve: visualising detection probability vs sample size
  - Cohen's d: standardised effect size for comparing experiments
  - Applied sizing: Shopee Singapore checkout conversion experiment

  NEXT: In Exercise 4.2 you'll learn SRM detection — the first sanity
  check every experiment must pass before interpreting results.
"""
)

print(">>> Exercise 4.1 complete — Experiment Design & Power Analysis")
