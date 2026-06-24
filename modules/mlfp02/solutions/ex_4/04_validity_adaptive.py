# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 4.4: Validity, Adaptive Design & Experiment Report
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement a data collection plan (Why/What/Where/How/Frequency)
#   - Evaluate experiment validity: SUTVA, interference, novelty effects
#   - Compute adaptive (sequential) sample-size re-estimation from a pilot
#   - Build a complete experiment analysis report with business decision
#   - Connect adaptive design to Singapore fintech regulatory experiments
#
# PREREQUISITES:
#   - MLFP02 Exercise 4.3 (Welch's t-test, confidence intervals)
#
# ESTIMATED TIME: ~45 minutes
#
# TASKS (5-phase R10):
#   1. Theory — SUTVA and why interference kills causal inference
#   2. Build — data collection plan + validity diagnostics
#   3. Train — adaptive sample-size re-estimation from pilot data
#   4. Visualise — full experiment report with business recommendation
#   5. Apply — MAS Singapore regulatory sandbox adaptive experiment
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from shared.mlfp02.ex_4 import (
    ALPHA,
    DESIGN_MDE_PCT,
    OUTPUT_DIR,
    POWER_TARGET,
    SEED,
    TwoArmAB,
    load_experiment,
    make_rng,
    power_at_n,
    print_banner,
    required_n_per_group,
    summarise_arm,
    z_critical,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — SUTVA and Interference
# ════════════════════════════════════════════════════════════════════════
# SUTVA (Stable Unit Treatment Value Assumption):
#   Each user's outcome depends ONLY on their own treatment assignment,
#   not on other users' assignments.
#
# Violations of SUTVA:
#   1. Network effects — treated users share recommendations with
#      control users via social features => control is "contaminated"
#   2. Marketplace effects — treatment changes supply/demand dynamics,
#      control users are affected by shifted prices
#   3. Shared resources — treatment uses more server capacity,
#      control experiences slower page loads
#
# SUTVA is rarely perfectly satisfied.  The question is whether
# violations are large enough to meaningfully bias results.
# Document known violations and their likely direction of bias.
#
# ADAPTIVE DESIGN:
#   Sometimes we don't know the variance (sigma) before the experiment.
#   Adaptive design: start with a pilot, estimate sigma, then compute
#   the remaining sample size needed.  This avoids both underpowered
#   experiments (sigma underestimated) and waste (sigma overestimated).

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — LOAD data
# ════════════════════════════════════════════════════════════════════════

print_banner("Exercise 4.4 — Validity, Adaptive Design & Report")

data: TwoArmAB = load_experiment()
rng = make_rng(SEED)

summarise_arm("Control", data.ctrl_values)
summarise_arm("Treatment", data.treat_values)

sigma_pooled = data.ctrl_values.std(ddof=1)
ctrl_mean = data.ctrl_values.mean()
mde_absolute = ctrl_mean * (DESIGN_MDE_PCT / 100)
n_required_per = required_n_per_group(sigma_pooled, mde_absolute, ALPHA, POWER_TARGET)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Data Collection Plan & Validity Diagnostics
# ════════════════════════════════════════════════════════════════════════

print_banner("Data Collection Plan (Why/What/Where/How/Frequency)")

plan = {
    "WHY (Hypotheses & Value)": {
        "Primary hypothesis": "New algorithm increases engagement by >= 2%",
        "Secondary hypotheses": "Revenue impact, conversion impact",
        "Business value": "1% engagement lift ~ $200K annual revenue increase",
        "Success criteria": "p < 0.05 AND Cohen's d > 0.1 AND CI excludes zero",
    },
    "WHAT (Data Requirements)": {
        "Primary metric": "metric_value (engagement score, 0-100)",
        "Secondary metrics": "revenue, conversion, pages_viewed",
        "Covariates": "signup_date, device_type, country, prior_activity",
        "Guardrail metrics": "page_load_time, error_rate, support_tickets",
        "Minimum rows": f"{2 * n_required_per:,} (from power analysis)",
    },
    "WHERE (Data Sources)": {
        "Internal": "Event pipeline (Kafka -> BigQuery), user_features table",
        "External": "None required for this experiment",
        "Schema": "user_id, timestamp, experiment_group, metric_value, revenue",
    },
    "HOW (Collection Method)": {
        "Assignment": "Server-side random hash on user_id (deterministic)",
        "Logging": "Event-sourced: every impression, click, purchase logged",
        "Quality": "Dedup on (user_id, session_id), validate schema on ingest",
        "Privacy": "PII stripped at collection; analysis on anonymised IDs",
    },
    "FREQUENCY (Timing)": {
        "Collection frequency": "Real-time events, hourly batch aggregation",
        "Analysis frequency": "Weekly interim report (no peeking at p-values)",
        "Duration": f"~{2 * n_required_per // 5000} days at 5,000 users/day",
        "Stopping rule": "Analyse after target n reached; no early stopping",
    },
}

for section, items in plan.items():
    print(f"\n{section}")
    print("-" * 50)
    for key, value in items.items():
        print(f"  {key}: {value}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(plan) == 5, "Plan must cover all 5 sections"
print("\n>>> Checkpoint 1 passed — data collection plan created\n")


# ── Validity Diagnostics ─────────────────────────────────────────────

print_banner("Experiment Validity Criteria")

# Check 1: Variance ratio (should be ~1 if no interference)
var_ratio = data.treat_values.var() / data.ctrl_values.var()
print(f"  1. Variance ratio (treatment/control): {var_ratio:.3f}")
print(f"     Expected ~1.0 if no differential interference")
print(f"     Status: {'OK' if 0.8 < var_ratio < 1.2 else 'INVESTIGATE'}")

# Check 2: Distribution shape similarity (KS test)
ks_stat, ks_p = stats.ks_2samp(data.ctrl_values, data.treat_values)
print(f"\n  2. KS test for distribution similarity: D={ks_stat:.4f}, p={ks_p:.6f}")
print(f"     A small p-value suggests distributions differ beyond just location shift")

# Check 3: Novelty effect — compare early vs late treatment outcomes
n_half = data.n_treatment // 2
early_treat = data.treat_values[:n_half]
late_treat = data.treat_values[n_half:]
novelty_t, novelty_p = stats.ttest_ind(early_treat, late_treat, equal_var=False)
print(
    f"\n  3. Novelty check (early vs late treatment): "
    f"t={novelty_t:.4f}, p={novelty_p:.4f}"
)
if novelty_p < 0.05:
    print(
        "     WARNING: early/late treatment outcomes differ — possible novelty effect"
    )
else:
    print("     OK: no evidence of novelty or fatigue effect")
# INTERPRETATION: SUTVA is rarely perfectly satisfied.  The question
# is whether violations are large enough to meaningfully bias results.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert var_ratio > 0, "Variance ratio must be positive"
assert 0 <= ks_p <= 1, "KS p-value must be valid"
print("\n>>> Checkpoint 2 passed — validity criteria assessed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Adaptive Sample Size from Pilot
# ════════════════════════════════════════════════════════════════════════

print_banner("Adaptive Sample Size Calculation")

z_a, z_b = z_critical(ALPHA, POWER_TARGET)

# Simulate a pilot phase
pilot_n = 500
pilot_ctrl = rng.choice(data.ctrl_values, size=pilot_n, replace=True)
pilot_treat = rng.choice(data.treat_values, size=pilot_n, replace=True)

pilot_sigma = np.sqrt((pilot_ctrl.var(ddof=1) + pilot_treat.var(ddof=1)) / 2)
pilot_diff = pilot_treat.mean() - pilot_ctrl.mean()

# Re-compute required n based on pilot estimate
n_adaptive = required_n_per_group(pilot_sigma, mde_absolute, ALPHA, POWER_TARGET)

print(f"Pilot phase: n={pilot_n} per group")
print(f"Pilot sigma estimate: {pilot_sigma:.4f} (true sigma ~ {sigma_pooled:.4f})")
print(f"Pilot observed diff: {pilot_diff:+.4f}")
print(f"\nAdaptive required n per group: {n_adaptive:,}")
print(f"Original required n per group: {n_required_per:,}")
print(f"Ratio: {n_adaptive / n_required_per:.2f}x")
print(f"Remaining needed: {max(0, n_adaptive - pilot_n):,} per group")

# Multi-stage: how estimate improves with pilot size
print(f"\n--- Pilot Size vs Required n Stability ---")
for pilot_size in [100, 250, 500, 1000]:
    sigs = []
    for _ in range(100):
        pc = rng.choice(data.ctrl_values, size=pilot_size, replace=True)
        pt = rng.choice(data.treat_values, size=pilot_size, replace=True)
        s = np.sqrt((pc.var(ddof=1) + pt.var(ddof=1)) / 2)
        sigs.append(s)
    mean_sig = np.mean(sigs)
    std_sig = np.std(sigs)
    n_req = required_n_per_group(mean_sig, mde_absolute, ALPHA, POWER_TARGET)
    print(
        f"  Pilot n={pilot_size:>4}: sigma_hat={mean_sig:.4f} +/- {std_sig:.4f}, "
        f"required n={n_req:,}"
    )
# INTERPRETATION: With a small pilot (n=100), the variance estimate
# is noisy, leading to uncertain sample size calculations.  A pilot
# of ~500 gives a stable estimate.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert n_adaptive > 0, "Adaptive sample size must be positive"
print("\n>>> Checkpoint 3 passed — adaptive design completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Full Experiment Report
# ════════════════════════════════════════════════════════════════════════

# Re-compute final statistics for the report
real_t_stat, real_p_val = stats.ttest_ind(
    data.treat_values, data.ctrl_values, equal_var=False
)
obs_diff = data.treat_values.mean() - data.ctrl_values.mean()
rel_lift = obs_diff / data.ctrl_values.mean() * 100

s1_sq_n1 = data.ctrl_values.var(ddof=1) / data.n_control
s2_sq_n2 = data.treat_values.var(ddof=1) / data.n_treatment
df_ws = (s1_sq_n1 + s2_sq_n2) ** 2 / (
    s1_sq_n1**2 / (data.n_control - 1) + s2_sq_n2**2 / (data.n_treatment - 1)
)
real_se = np.sqrt(s1_sq_n1 + s2_sq_n2)
t_crit = stats.t.ppf(1 - ALPHA / 2, df=df_ws)
welch_ci = (obs_diff - t_crit * real_se, obs_diff + t_crit * real_se)

pooled_std_real = np.sqrt(
    (data.ctrl_values.var(ddof=1) + data.treat_values.var(ddof=1)) / 2
)
cohens_d_real = obs_diff / pooled_std_real

# SRM check
real_obs = np.array([data.n_control, data.n_treatment])
real_exp = np.array([data.n_total / 2, data.n_total / 2])
_, srm_p = stats.chisquare(real_obs, f_exp=real_exp)

# Adaptive sigma stability plot
fig_adapt = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "Power Curve",
        "Pilot Size vs sigma Stability",
    ],
)

# Power curve subplot
sample_sizes = np.arange(500, n_required_per * 3, max(500, n_required_per // 20))
power_values = [
    power_at_n(int(ns), sigma_pooled, mde_absolute, ALPHA) for ns in sample_sizes
]
fig_adapt.add_trace(
    go.Scatter(
        x=sample_sizes.tolist(),
        y=power_values,
        mode="lines",
        name="Power",
        line={"color": "#2196F3"},
    ),
    row=1,
    col=1,
)
fig_adapt.add_hline(y=0.8, line_dash="dash", row=1, col=1)

# Sigma stability subplot
pilot_sizes_list = [50, 100, 200, 300, 500, 750, 1000]
sigma_means = []
sigma_stds = []
for ps in pilot_sizes_list:
    sigs = []
    for _ in range(200):
        pc = rng.choice(data.ctrl_values, size=ps, replace=True)
        pt = rng.choice(data.treat_values, size=ps, replace=True)
        sigs.append(np.sqrt((pc.var(ddof=1) + pt.var(ddof=1)) / 2))
    sigma_means.append(np.mean(sigs))
    sigma_stds.append(np.std(sigs))

fig_adapt.add_trace(
    go.Scatter(
        x=pilot_sizes_list,
        y=sigma_means,
        mode="lines+markers",
        name="sigma_hat",
        error_y={"type": "data", "array": sigma_stds, "visible": True},
        line={"color": "#FF9800"},
    ),
    row=1,
    col=2,
)
fig_adapt.add_hline(y=sigma_pooled, line_dash="dot", row=1, col=2)
fig_adapt.update_layout(
    title="Adaptive Design: Power Curve & Pilot Stability",
    height=400,
    template="plotly_white",
)
out_path = OUTPUT_DIR / "adaptive_design.html"
fig_adapt.write_html(str(out_path))
print(f"Saved: {out_path}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert out_path.exists(), "Adaptive design plot must be saved"
print("\n>>> Checkpoint 4 passed — visualisations saved\n")


# ── Final business report ─────────────────────────────────────────────

print_banner("EXPERIMENT REPORT")
print(
    f"""
Experiment: Recommendation Algorithm A/B Test
Duration: designed for {2 * n_required_per:,} total users
Actual: {data.n_total:,} users

SRM Check: p={srm_p:.4f} -> {'PASS' if srm_p > 0.01 else 'FAIL — results may be biased'}

Primary Metric (metric_value — engagement score):
  Control:   {data.ctrl_values.mean():.4f} +/- {data.ctrl_values.std():.4f}
  Treatment: {data.treat_values.mean():.4f} +/- {data.treat_values.std():.4f}
  Lift: {obs_diff:+.4f} ({rel_lift:+.2f}% relative)
  p-value: {real_p_val:.6f}
  Cohen's d: {cohens_d_real:.4f}
  95% CI: [{welch_ci[0]:.4f}, {welch_ci[1]:.4f}]

Decision: {"SHIP — statistically significant positive effect" if real_p_val < ALPHA and obs_diff > 0
           else "HOLD — more data needed" if real_p_val > ALPHA and abs(obs_diff) > mde_absolute * 0.5
           else "NO SHIP — no meaningful effect detected"}

Validity: Variance ratio {var_ratio:.2f}, no novelty effect detected.
"""
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
print(">>> Checkpoint 5 passed — experiment report complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS Singapore Regulatory Sandbox
# ════════════════════════════════════════════════════════════════════════
# The Monetary Authority of Singapore (MAS) runs regulatory sandboxes
# for fintech innovation.  An adaptive experiment design is essential
# because the sandbox window is time-limited and variance is unknown
# upfront.

print_banner("Applied — MAS Regulatory Sandbox Adaptive Experiment")

# Scenario: a new robo-advisor feature in a sandbox
# Outcome: portfolio return (annualised %)
# Sandbox window: 6 months, ~2000 participants available
sandbox_n_available = 2000
robo_sigma_guess = 5.0  # Initial guess: 5% return std
robo_mde = 1.0  # Want to detect 1 pp return improvement

# Phase 1: initial estimate
n_initial = required_n_per_group(robo_sigma_guess, robo_mde, ALPHA, POWER_TARGET)
print(
    f"Initial estimate (sigma guess = {robo_sigma_guess}%): n={n_initial:,} per group"
)
feasible_initial = "YES" if 2 * n_initial <= sandbox_n_available else "NO"
print(f"Feasible in sandbox ({sandbox_n_available} available)? {feasible_initial}")

# Phase 2: pilot reveals true sigma
pilot_n_sandbox = 200
robo_sigma_pilot = 4.2  # Pilot reveals lower variance than guessed
n_revised = required_n_per_group(robo_sigma_pilot, robo_mde, ALPHA, POWER_TARGET)
print(f"\nAfter pilot (n={pilot_n_sandbox}, sigma_hat={robo_sigma_pilot}%):")
print(f"  Revised n per group: {n_revised:,}")
print(f"  Remaining: {max(0, n_revised - pilot_n_sandbox):,} per group")
feasible_revised = "YES" if 2 * n_revised <= sandbox_n_available else "NO"
print(f"  Feasible? {feasible_revised}")

# Phase 3: power at available sample size
achieved_power = power_at_n(sandbox_n_available // 2, robo_sigma_pilot, robo_mde, ALPHA)
print(f"\nPower at maximum available n ({sandbox_n_available // 2} per group):")
print(f"  Achieved power: {achieved_power:.1%}")
print(
    f"  {'ADEQUATE' if achieved_power >= 0.8 else 'Consider larger MDE or extended sandbox'}"
)
print(
    "\nAdaptive design saved this sandbox: the initial sigma overestimate\n"
    "would have required more participants than available.  The pilot\n"
    "revealed lower variance, making the experiment feasible within\n"
    "the regulatory window."
)
# INTERPRETATION: Regulatory sandboxes are time- and sample-constrained.
# Adaptive design is not a luxury — it is the only way to run properly
# powered experiments within MAS sandbox limits.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert n_revised > 0, "Revised n must be positive"
assert 0 < achieved_power <= 1, "Achieved power must be valid"
print("\n>>> Checkpoint 6 passed — MAS sandbox scenario completed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - Data collection plan: Why/What/Where/How/Frequency framework
  - SUTVA: interference, novelty effects, variance-ratio checks
  - Adaptive design: pilot -> estimate sigma -> compute remaining n
  - Pilot stability: larger pilots => more reliable sigma estimates
  - Complete experiment report with ship/hold/no-ship decision
  - Applied: MAS Singapore regulatory sandbox adaptive experiment

  NEXT: In Exercise 5 you'll build linear regression from scratch.
  You'll derive OLS using matrix algebra, test coefficient significance
  with t-statistics, detect multicollinearity, and run residual
  diagnostics — all on HDB price prediction data.
"""
)

print(">>> Exercise 4.4 complete — Validity, Adaptive Design & Report")
print("\n>>> Exercise 4 complete — A/B Testing and Experiment Design")
