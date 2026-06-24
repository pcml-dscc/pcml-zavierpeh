# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 4.2: Sample Ratio Mismatch Detection
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Simulate an experiment with a known effect (positive control)
#   - Detect Sample Ratio Mismatch (SRM) via chi-squared test
#   - Simulate SRM to see how biased allocation inflates effect estimates
#   - Connect SRM diagnostics to Singapore ride-hailing A/B tests
#
# PREREQUISITES:
#   - MLFP02 Exercise 4.1 (experiment design, power analysis)
#   - MLFP02 Exercise 3 (chi-squared, hypothesis testing)
#
# ESTIMATED TIME: ~45 minutes
#
# TASKS (5-phase R10):
#   1. Theory — why SRM breaks experiments before they start
#   2. Build — positive control simulation with known treatment effect
#   3. Train — SRM detection on simulated and real data
#   4. Visualise — SRM impact: biased vs unbiased estimation
#   5. Apply — Grab Singapore surge-pricing A/B with device-type SRM
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from shared.mlfp02.ex_4 import (
    ALPHA,
    OUTPUT_DIR,
    SEED,
    TwoArmAB,
    load_experiment,
    make_rng,
    print_banner,
    summarise_arm,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — SRM Breaks Experiments Before They Start
# ════════════════════════════════════════════════════════════════════════
# Sample Ratio Mismatch (SRM) means the observed split between control
# and treatment differs from the designed split (usually 50/50).
#
# Common causes:
#   1. Bot filtering applied asymmetrically
#   2. Redirect latency: treatment page loads slower, users bail
#   3. Randomisation bug: hash collision in user-ID bucketing
#   4. Post-randomisation filter changes who looks "active"
#
# Detection: chi-squared goodness-of-fit test comparing observed vs
# expected group sizes.  Threshold: p < 0.01 (strict, because the
# cost of missing SRM is much higher than investigating it).

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — LOAD data & prepare
# ════════════════════════════════════════════════════════════════════════

print_banner("Exercise 4.2 — SRM Detection")

data: TwoArmAB = load_experiment()
rng = make_rng(SEED)

summarise_arm("Control", data.ctrl_values)
summarise_arm("Treatment", data.treat_values)

sigma_pooled = data.ctrl_values.std(ddof=1)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Positive Control Simulation
# ════════════════════════════════════════════════════════════════════════
# Inject a KNOWN effect into synthetic data.  If we can't detect it,
# the analysis pipeline has a bug.

print_banner("Positive Control Simulation")

sim_n_per = 10_000
true_effect = 2.0
sim_mu_control = data.ctrl_values.mean()

# TODO: Generate sim_control and sim_treatment arrays using rng.normal().
# sim_control has loc=sim_mu_control, scale=sigma_pooled, size=sim_n_per.
# sim_treatment has loc=sim_mu_control + true_effect.
sim_control = ____
sim_treatment = ____

print(f"True control mean:   {sim_mu_control:.4f}")
print(f"True treatment mean: {sim_mu_control + true_effect:.4f}")
print(f"True effect (delta): {true_effect:.4f}")
print(f"n per group:         {sim_n_per:,}")
print(f"\nRealised:")
print(f"  Control mean:   {sim_control.mean():.4f}")
print(f"  Treatment mean: {sim_treatment.mean():.4f}")
print(
    f"  Observed diff:  {sim_treatment.mean() - sim_control.mean():.4f} "
    f"(true: {true_effect})"
)

# TODO: Run Welch's t-test on simulated data using stats.ttest_ind
# with equal_var=False. Returns (t_statistic, p_value).
sim_t, sim_p = ____
print(f"  t-statistic: {sim_t:.4f}, p-value: {sim_p:.2e}")
print(
    f"  {'DETECTED' if sim_p < ALPHA else 'MISSED'} "
    f"(positive control should be detected)"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert sim_p < ALPHA, "Positive control must be detected"
assert (
    abs(sim_treatment.mean() - sim_control.mean() - true_effect) < 3.0
), "Realised effect should be within 3.0 of true"
print("\n>>> Checkpoint 1 passed — positive control validated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: SRM Detection (Simulated & Real)
# ════════════════════════════════════════════════════════════════════════

print_banner("SRM Detection — chi-squared test")

# --- On simulated data (designed 50/50 — should pass) ---
sim_obs = np.array([sim_n_per, sim_n_per])
sim_exp = np.array([sim_n_per, sim_n_per])

# TODO: Run chi-squared goodness-of-fit test using stats.chisquare().
# Pass observed counts and f_exp=expected counts.
sim_chi2, sim_srm_p = ____
print(f"\n--- Simulated (designed 50/50) ---")
print(
    f"chi2={sim_chi2:.4f}, p={sim_srm_p:.6f} -> "
    f"{'SRM' if sim_srm_p < 0.01 else 'OK'}"
)

# --- On real data ---
real_obs = np.array([data.n_control, data.n_treatment])
real_exp = np.array([data.n_total / 2, data.n_total / 2])

# TODO: Same chi-squared test on real data.
chi2_stat, srm_p = ____
print(f"\n--- Real Data (intended 50/50) ---")
print(
    f"Observed: {data.n_control:,} / {data.n_treatment:,} "
    f"({data.n_control / data.n_total:.4f}/{data.n_treatment / data.n_total:.4f})"
)
print(f"chi2={chi2_stat:.4f}, p={srm_p:.2e}")
if srm_p < 0.01:
    print("SRM DETECTED — investigate:")
    print("  1. Bot filtering differential")
    print("  2. Technical redirect issues")
    print("  3. Randomisation bug")
    print("  4. Post-randomisation population filter")
else:
    print("No SRM detected — split is consistent with design")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert sim_chi2 == 0.0, "Simulated equal groups should give chi2=0"
assert 0 <= srm_p <= 1, "SRM p-value must be valid"
print("\n>>> Checkpoint 2 passed — SRM detection completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3b — Simulating SRM Impact on Estimation
# ════════════════════════════════════════════════════════════════════════
# What happens when high-value users are systematically more likely
# to end up in treatment?

print_banner("Simulating SRM — Impact on Estimation")

n_srm_sim = 1000
srm_biases = []
no_srm_biases = []
true_lift = 1.0

for _ in range(n_srm_sim):
    # No SRM: random 50/50 allocation
    users = rng.normal(50, 10, size=2000)
    assigned = rng.choice([0, 1], size=2000)
    ctrl_outcome = users[assigned == 0] + rng.normal(0, 5, size=(assigned == 0).sum())
    treat_outcome = (
        users[assigned == 1] + true_lift + rng.normal(0, 5, size=(assigned == 1).sum())
    )
    no_srm_biases.append((treat_outcome.mean() - ctrl_outcome.mean()) - true_lift)

    # TODO: Simulate WITH SRM — high-value users (quality > 55) have 60%
    # probability of being assigned to treatment, others 40%.
    # Hint: assign_prob = np.where(users > 55, 0.6, 0.4)
    assign_prob = ____
    assigned_srm = (rng.random(size=2000) < assign_prob).astype(int)
    ctrl_out_srm = users[assigned_srm == 0] + rng.normal(
        0, 5, size=(assigned_srm == 0).sum()
    )
    treat_out_srm = (
        users[assigned_srm == 1]
        + true_lift
        + rng.normal(0, 5, size=(assigned_srm == 1).sum())
    )
    srm_biases.append((treat_out_srm.mean() - ctrl_out_srm.mean()) - true_lift)

print(f"True treatment effect: {true_lift}")
print(f"\nNo SRM:")
print(f"  Mean estimation bias: {np.mean(no_srm_biases):+.4f}")
print(f"  Std of bias: {np.std(no_srm_biases):.4f}")
print(f"\nWith SRM (high-value -> treatment):")
print(f"  Mean estimation bias: {np.mean(srm_biases):+.4f}")
print(f"  Std of bias: {np.std(srm_biases):.4f}")
bias_gap = np.mean(srm_biases) - np.mean(no_srm_biases)
print(f"\nSRM adds a POSITIVE bias of ~{bias_gap:+.2f}")
print("because high-value users inflate treatment outcomes.")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert abs(np.mean(no_srm_biases)) < 0.5, "No-SRM bias should be near zero"
assert abs(np.mean(srm_biases)) > abs(
    np.mean(no_srm_biases)
), "SRM should introduce larger bias"
print("\n>>> Checkpoint 3 passed — SRM impact demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: SRM Bias Distributions
# ════════════════════════════════════════════════════════════════════════

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["No SRM (unbiased)", "With SRM (biased)"],
)
fig.add_trace(
    go.Histogram(x=no_srm_biases, nbinsx=40, name="No SRM", marker_color="#4CAF50"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Histogram(x=srm_biases, nbinsx=40, name="SRM", marker_color="#F44336"),
    row=1,
    col=2,
)
fig.update_layout(
    title="SRM Impact on Treatment Effect Estimation Bias",
    height=400,
    template="plotly_white",
)
out_path = OUTPUT_DIR / "srm_simulation.html"
fig.write_html(str(out_path))
print(f"Saved: {out_path}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert out_path.exists(), "SRM simulation plot must be saved"
print("\n>>> Checkpoint 4 passed — SRM visualisation saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Grab Singapore Surge Pricing A/B
# ════════════════════════════════════════════════════════════════════════
# Grab tests a new surge-pricing algorithm.  A device-type SRM appears:
# iOS users are 5% more likely to land in treatment.

print_banner("Applied — Grab Singapore Surge Pricing A/B")

n_grab = 20_000
ios_share = 0.45

# Simulate device-type SRM
grab_devices = rng.choice(
    ["iOS", "Android"],
    size=n_grab,
    p=[ios_share, 1 - ios_share],
)
# Biased assignment: iOS users 55% likely to get treatment
assign_prob_grab = np.where(grab_devices == "iOS", 0.55, 0.45)
grab_assignment = (rng.random(n_grab) < assign_prob_grab).astype(int)

# Fare depends on device (iOS users have higher AOV)
base_fare = np.where(grab_devices == "iOS", 18.0, 12.0)
true_algo_effect = 0.5
grab_fare = (
    base_fare + grab_assignment * true_algo_effect + rng.normal(0, 4, size=n_grab)
)

# TODO: Run the SRM chi-squared test on the Grab data.
# Compute n_ctrl_grab and n_treat_grab from grab_assignment.
n_ctrl_grab = ____
n_treat_grab = ____
grab_obs = np.array([n_ctrl_grab, n_treat_grab])
grab_exp = np.array([n_grab / 2, n_grab / 2])
grab_chi2, grab_srm_p = ____

naive_effect = (
    grab_fare[grab_assignment == 1].mean() - grab_fare[grab_assignment == 0].mean()
)

print(f"n = {n_grab:,}  |  True algo effect: ${true_algo_effect:.2f}")
print(f"Control: {n_ctrl_grab:,}  Treatment: {n_treat_grab:,}")
print(f"SRM chi2={grab_chi2:.2f}, p={grab_srm_p:.4f}")
print(f"SRM {'DETECTED' if grab_srm_p < 0.01 else 'not detected'}")
print(f"\nNaive estimated effect:  ${naive_effect:.2f}")
print(f"True effect:             ${true_algo_effect:.2f}")
print(f"Bias from SRM:           ${naive_effect - true_algo_effect:+.2f}")
print(
    "\nThe iOS enrichment in treatment inflates the fare estimate.\n"
    "At Grab's scale (~8M rides/month), this bias would cause\n"
    "incorrect pricing decisions worth millions in revenue."
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert n_grab == n_ctrl_grab + n_treat_grab, "All users accounted for"
print("\n>>> Checkpoint 5 passed — Grab SRM scenario completed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - Positive controls: inject known effect to validate pipeline
  - SRM detection: chi-squared test on observed vs expected split
  - SRM causes: bot filtering, redirects, randomisation bugs
  - SRM impact: biased allocation => biased treatment effect estimates
  - Applied: Grab Singapore ride-hailing device-type SRM

  NEXT: In Exercise 4.3 you'll learn Welch's t-test — the robust
  test for comparing means when variances may differ.
"""
)

print(">>> Exercise 4.2 complete — SRM Detection")
