# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 1.2: Bayes' Theorem — From Medical Tests to
#                         Property Valuation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Apply Bayes' theorem P(A|B) = P(B|A)P(A)/P(B) to real scenarios
#   - Understand the base rate fallacy — why a 99.5% specificity test
#     still produces many false positives at low prevalence
#   - Sweep prevalence to see how posterior probability changes
#   - Apply Bayesian reasoning to Singapore HDB property valuation
#   - Quantify dollar impact of ignoring base rates in decision-making
#
# PREREQUISITES: Complete 01_probability_mle.py (joint/conditional probs)
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — why Bayes' theorem matters for real-world decisions
#   2. Build — COVID ART test: sensitivity, specificity, prevalence
#   3. Train — prevalence sweep shows base rate fallacy in action
#   4. Visualise — posterior probability vs prevalence curve
#   5. Apply — HDB Bishan valuation: is a $650K listing overpriced?
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from shared.mlfp02.ex_1 import (
    OUTPUT_DIR,
    fmt_money,
    load_hdb_4room,
    load_hdb_prices_4room,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Bayes' Theorem Matters for Real-World Decisions
# ════════════════════════════════════════════════════════════════════════
# Bayes' theorem:
#   P(A|B) = P(B|A) × P(A) / P(B)
#
# In plain language: to find the probability of A given that B happened,
# you need THREE things:
#   1. P(B|A) — the likelihood (how likely is B if A is true?)
#   2. P(A) — the prior (how common is A before any observation?)
#   3. P(B) — the marginal (how common is B overall?)
#
# The base rate fallacy: people overweight the test result and underweight
# the base rate P(A). A COVID test with 99.5% specificity sounds nearly
# perfect, but when prevalence is only 2%, most positive results are false
# positives. This same error affects:
#   - Fraud detection (low base rate of actual fraud)
#   - Medical screening (low prevalence of disease)
#   - Property valuation (low base rate of "overpriced" listings)
#
# The cure: always ask "how common is the condition in the first place?"


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: COVID ART Test — Sensitivity, Specificity, Prevalence
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP02 Exercise 1.2: Bayes' Theorem Applications")
print("=" * 70)

# COVID ART test parameters (Singapore community testing context)
sensitivity = 0.85  # P(positive test | infected)
specificity = 0.995  # P(negative test | not infected)
prevalence = 0.02  # P(infected) — base rate in Singapore community

# P(positive test) = P(+|infected)P(infected) + P(+|not infected)P(not infected)
p_positive = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)

# P(infected | positive test)
p_infected_given_positive = (sensitivity * prevalence) / p_positive

# P(not infected | positive test) — false positive rate among positives
p_false_positive = 1 - p_infected_given_positive

print(f"\n=== Bayes' Theorem: COVID ART Test ===")
print(f"Sensitivity: {sensitivity:.1%} — P(+test | infected)")
print(f"Specificity: {specificity:.1%} — P(-test | not infected)")
print(f"Prevalence:  {prevalence:.1%} — P(infected)")
print(f"")
print(f"P(positive test)           = {p_positive:.4f} ({p_positive:.2%})")
print(
    f"P(infected | positive test) = {p_infected_given_positive:.4f} "
    f"({p_infected_given_positive:.1%})"
)
print(f"P(false positive)           = {p_false_positive:.4f} ({p_false_positive:.1%})")
# INTERPRETATION: Even with a 99.5% specificity test, when prevalence is
# only 2%, a positive test means you're truly infected only ~77% of the
# time. This is the base rate fallacy — ignoring prevalence leads to
# overconfidence in test results. This is why confirmatory tests exist.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 < p_infected_given_positive < 1, "Posterior probability must be valid"
assert (
    p_infected_given_positive > prevalence
), "Positive test must increase probability of infection above base rate"
print("\n✓ Checkpoint 1 passed — Bayes' theorem computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Prevalence Sweep — Base Rate Fallacy in Action
# ════════════════════════════════════════════════════════════════════════

print("--- Effect of Prevalence on P(infected | +test) ---")
prevalence_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
posterior_values = []

for prev in prevalence_values:
    p_pos = sensitivity * prev + (1 - specificity) * (1 - prev)
    p_inf = (sensitivity * prev) / p_pos
    posterior_values.append(p_inf)
    print(f"  Prevalence {prev:>5.1%} → P(infected | +) = {p_inf:.1%}")

# INTERPRETATION: At 0.1% prevalence (non-epidemic), a positive test means
# you're only ~15% likely to be infected. At 50% (outbreak peak), a positive
# test means ~99.4%. The same test result means completely different things
# depending on the base rate.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(posterior_values) == len(prevalence_values), "One posterior per prevalence"
assert (
    posterior_values[0] < posterior_values[-1]
), "Higher prevalence → higher posterior"
assert all(0 < p < 1 for p in posterior_values), "All posteriors must be valid"
print("\n✓ Checkpoint 2 passed — prevalence sweep complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Posterior Probability vs Prevalence Curve
# ════════════════════════════════════════════════════════════════════════

# Fine-grained sweep for smooth curve
prev_fine = np.linspace(0.001, 0.5, 200)
post_fine = [
    (sensitivity * p) / (sensitivity * p + (1 - specificity) * (1 - p))
    for p in prev_fine
]

fig1 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "P(infected | positive test) vs Prevalence",
        "False Positive Rate vs Prevalence",
    ],
)

# Left: posterior probability
fig1.add_trace(
    go.Scatter(
        x=prev_fine * 100,
        y=np.array(post_fine) * 100,
        mode="lines",
        name="P(infected | +test)",
        line={"color": "red", "width": 2},
    ),
    row=1,
    col=1,
)
fig1.add_trace(
    go.Scatter(
        x=[p * 100 for p in prevalence_values],
        y=[p * 100 for p in posterior_values],
        mode="markers",
        name="Computed points",
        marker={"color": "red", "size": 8},
    ),
    row=1,
    col=1,
)
fig1.add_hline(
    y=50,
    line_dash="dash",
    line_color="gray",
    row=1,
    col=1,
    annotation_text="50% — coin flip",
)

# Right: false positive rate
fp_fine = [1 - p for p in post_fine]
fig1.add_trace(
    go.Scatter(
        x=prev_fine * 100,
        y=np.array(fp_fine) * 100,
        mode="lines",
        name="False positive rate",
        line={"color": "blue", "width": 2},
    ),
    row=1,
    col=2,
)

fig1.update_xaxes(title_text="Prevalence (%)", row=1, col=1)
fig1.update_xaxes(title_text="Prevalence (%)", row=1, col=2)
fig1.update_yaxes(title_text="Posterior (%)", row=1, col=1)
fig1.update_yaxes(title_text="False Positive Rate (%)", row=1, col=2)
fig1.update_layout(title="Base Rate Fallacy: Why Test Results Mislead", height=450)
fig1.write_html(str(OUTPUT_DIR / "bayes_prevalence_sweep.html"))
print("Saved: bayes_prevalence_sweep.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(post_fine) == 200, "Fine sweep should have 200 points"
print("\n✓ Checkpoint 3 passed — visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: HDB Bishan Valuation — Is a $650K Listing Overpriced?
# ════════════════════════════════════════════════════════════════════════
# Scenario: a 4-room flat in Bishan is listed at $650K. A buyer asks:
# "What fraction of similar flats actually sell above $600K?"
#
# This is conditional probability from real data — no formula needed,
# just counting. But the Bayesian insight is: your PRIOR expectation
# (market average for 4-room is ~$500K) should be updated by the
# EVIDENCE (Bishan-specific transactions).

print("=== APPLICATION: HDB Bishan Valuation ===")

hdb_4room = load_hdb_4room()
bishan_flats = hdb_4room.filter(pl.col("town") == "BISHAN")

if bishan_flats.height > 0:
    p_above_600k_bishan = (
        bishan_flats.filter(pl.col("resale_price") > 600_000).height
        / bishan_flats.height
    )
    mean_bishan = bishan_flats["resale_price"].mean()
    print(f"Bishan 4-room data: {bishan_flats.height} transactions")
    print(f"Mean price: {fmt_money(mean_bishan)}")
    print(f"P(price > $600K | Bishan 4-room) = {p_above_600k_bishan:.2%}")
    print(f"This empirical probability is the 'data-driven prior' for Bishan.")

    # Dollar impact analysis
    listing_price = 650_000
    if p_above_600k_bishan > 0.5:
        print(f"\n→ {p_above_600k_bishan:.0%} of Bishan 4-room flats sell above $600K.")
        print(f"  A $650K listing is within the normal range for this location.")
        overpay_risk = listing_price - mean_bishan
        if overpay_risk > 0:
            print(f"  But at {fmt_money(overpay_risk)} above the mean, negotiate down.")
        else:
            print(f"  At {fmt_money(abs(overpay_risk))} below the mean — good deal.")
    else:
        print(f"\n→ Only {p_above_600k_bishan:.0%} sell above $600K.")
        print(f"  $650K is above the market norm — negotiate or walk away.")
else:
    print("No Bishan 4-room data found — using uninformative estimate")

# Compare Bishan vs market-wide
prices_array = load_hdb_prices_4room()
market_p_above_600k = float((prices_array > 600_000).mean())
print(f"\nMarket-wide P(4-room > $600K) = {market_p_above_600k:.2%}")
if bishan_flats.height > 0:
    lift = p_above_600k_bishan / market_p_above_600k if market_p_above_600k > 0 else 0
    print(f"Bishan premium: {lift:.2f}x the market-wide rate")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert market_p_above_600k >= 0, "Market probability must be non-negative"
print("\n✓ Checkpoint 4 passed — Bishan valuation complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED (1.2 — Bayes' Theorem)")
print("═" * 70)
print(
    """
  ✓ Bayes' theorem: P(A|B) = P(B|A)P(A)/P(B) — the engine behind
    every Bayesian update
  ✓ Base rate fallacy: a 99.5% specificity test is unreliable when
    prevalence is low — most positives are false positives
  ✓ Prevalence sweep: posterior probability is a smooth function of
    the base rate — visualised for stakeholder communication
  ✓ Real-world application: Bishan property valuation using
    conditional probabilities from actual transaction data
  ✓ Dollar framing: translate statistical results into actionable
    pricing decisions with explicit overpay risk

  NEXT: In 03_conjugate_priors.py, you'll implement the Normal-Normal
  and Beta-Binomial conjugate families — the mathematical machinery
  that makes Bayesian updating computationally tractable.
"""
)

print("\n✓ Exercise 1.2 complete — Bayes' Theorem")
