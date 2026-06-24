# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 1.1: Probability Fundamentals and Maximum Likelihood
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Construct truth tables for probability problems and compute joint /
#     conditional probabilities from real HDB transaction data
#   - Test whether two events are independent using the product rule
#   - Compute MLE for Normal distribution parameters (μ̂, σ̂²)
#   - Quantify estimation uncertainty via the Cramér-Rao lower bound
#   - Apply MLE precision to a Singapore property valuation scenario
#
# PREREQUISITES: Complete M1 — comfortable loading data, computing
#   summary statistics, and reading Polars DataFrames.
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — what probability means for property valuation
#   2. Build — load HDB data, compute truth table and joint probs
#   3. Train — MLE for Normal parameters with Cramér-Rao bound
#   4. Visualise — cross-tabulation heatmap + MLE precision plot
#   5. Apply — Singapore property valuation: how precise is "the average"?
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from shared.mlfp02.ex_1 import (
    OUTPUT_DIR,
    fmt_money,
    load_hdb_4room,
    load_hdb_all,
    load_hdb_prices_4room,
    normal_mle,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — What Probability Means for Property Valuation
# ════════════════════════════════════════════════════════════════════════
# Probability begins with counting. Before any Bayesian update, before any
# machine learning model, we need to know how to answer:
#
#   "What fraction of events satisfy condition A AND condition B?"
#   "Given that A occurred, what is the probability of B?"
#
# These are joint and conditional probabilities — the building blocks of
# every statistical model. In property valuation, this translates to
# questions like:
#
#   "If I know a flat is 4-room, what's the chance it sold above $500K?"
#   "Are flat type and price range independent, or does one tell me
#    something about the other?"
#
# Key rules:
#   P(A) + P(A') = 1
#   P(A,B) = P(A) × P(B|A)
#   Independent events: P(A,B) = P(A) × P(B)
#
# Maximum Likelihood Estimation (MLE) extends counting to continuous
# distributions. For X ~ N(μ, σ²):
#   μ̂ = x̄   (sample mean)
#   σ̂² = (1/n) Σ(xᵢ - x̄)²   (biased variance, ddof=0)
#
# The Fisher information I(μ) = n/σ² tells us how much the data reveals
# about μ. The Cramér-Rao bound says Var(μ̂) ≥ 1/I(μ) = σ²/n — no
# unbiased estimator can do better.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Load HDB data and compute truth table
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP02 Exercise 1.1: Probability Fundamentals & MLE")
print("=" * 70)

# Load data via shared helpers
prices = load_hdb_prices_4room()
hdb_all = load_hdb_all()
total_n = hdb_all.height

print(f"\n  Data loaded: {len(prices):,} 4-room HDB transactions (2020+)")
print(f"  Price range: {fmt_money(prices.min())} – {fmt_money(prices.max())}")
print(f"  Sample mean: {fmt_money(prices.mean())}")
print(f"  Sample std:  {fmt_money(prices.std())}\n")

# ── Event definitions ──
# Event A: transaction is a 4-room flat
n_4room = hdb_all.filter(pl.col("flat_type") == "4 ROOM").height
p_4room = n_4room / total_n

# Event B: price above $500K
n_above_500k = hdb_all.filter(pl.col("resale_price") > 500_000).height
p_above_500k = n_above_500k / total_n

# Joint probability: P(4-room AND price > 500K)
n_4room_and_above = hdb_all.filter(
    (pl.col("flat_type") == "4 ROOM") & (pl.col("resale_price") > 500_000)
).height
p_joint = n_4room_and_above / total_n

# Conditional: P(price > 500K | 4-room)
p_above_given_4room = n_4room_and_above / n_4room if n_4room > 0 else 0

# Test independence: P(A,B) vs P(A)×P(B)
p_independent = p_4room * p_above_500k

print("--- Truth Table (empirical) ---")
print(f"Total transactions (2020+): {total_n:,}")
print(f"P(4-room)           = {p_4room:.4f} ({p_4room:.1%})")
print(f"P(price > $500K)    = {p_above_500k:.4f} ({p_above_500k:.1%})")
print(f"P(4-room AND >$500K)= {p_joint:.4f} ({p_joint:.1%})")
print(f"P(>$500K | 4-room)  = {p_above_given_4room:.4f} ({p_above_given_4room:.1%})")
print(f"\n--- Independence Check ---")
print(f"P(A)×P(B) = {p_independent:.4f}")
print(f"P(A,B)    = {p_joint:.4f}")
print(f"Difference: {abs(p_joint - p_independent):.4f}")
if abs(p_joint - p_independent) < 0.01:
    print("Events are approximately independent")
else:
    print("Events are NOT independent — flat type affects price probability")
# INTERPRETATION: If flat type and price are not independent, knowing
# the flat type tells you something about the price distribution. This
# is the foundation for conditional reasoning in property valuation.

# Cross-tabulation: flat_type × price_category
price_cats = hdb_all.with_columns(
    pl.when(pl.col("resale_price") <= 400_000)
    .then(pl.lit("≤400K"))
    .when(pl.col("resale_price") <= 600_000)
    .then(pl.lit("400K-600K"))
    .when(pl.col("resale_price") <= 800_000)
    .then(pl.lit("600K-800K"))
    .otherwise(pl.lit(">800K"))
    .alias("price_band")
)
cross_tab = (
    price_cats.group_by("flat_type", "price_band")
    .agg(pl.len().alias("count"))
    .sort("flat_type", "price_band")
)
print(f"\n--- Cross-Tabulation (sample) ---")
print(cross_tab.head(12))

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 < p_4room < 1, "P(4-room) must be a valid probability"
assert 0 < p_above_500k < 1, "P(>500K) must be a valid probability"
assert p_joint <= min(p_4room, p_above_500k), "Joint prob cannot exceed marginals"
assert (
    abs(p_above_given_4room - p_joint / p_4room) < 1e-10
), "Conditional probability identity must hold: P(B|A) = P(A,B)/P(A)"
print("\n✓ Checkpoint 1 passed — probability fundamentals computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Maximum Likelihood Estimation with Cramér-Rao bound
# ════════════════════════════════════════════════════════════════════════

mle = normal_mle(prices)

print("=== MLE Estimates ===")
print(f"μ̂ = {fmt_money(mle.mean)}")
print(f"σ̂ (MLE, ddof=0)     = {fmt_money(mle.mle_std)}")
print(f"σ̂ (unbiased, ddof=1) = {fmt_money(mle.unbiased_std)}")
print(f"Bias: MLE σ underestimates by ${mle.unbiased_std - mle.mle_std:,.2f}")
print(f"\nFisher information I(μ) = {mle.fisher_information:.4f}")
print(f"Cramér-Rao lower bound: Var(μ̂) ≥ {mle.cramer_rao_bound:.2f}")
print(f"MLE standard error: ${mle.standard_error:,.2f}")
# INTERPRETATION: The standard error tells you the precision of the
# mean estimate. With many thousands of transactions, the SE is tiny
# relative to the mean — our estimate of the average 4-room HDB price
# is very precise. The Cramér-Rao bound guarantees no unbiased estimator
# can do better than this SE for the Normal model.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert mle.n > 0, "No data loaded — check the filter conditions"
assert mle.mean > 0, "MLE mean should be positive (price cannot be zero)"
assert mle.mle_std > 0, "MLE std should be positive"
assert mle.standard_error > 0, "Standard error should be positive"
assert mle.standard_error < mle.mle_std, "SE of mean should be much smaller than std"
assert (
    mle.unbiased_std > mle.mle_std
), "Unbiased σ must be > MLE σ (Bessel's correction)"
print("\n✓ Checkpoint 2 passed — MLE estimates and Cramér-Rao bound computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Cross-tabulation heatmap + MLE precision
# ════════════════════════════════════════════════════════════════════════

# -- Plot 1: Cross-tab heatmap of flat_type × price_band --
pivot = cross_tab.pivot(on="price_band", index="flat_type", values="count").fill_null(0)
flat_types_order = sorted(pivot["flat_type"].to_list())
bands = ["≤400K", "400K-600K", "600K-800K", ">800K"]
z_matrix = []
for ft in flat_types_order:
    row_data = pivot.filter(pl.col("flat_type") == ft)
    row = [int(row_data[b].item()) if b in row_data.columns else 0 for b in bands]
    z_matrix.append(row)

fig1 = go.Figure(
    data=go.Heatmap(
        z=z_matrix,
        x=bands,
        y=flat_types_order,
        colorscale="Blues",
        text=[[str(v) for v in row] for row in z_matrix],
        texttemplate="%{text}",
    )
)
fig1.update_layout(
    title="HDB Cross-Tabulation: Flat Type × Price Band (2020+)",
    xaxis_title="Price Band",
    yaxis_title="Flat Type",
    height=400,
)
fig1.write_html(str(OUTPUT_DIR / "cross_tab_heatmap.html"))
print("Saved: cross_tab_heatmap.html")

# -- Plot 2: MLE precision — sample size vs standard error --
sample_sizes = np.arange(10, mle.n + 1, max(1, mle.n // 200))
se_values = mle.mle_std / np.sqrt(sample_sizes)

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=sample_sizes,
        y=se_values,
        mode="lines",
        name="SE = σ / √n",
        line={"color": "blue"},
    )
)
fig2.add_hline(
    y=mle.standard_error,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Actual SE = ${mle.standard_error:,.0f}",
)
fig2.update_layout(
    title="MLE Precision: How Standard Error Shrinks with Sample Size",
    xaxis_title="Sample Size (n)",
    yaxis_title="Standard Error ($)",
    height=400,
)
fig2.write_html(str(OUTPUT_DIR / "mle_precision.html"))
print("Saved: mle_precision.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(z_matrix) > 0, "Cross-tab matrix should not be empty"
print("\n✓ Checkpoint 3 passed — visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore Property Valuation — How Precise Is
#          "The Average"?
# ════════════════════════════════════════════════════════════════════════
# A property developer planning a 4-room HDB launch in Queenstown wants
# to know the average resale price to set a competitive listing. They
# pull data from data.gov.sg.
#
# With n transactions, MLE gives μ̂ ± SE. The developer's margin is $20K
# per unit. If SE > $20K, the estimate is not precise enough to make a
# confident pricing decision — they need more data or a tighter segment.

print("=== APPLICATION: Property Developer Pricing Decision ===")
margin = 20_000
print(f"\nDeveloper margin per unit: {fmt_money(margin)}")
print(f"MLE estimate: {fmt_money(mle.mean)} ± {fmt_money(mle.standard_error)}")

if mle.standard_error < margin:
    print(f"\n✓ SE ({fmt_money(mle.standard_error)}) < margin ({fmt_money(margin)})")
    print("  → The estimate is precise enough for confident pricing.")
    print(f"  → List at {fmt_money(mle.mean)} with {fmt_money(margin)} margin")
    print(
        f"    gives a range of {fmt_money(mle.mean - margin)} – {fmt_money(mle.mean + margin)}"
    )
else:
    print(f"\n✗ SE ({fmt_money(mle.standard_error)}) ≥ margin ({fmt_money(margin)})")
    print("  → Need a narrower segment or more data for confident pricing.")

# How many observations would we need for SE < $5K?
target_se = 5_000
n_needed = int(np.ceil((mle.mle_std / target_se) ** 2))
print(f"\nTo achieve SE < {fmt_money(target_se)}: need n ≥ {n_needed:,} transactions")
print(
    f"  (currently have {mle.n:,} — {'sufficient' if mle.n >= n_needed else 'insufficient'})"
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert margin > 0, "Margin must be positive"
assert n_needed > 0, "Required n must be positive"
print("\n✓ Checkpoint 4 passed — business application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED (1.1 — Probability & MLE)")
print("═" * 70)
print(
    """
  ✓ Joint probability P(A,B) and conditional P(A|B) from real data
  ✓ Independence test: P(A,B) vs P(A)×P(B) — flat type and price
    are NOT independent
  ✓ Cross-tabulation reveals where the market volume concentrates
  ✓ MLE for Normal: μ̂ = x̄, σ̂² with ddof=0 (biased) vs ddof=1
  ✓ Cramér-Rao bound: MLE achieves minimum possible variance
  ✓ Business framing: SE determines whether the estimate is
    actionable for a $20K pricing margin

  NEXT: In 02_bayes_theorem.py, you'll apply Bayes' theorem to
  medical testing and property valuation — learning why base rates
  make positive tests less trustworthy than you'd expect.
"""
)

print("\n✓ Exercise 1.1 complete — Probability Fundamentals & MLE")
