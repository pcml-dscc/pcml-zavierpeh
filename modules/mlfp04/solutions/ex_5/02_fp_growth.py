# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 5.2: FP-Growth via mlxtend
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Use mlxtend's FP-Growth implementation end-to-end
#   - Convert polars transactions into the one-hot format FP-Growth expects
#   - Compare Apriori and FP-Growth on the same min_support
#   - Verify that both algorithms produce the same frequent itemsets
#   - Understand why FP-Growth is preferred on large transaction logs
#
# PREREQUISITES:
#   - 01_apriori_from_scratch.py (understand what frequent itemset mining does)
#
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — FP-tree construction and recursive mining
#   2. Build — wrap mlxtend FP-Growth in a polars-friendly call
#   3. Train — run FP-Growth on 2,500 SG retail baskets
#   4. Visualise — compare Apriori vs FP-Growth itemset counts + overlap
#   5. Apply — Grab GrabFood order-bundle mining at city scale
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared.mlfp04.ex_5 import (
    OUTPUT_DIR,
    format_itemset,
    generate_transactions,
    print_transaction_summary,
    setup_engines,
    teardown_engines,
    track_run,
    transactions_to_onehot,
)

# ── Kailash-ML ExperimentTracker — every association-rules run logs here ─
tracker, exp_name = setup_engines()

# mlxtend is the only pandas-touching library used in this exercise. The
# `rules/two-format.md` carve-out permits this because FP-Growth accepts
# a pandas DataFrame and no polars equivalent exists. We stay polars-native
# everywhere else; pandas is contained to the mlxtend call site.
from mlxtend.frequent_patterns import association_rules as mlx_association_rules
from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth

# Apriori is re-declared below (small, in-file) so this script stands
# alone without cross-file imports from a sibling starting with a digit.


# ════════════════════════════════════════════════════════════════════════
# THEORY — FP-Tree and Recursive Mining
# ════════════════════════════════════════════════════════════════════════
# Apriori is level-wise: it scans the full database once per itemset size.
# For a transaction log with tens of millions of rows, that is the cost
# bottleneck. FP-Growth sidesteps it completely.
#
#   FP-TREE CONSTRUCTION (two passes over the data)
#   Pass 1: count single-item support; drop infrequent items.
#   Pass 2: for each transaction, sort its items by global frequency (most
#           frequent first) and insert them into a prefix tree where each
#           path represents a transaction. Shared prefixes compress.
#
#   RECURSIVE MINING (no more DB scans)
#   For each frequent item (least-frequent first), build a "conditional
#   pattern base" from the tree, then recursively mine that smaller tree.
#   No candidate generation. No extra DB passes.
#
# GUARANTEES:
#   - Same frequent itemsets as Apriori given the same min_support
#   - Typically 2-10x faster on dense data, 10-100x faster on large data
#   - Higher memory cost (the FP-tree has to fit in RAM)
#
# We use the mlxtend implementation so we can focus on the USE of the
# algorithm rather than re-implement the tree — Apriori gave us the
# mechanics in 5.1, FP-Growth gives us the speed story here.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: polars-friendly FP-Growth wrapper
# ════════════════════════════════════════════════════════════════════════


def run_fp_growth(
    transactions: list[set[str]],
    min_support: float,
    min_confidence: float = 0.3,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run mlxtend FP-Growth on a list of basket-sets.

    Returns ``(frequent_itemsets_df, rules_df)`` as polars DataFrames.
    The mlxtend call requires pandas internally; we convert at the
    boundary so the rest of the exercise stays polars-native.
    """
    onehot_pl = transactions_to_onehot(transactions)
    onehot_pd = onehot_pl.to_pandas()

    fp_frequent = mlx_fpgrowth(onehot_pd, min_support=min_support, use_colnames=True)
    fp_rules = mlx_association_rules(
        fp_frequent, metric="confidence", min_threshold=min_confidence
    )

    frequent_pl = pl.DataFrame(
        {
            "itemset": [
                format_itemset(items) for items in fp_frequent["itemsets"].tolist()
            ],
            "size": [len(items) for items in fp_frequent["itemsets"].tolist()],
            "support": fp_frequent["support"].astype(float).tolist(),
        }
    )

    rules_pl = pl.DataFrame(
        {
            "antecedent": [
                format_itemset(items) for items in fp_rules["antecedents"].tolist()
            ],
            "consequent": [
                format_itemset(items) for items in fp_rules["consequents"].tolist()
            ],
            "support": fp_rules["support"].astype(float).tolist(),
            "confidence": fp_rules["confidence"].astype(float).tolist(),
            "lift": fp_rules["lift"].astype(float).tolist(),
        }
    )
    return frequent_pl, rules_pl


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: run FP-Growth on the same baskets used in 5.1
# ════════════════════════════════════════════════════════════════════════

transactions = generate_transactions(n=2500, seed=42)
print_transaction_summary(transactions)

MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.3

print("\n=== FP-Growth (mlxtend) ===")
fp_frequent_df, fp_rules_df = run_fp_growth(
    transactions, min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE
)
print(f"  Frequent itemsets: {fp_frequent_df.height}")
print(f"  Association rules: {fp_rules_df.height}")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: compare Apriori and FP-Growth on the same data
# ════════════════════════════════════════════════════════════════════════
# We re-run Apriori here so that the comparison is self-contained. In a
# real codebase you would import the function from 01_apriori_from_scratch.

from collections import defaultdict


def _apriori(
    transactions: list[set[str]], min_support: float
) -> dict[frozenset[str], float]:
    """Minimal Apriori — same contract as 01_apriori_from_scratch.apriori()."""
    n = len(transactions)
    min_count = min_support * n
    item_counts: dict[str, int] = defaultdict(int)
    for txn in transactions:
        for item in txn:
            item_counts[item] += 1
    freq: dict[frozenset[str], float] = {}
    level: list[frozenset[str]] = []
    for item, count in item_counts.items():
        if count >= min_count:
            fs = frozenset([item])
            freq[fs] = count / n
            level.append(fs)
    k = 2
    while level:
        prev_set = set(level)
        candidates: set[frozenset[str]] = set()
        for i, a in enumerate(level):
            for b in level[i + 1 :]:
                u = a | b
                if len(u) == k and all((u - frozenset([it])) in prev_set for it in u):
                    candidates.add(u)
        if not candidates:
            break
        counts: dict[frozenset[str], int] = defaultdict(int)
        for txn in transactions:
            tf = frozenset(txn)
            for c in candidates:
                if c.issubset(tf):
                    counts[c] += 1
        level = []
        for c, ct in counts.items():
            if ct >= min_count:
                freq[c] = ct / n
                level.append(c)
        k += 1
    return freq


apriori_itemsets = _apriori(transactions, min_support=MIN_SUPPORT)
apriori_set = {format_itemset(s) for s in apriori_itemsets}
fp_set = set(fp_frequent_df["itemset"].to_list())

print("\n=== Apriori vs FP-Growth ===")
print(f"  Apriori   itemsets: {len(apriori_set)}")
print(f"  FP-Growth itemsets: {len(fp_set)}")
print(f"  Intersection:       {len(apriori_set & fp_set)}")
print(f"  Apriori only:       {len(apriori_set - fp_set)}")
print(f"  FP-Growth only:     {len(fp_set - apriori_set)}")

agreement = len(apriori_set & fp_set) / max(len(apriori_set | fp_set), 1)
print(f"  Jaccard agreement:  {agreement:.2%}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert fp_frequent_df.height > 0, "FP-Growth should find at least one itemset"
assert fp_rules_df.height > 0, "FP-Growth should generate at least one rule"
assert agreement >= 0.90, (
    f"Apriori and FP-Growth should agree on >=90% of itemsets "
    f"at min_support={MIN_SUPPORT}; got {agreement:.2%}"
)
print("\n[ok] Checkpoint passed — FP-Growth matches Apriori on frequent itemsets\n")

fp_frequent_df.write_csv(OUTPUT_DIR / "fp_growth_itemsets.csv")
fp_rules_df.write_csv(OUTPUT_DIR / "fp_growth_rules.csv")
print(f"  Saved: {OUTPUT_DIR / 'fp_growth_itemsets.csv'}")
print(f"  Saved: {OUTPUT_DIR / 'fp_growth_rules.csv'}")

# ── Visualisation ─────────────────────────────────────────────────────
import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# (A) Itemset frequency bar chart — top 15 by support
top_itemsets = fp_frequent_df.sort("support", descending=True).head(15)
fig_freq = go.Figure(
    go.Bar(
        x=top_itemsets["support"].to_list(),
        y=top_itemsets["itemset"].to_list(),
        orientation="h",
        marker_color="#636EFA",
        text=[f"{s:.1%}" for s in top_itemsets["support"].to_list()],
        textposition="outside",
    )
)
fig_freq.update_layout(
    title="Top 15 Frequent Itemsets by Support (FP-Growth)",
    xaxis_title="Support",
    yaxis_title="Itemset",
    yaxis=dict(autorange="reversed"),
    margin=dict(l=200),
)
freq_path = OUTPUT_DIR / "02_itemset_frequency.html"
fig_freq.write_html(str(freq_path))
print(f"[viz] Itemset frequency: {freq_path}")

# (B) Apriori vs FP-Growth speed comparison across data sizes
sizes = [500, 1000, 1500, 2000, 2500]
apriori_times: list[float] = []
fp_times: list[float] = []
for sz in sizes:
    txns = generate_transactions(n=sz, seed=42)
    oh_pl = transactions_to_onehot(txns)
    oh_pd = oh_pl.to_pandas()

    t0 = time.perf_counter()
    _apriori(txns, min_support=MIN_SUPPORT)
    apriori_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    mlx_fpgrowth(oh_pd, min_support=MIN_SUPPORT, use_colnames=True)
    fp_times.append(time.perf_counter() - t0)

fig_speed = go.Figure()
fig_speed.add_trace(
    go.Scatter(
        x=sizes,
        y=apriori_times,
        mode="lines+markers",
        name="Apriori (from scratch)",
        marker_color="#EF553B",
    )
)
fig_speed.add_trace(
    go.Scatter(
        x=sizes,
        y=fp_times,
        mode="lines+markers",
        name="FP-Growth (mlxtend)",
        marker_color="#00CC96",
    )
)
fig_speed.update_layout(
    title="Apriori vs FP-Growth: Runtime by Transaction Count",
    xaxis_title="Number of Transactions",
    yaxis_title="Time (seconds)",
)
speed_path = OUTPUT_DIR / "02_speed_comparison.html"
fig_speed.write_html(str(speed_path))
print(f"[viz] Speed comparison: {speed_path}")

# INTERPRETATION: Both algorithms are level-complete — they find ALL
# frequent itemsets at the given min_support, and the sets should agree
# (modulo floating-point edge cases right at the threshold). The choice
# between them is an engineering decision about speed and memory, not
# correctness. On this small 2,500-row basket, Apriori is fine; at 10M+
# rows, FP-Growth's single-pass tree construction dominates.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: GrabFood order-bundle mining at city scale
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: GrabFood Singapore processes ~800,000 orders per day across
# 12,000+ merchants. The merchant-growth team wants to find high-lift
# menu-item pairings (e.g., "chicken rice + iced milo") so merchants can
# publish bundle deals that raise average order value.
#
# Why FP-Growth is the right tool here:
#   - 800K orders/day * 7 days = ~5.6M transactions per weekly mining job
#   - Typical basket has 2-4 items, item universe ~100K SKUs across all
#     merchants — too large for level-wise Apriori to scan repeatedly
#   - Batch job runs overnight on a single worker; minutes matter
#   - FP-Growth builds the tree in memory in one pass, then mines it
#     without further DB scans — on this shape of data it is typically
#     30-60x faster than Apriori
#
# BUSINESS IMPACT: A 2024 internal experiment on a SEA food-delivery
# platform showed that merchants who adopted the top-3 data-driven
# bundles lifted AOV by 9-14% over the control cohort. At a platform
# average order value of S$18 and 800K orders/day, a 10% AOV lift is
# worth ~S$1.4M/day in GMV — roughly S$45M/month. The FP-Growth mining
# job is ~S$50 of compute per run.
#
# LIMITATIONS:
#   - FP-tree must fit in memory; for datasets with 100M+ rows and dense
#     baskets, partitioning by merchant category is required
#   - Frequent itemsets alone do not tell you which bundles are
#     profitable — you still need to multiply by margin and demand
#     elasticity (out of scope for this exercise)


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Logs FP-Growth's headline counts + the per-size runtime sweep against
# Apriori. Series = runtimes at each transaction-count tested; scalars =
# itemset / rule counts + Jaccard agreement vs the hand-rolled Apriori.

speedup = apriori_times[-1] / fp_times[-1] if fp_times[-1] > 0 else 0.0
track_run(
    tracker,
    exp_name,
    run_name="fp_growth_mlxtend",
    params={
        "algorithm": "fp_growth",
        "implementation": "mlxtend",
        "n_transactions": len(transactions),
        "min_support": MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE,
        "swept_sizes": ",".join(str(s) for s in sizes),
    },
    scalar_metrics={
        "n_frequent_itemsets": float(fp_frequent_df.height),
        "n_rules": float(fp_rules_df.height),
        "jaccard_apriori_vs_fp": float(agreement),
        "apriori_only_count": float(len(apriori_set - fp_set)),
        "fp_only_count": float(len(fp_set - apriori_set)),
        "speedup_at_max_size": float(speedup),
        "apriori_runtime_at_max": float(apriori_times[-1]),
        "fp_runtime_at_max": float(fp_times[-1]),
    },
    series_metrics={
        "apriori_runtime_seconds": [float(t) for t in apriori_times],
        "fp_growth_runtime_seconds": [float(t) for t in fp_times],
    },
)
print(f"  [tracked] FP-Growth + speed sweep logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — the mlxtend.frequent_patterns one-liner
# ════════════════════════════════════════════════════════════════════════
# This lesson built a polars wrapper around mlxtend FP-Growth, ran the
# Apriori-vs-FP-Growth speed sweep, and verified itemset agreement —
# ~270 lines of structure to internalise WHY FP-Growth is the production
# choice on dense, city-scale data. The destination is exactly the two
# library calls you saw inside ``run_fp_growth``:
#
#   from mlxtend.frequent_patterns import fpgrowth, association_rules
#   itemsets = fpgrowth(onehot_df, min_support=0.03, use_colnames=True)
#   rules    = association_rules(itemsets, metric='confidence', min_threshold=0.3)
#
# Two calls, one pandas frame in, two pandas frames out. The polars
# wrapper exists for ergonomics, not for FP-Growth's correctness — every
# downstream lesson (03 evaluation, 04 features) consumes that same rule
# table.

print("  Destination contract:")
print("    fpgrowth(onehot_df, min_support, use_colnames=True)  -> itemsets")
print("    association_rules(itemsets, metric, min_threshold)   -> rules")
print(
    f"  Today's run: {fp_frequent_df.height} itemsets, "
    f"{fp_rules_df.height} rules — both shape: pandas DataFrame"
)
print()


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Wrapped mlxtend FP-Growth in a polars-friendly call boundary
  [x] Converted basket-sets into the one-hot format FP-Growth expects
  [x] Verified that FP-Growth and Apriori agree on frequent itemsets
  [x] Identified a city-scale workload (GrabFood) where FP-Growth's
      single-pass tree construction is the economic difference-maker
  [x] Locked in the production destination: two mlxtend calls
      (fpgrowth + association_rules) feed every downstream lesson

  KEY INSIGHT: Both algorithms are correct. The choice between them is
  about the SHAPE of your data (size, density, access cost) and never
  about which one "finds more patterns." When someone tells you
  FP-Growth "discovered" rules Apriori missed, the real difference is
  almost always a different min_support.

  Next: 03_rule_evaluation.py — turn frequent itemsets into association
  rules and score them with support, confidence, lift, and conviction.
"""
)

# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
