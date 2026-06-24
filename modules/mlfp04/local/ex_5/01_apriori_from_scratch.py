# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 5.1: Apriori Algorithm from Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement the Apriori algorithm for frequent itemset mining
#   - Apply the anti-monotone pruning principle by hand
#   - Generate candidate k-itemsets from frequent (k-1)-itemsets
#   - Count support with a single pass per level
#   - Understand why Apriori scales beyond brute-force enumeration
#
# PREREQUISITES:
#   - MLFP04 Exercise 1 (clustering)
#   - Basic set theory (subset, superset, intersection)
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — why Apriori works (anti-monotone pruning)
#   2. Build — implement `apriori()` and `_generate_candidates()`
#   3. Train — run it on 2,500 Singapore retail transactions
#   4. Visualise — L1 -> L2 -> L3 ladder and top frequent itemsets
#   5. Apply — FairPrice/Sheng Siong shelf layout optimisation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from collections import defaultdict

import polars as pl

from shared.mlfp04.ex_5 import (
    OUTPUT_DIR,
    PRODUCTS,
    format_itemset,
    generate_transactions,
    print_transaction_summary,
    setup_engines,
    teardown_engines,
    track_run,
    transactions_to_onehot,
)

# ── Kailash-ML ExperimentTracker — association-rules zoo shared store ────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Apriori Works
# ════════════════════════════════════════════════════════════════════════
# ANTI-MONOTONE PRINCIPLE:
#   If an itemset X is INFREQUENT, every superset of X is ALSO infrequent.
#
# Consequence: once L_k (frequent k-itemsets) is known, form candidates
# for L_{k+1} by joining L_k with itself AND requiring every (k)-subset
# of the new candidate to already be in L_k. Any itemset with an
# infrequent subset is pruned before it is ever counted.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Apriori from scratch
# ════════════════════════════════════════════════════════════════════════


def _generate_candidates(
    prev_level: list[frozenset[str]],
    k: int,
) -> list[frozenset[str]]:
    """Generate candidate k-itemsets from frequent (k-1)-itemsets.

    Apply the anti-monotone pruning rule: every (k-1)-subset of a
    candidate MUST already be in `prev_level`.
    """
    prev_set = set(prev_level)
    candidates: set[frozenset[str]] = set()

    for i, a in enumerate(prev_level):
        for b in prev_level[i + 1 :]:
            # TODO: union the two itemsets. Hint: use the | operator.
            union = ____

            # Only keep unions that grew to exactly k items.
            if len(union) != k:
                continue

            # TODO: anti-monotone check — every (k-1)-subset of `union`
            # must already be frequent (i.e. in prev_set). Build the
            # subsets by removing one item at a time and test with `in`.
            # Hint: `all((union - frozenset([item])) in prev_set for item in union)`
            all_subsets_frequent = ____

            if all_subsets_frequent:
                candidates.add(union)

    return list(candidates)


def apriori(
    transactions: list[set[str]],
    min_support: float,
    verbose: bool = True,
) -> dict[frozenset[str], float]:
    """Mine frequent itemsets with the Apriori algorithm."""
    n = len(transactions)
    min_count = min_support * n

    # ── L1: frequent single items ────────────────────────────────────
    item_counts: dict[str, int] = defaultdict(int)
    for txn in transactions:
        for item in txn:
            item_counts[item] += 1

    freq_itemsets: dict[frozenset[str], float] = {}
    current_level: list[frozenset[str]] = []
    for item, count in item_counts.items():
        if count >= min_count:
            fs = frozenset([item])
            # TODO: record the support (count/n) in freq_itemsets and
            # add `fs` to current_level so the next iteration can join.
            ____

    if verbose:
        print(f"  L1: {len(current_level)} frequent items (min_support={min_support})")

    # ── L2, L3, ... Lk ──────────────────────────────────────────────
    k = 2
    while current_level:
        # TODO: generate candidates for level k using _generate_candidates().
        candidates = ____
        if not candidates:
            break

        # Single pass to count support for every candidate.
        candidate_counts: dict[frozenset[str], int] = defaultdict(int)
        for txn in transactions:
            txn_frozen = frozenset(txn)
            for candidate in candidates:
                if candidate.issubset(txn_frozen):
                    candidate_counts[candidate] += 1

        # TODO: retain only candidates whose count >= min_count; store
        # their support in freq_itemsets and seed `current_level` for
        # the next iteration.
        current_level = []
        for candidate, count in candidate_counts.items():
            if count >= min_count:
                ____

        if verbose:
            print(f"  L{k}: {len(current_level)} frequent {k}-itemsets")
        k += 1

    return freq_itemsets


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: run Apriori on Singapore retail baskets
# ════════════════════════════════════════════════════════════════════════

transactions = generate_transactions(n=2500, seed=42)
print_transaction_summary(transactions)

print("\n=== Apriori (from scratch) ===")
MIN_SUPPORT = 0.03
frequent_itemsets = apriori(transactions, min_support=MIN_SUPPORT)
print(f"\n  Total frequent itemsets: {len(frequent_itemsets)}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(frequent_itemsets) > 0, "Apriori should find at least one frequent itemset"
n_txn = len(transactions)
for itemset, support in list(frequent_itemsets.items())[:5]:
    actual_count = sum(1 for t in transactions if itemset.issubset(frozenset(t)))
    actual_support = actual_count / n_txn
    assert (
        abs(actual_support - support) < 0.005
    ), f"Computed support {support:.4f} disagrees with actual {actual_support:.4f}"
    assert (
        support >= MIN_SUPPORT - 0.001
    ), f"Itemset with support {support:.4f} should meet min_support={MIN_SUPPORT}"
print("\n[ok] Checkpoint passed — Apriori computes correct support values\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the L1 -> Lk ladder and top itemsets
# ════════════════════════════════════════════════════════════════════════

sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda kv: -kv[1])

print("Top 15 frequent itemsets by support:")
print(f"  {'Itemset':<45} {'Support':>8}")
print("  " + "-" * 55)
for itemset, support in sorted_itemsets[:15]:
    print(f"  {format_itemset(itemset):<45} {support:>8.4f}")

top_df = pl.DataFrame(
    {
        "itemset": [format_itemset(s) for s, _ in sorted_itemsets[:30]],
        "size": [len(s) for s, _ in sorted_itemsets[:30]],
        "support": [float(v) for _, v in sorted_itemsets[:30]],
    }
)
top_df.write_csv(OUTPUT_DIR / "apriori_top_itemsets.csv")
print(f"\n  Saved: {OUTPUT_DIR / 'apriori_top_itemsets.csv'}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: FairPrice/Sheng Siong shelf layout optimisation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore supermarket chain operates ~200 neighbourhood
# outlets, each stocking ~8,000 SKUs in 1,200 sqft of HDB floor space.
# The merchandising analyst wants frequent 2- and 3-itemsets so that
# adjacent shelving can be re-planned.
#
# WHY APRIORI FITS: large item universe, monthly batch run, anti-monotone
# pruning removes ~99.9% of candidates, output is directly interpretable.
#
# BUSINESS IMPACT: Internal A/B tests at tier-1 SG grocers show that
# re-locating the top 50 cross-category pairs into adjacent shelving
# lifts basket size 4-7%. On S$250M GMV that is S$10-17M uplift per year
# for a weekend of merchandising work.


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# All M4 ex_5 lessons share one experiment ('m4_assoc_rules_zoo') so you
# can compare Apriori / FP-Growth / rules / rule-features in one store.

# Per-level itemset count series — the visible "ladder" Apriori climbs.
size_to_count: defaultdict[int, int] = defaultdict(int)
for itemset in frequent_itemsets:
    size_to_count[len(itemset)] += 1
max_k = max(size_to_count) if size_to_count else 0
ladder = [size_to_count[k] for k in range(1, max_k + 1)]
support_values = [float(s) for s in frequent_itemsets.values()]

# TODO: pick a stable run_name (e.g. "apriori_from_scratch") and fill in
# the two blank scalars: how many singletons (size 1) and how many pairs
# (size 2) Apriori found at min_support={MIN_SUPPORT}.
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "algorithm": "apriori",
        "implementation": "from_scratch",
        "n_transactions": len(transactions),
        "n_products": len(PRODUCTS),
        "min_support": MIN_SUPPORT,
        "max_k_reached": max_k,
    },
    scalar_metrics={
        "n_frequent_itemsets": float(len(frequent_itemsets)),
        "n_singletons_L1": ____,
        "n_pairs_L2": ____,
        "max_k_reached": float(max_k),
        "max_support": float(max(support_values) if support_values else 0.0),
    },
    series_metrics={"frequent_count_by_level": [float(c) for c in ladder]},
)
print(f"  [tracked] Apriori ladder + headline counts logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — mlxtend.frequent_patterns.apriori
# ════════════════════════════════════════════════════════════════════════
# The hand-rolled walk-through internalised the algorithm; the production
# destination is one library call:
#   from mlxtend.frequent_patterns import apriori as mlx_apriori
#   onehot = transactions_to_onehot(transactions).to_pandas().astype(bool)
#   mlx_apriori(onehot, min_support=MIN_SUPPORT, use_colnames=True)

# TODO: import mlxtend's apriori and call it on the one-hot frame; print
# how many frequent itemsets it returns. Confirm it matches your hand-
# rolled count.
from mlxtend.frequent_patterns import apriori as mlx_apriori  # noqa: E402

onehot_pd = transactions_to_onehot(transactions).to_pandas().astype(bool)
mlx_frequent = mlx_apriori(onehot_pd, min_support=____, use_colnames=True)
print(f"  mlxtend.apriori → {len(mlx_frequent)} frequent itemsets in 1 call")
print(f"  Hand-rolled    → {len(frequent_itemsets)}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Implemented the Apriori algorithm from scratch
  [x] Applied the anti-monotone pruning principle in _generate_candidates()
  [x] Counted support with one pass per level
  [x] Identified a production scenario where Apriori is the right tool
  [x] Confirmed mlxtend.apriori produces the same frequent itemsets

  Next: 02_fp_growth.py — mlxtend's FP-Growth with no candidate generation,
  compared head-to-head against your Apriori output from this file.
"""
)

# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
