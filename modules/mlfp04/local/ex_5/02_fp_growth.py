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
#
# PREREQUISITES:
#   - 01_apriori_from_scratch.py
#
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — FP-tree construction and recursive mining
#   2. Build — wrap mlxtend FP-Growth in a polars-friendly call
#   3. Train — run FP-Growth on 2,500 SG retail baskets
#   4. Visualise — Apriori vs FP-Growth itemset overlap
#   5. Apply — GrabFood order-bundle mining at city scale
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from collections import defaultdict

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

# ── Kailash-ML ExperimentTracker — association-rules zoo shared store ────
tracker, exp_name = setup_engines()

# mlxtend requires pandas internally. Use at the boundary only — keep
# the rest of this file polars-native.
from mlxtend.frequent_patterns import association_rules as mlx_association_rules
from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth


# ════════════════════════════════════════════════════════════════════════
# THEORY — FP-Tree and Recursive Mining
# ════════════════════════════════════════════════════════════════════════
# FP-Growth builds a prefix tree (FP-tree) in TWO passes over the data,
# then mines frequent itemsets recursively from conditional pattern
# bases in the tree — no more DB scans after tree construction, and no
# candidate generation at all.
#
# Guarantees: same frequent itemsets as Apriori given the same
# min_support, typically 2-10x faster on dense data, 10-100x faster on
# large data. Downside: the FP-tree has to fit in memory.


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
    """
    # TODO: build the one-hot transaction matrix using
    # transactions_to_onehot() and then convert to pandas for mlxtend.
    # Hint: polars DataFrames have a .to_pandas() method.
    onehot_pl = ____
    onehot_pd = ____

    # TODO: call mlx_fpgrowth() with use_colnames=True so the returned
    # frame uses product names (not column indices).
    fp_frequent = ____

    # TODO: derive association rules from `fp_frequent`. Filter on
    # confidence with min_threshold=min_confidence.
    # Hint: mlx_association_rules(fp_frequent, metric="confidence", ...)
    fp_rules = ____

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
# TASK 4 — VISUALISE: compare Apriori and FP-Growth
# ════════════════════════════════════════════════════════════════════════


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

# TODO: compute the Jaccard agreement between the two sets. Remember to
# guard against divide-by-zero on an empty union.
# Hint: |A ∩ B| / |A ∪ B|
agreement = ____
print(f"  Jaccard agreement:  {agreement:.2%}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert fp_frequent_df.height > 0, "FP-Growth should find at least one itemset"
assert fp_rules_df.height > 0, "FP-Growth should generate at least one rule"
assert (
    agreement >= 0.90
), f"Apriori and FP-Growth should agree on >=90% of itemsets; got {agreement:.2%}"
print("\n[ok] Checkpoint passed — FP-Growth matches Apriori on frequent itemsets\n")

fp_frequent_df.write_csv(OUTPUT_DIR / "fp_growth_itemsets.csv")
fp_rules_df.write_csv(OUTPUT_DIR / "fp_growth_rules.csv")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: GrabFood order-bundle mining
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: GrabFood SG processes ~800K orders/day across 12K+ merchants.
# Weekly mining run (~5.6M transactions) to surface high-lift menu pairs
# for merchant bundle promotions. FP-Growth's single-pass tree is 30-60x
# faster than Apriori on this shape of data.
#
# BUSINESS IMPACT: 10% AOV lift on S$18 average order with 800K orders/day
# is ~S$1.4M/day GMV — S$45M/month for ~S$50 of compute per weekly run.


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log FP-Growth + the Apriori-vs-FP-Growth speed sweep
# ════════════════════════════════════════════════════════════════════════
# Same shared experiment as lesson 01. Series = per-size runtimes for
# Apriori vs FP-Growth so you can see the speed crossover; scalars =
# Jaccard agreement on the frequent-itemset sets + headline counts.

speedup = apriori_times[-1] / fp_times[-1] if fp_times[-1] > 0 else 0.0

# TODO: pick run_name="fp_growth_mlxtend" and fill in the agreement +
# speedup scalars. Hint: agreement is the Jaccard you already computed.
track_run(
    tracker,
    exp_name,
    run_name=____,
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
        "jaccard_apriori_vs_fp": ____,
        "speedup_at_max_size": ____,
    },
    series_metrics={
        "apriori_runtime_seconds": [float(t) for t in apriori_times],
        "fp_growth_runtime_seconds": [float(t) for t in fp_times],
    },
)
print(f"  [tracked] FP-Growth + speed sweep logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — the two mlxtend calls every downstream lesson uses
# ════════════════════════════════════════════════════════════════════════
# The polars wrapper exists for ergonomics, not for FP-Growth's
# correctness. The destination contract is two pandas-frame calls:
#   fpgrowth(onehot_df, min_support, use_colnames=True)  -> itemsets
#   association_rules(itemsets, metric, min_threshold)   -> rules
# Lessons 03 and 04 consume that same rule table.

print("  Destination contract:")
print("    fpgrowth(onehot, min_support, use_colnames=True)  -> itemsets")
print("    association_rules(itemsets, metric, min_threshold) -> rules")
print(
    f"  Today's run: {fp_frequent_df.height} itemsets, "
    f"{fp_rules_df.height} rules — both shape: pandas DataFrame"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Wrapped mlxtend FP-Growth in a polars-friendly call boundary
  [x] Converted basket-sets to one-hot for FP-Growth input
  [x] Verified Apriori and FP-Growth agree on frequent itemsets
  [x] Locked in the production destination: two mlxtend calls feed
      every downstream lesson (rules + features)

  Next: 03_rule_evaluation.py — turn frequent itemsets into association
  rules with support, confidence, lift, and conviction.
"""
)

# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
