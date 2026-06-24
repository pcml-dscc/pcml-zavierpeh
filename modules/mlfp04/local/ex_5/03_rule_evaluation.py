# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 5.3: Rule Evaluation — Support, Confidence, Lift, Conviction
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Turn frequent itemsets into directional association rules (X -> Y)
#   - Compute the four standard rule-quality metrics
#   - Apply a three-threshold filter (support + confidence + lift)
#   - Separate cross-category rules from within-category rules
#
# PREREQUISITES:
#   - 01_apriori_from_scratch.py
#   - Basic probability (conditional probability, independence)
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — rule metric definitions
#   2. Build — implement `generate_rules()` + three-threshold filter
#   3. Train — mine + score rules on Singapore retail baskets
#   4. Visualise — top rules + category breakdown + polars export
#   5. Apply — Watsons cart-page recommender
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from shared.mlfp04.ex_5 import (
    OUTPUT_DIR,
    categorise_rule,
    format_itemset,
    generate_transactions,
    print_transaction_summary,
    rules_to_polars,
    setup_engines,
    teardown_engines,
    track_run,
    transactions_to_onehot,
)

# ── Kailash-ML ExperimentTracker — association-rules zoo shared store ────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Four Rule Quality Metrics
# ════════════════════════════════════════════════════════════════════════
# support     = supp(X ∪ Y) = count(X ∪ Y) / N
# confidence  = conf(X -> Y) = supp(X ∪ Y) / supp(X)   # = P(Y | X)
# lift        = conf(X -> Y) / supp(Y)                 # = P(Y|X) / P(Y)
# conviction  = (1 - supp(Y)) / (1 - conf(X -> Y))
#
# Three-threshold filter (the only one that matters in practice):
#   keep if support >= s_min and confidence >= c_min and lift > 1


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: rule generator + three-threshold filter
# ════════════════════════════════════════════════════════════════════════


def _apriori(
    transactions: list[set[str]], min_support: float
) -> dict[frozenset[str], float]:
    """Small self-contained Apriori — same contract as 5.1."""
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


def generate_rules(
    freq_itemsets: dict[frozenset[str], float],
    min_confidence: float,
) -> list[dict]:
    """Generate all association rules that clear ``min_confidence``."""
    rules: list[dict] = []
    for itemset, support in freq_itemsets.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for ant_tuple in combinations(items, r):
                antecedent = frozenset(ant_tuple)
                consequent = itemset - antecedent

                supp_ant = freq_itemsets.get(antecedent)
                supp_con = freq_itemsets.get(consequent)
                if supp_ant is None or supp_con is None:
                    continue

                # TODO: compute confidence = support / supp_ant
                # Hint: support is P(X and Y); supp_ant is P(X)
                confidence = ____
                if confidence < min_confidence:
                    continue

                # TODO: compute lift = confidence / supp_con
                lift = ____

                # TODO: compute conviction. When confidence == 1.0 the
                # denominator would be zero — return float("inf") instead.
                # Formula: (1 - supp_con) / (1 - confidence)
                conviction = ____

                rules.append(
                    {
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "support": support,
                        "confidence": confidence,
                        "lift": lift,
                        "conviction": conviction,
                    }
                )
    return rules


def filter_actionable(
    rules: list[dict],
    min_support: float,
    min_confidence: float,
    min_lift: float,
) -> list[dict]:
    """Apply the three-threshold filter and sort by descending lift."""
    # TODO: keep a rule only if it clears ALL three thresholds:
    #   support >= min_support
    #   confidence >= min_confidence
    #   lift > min_lift
    kept = ____

    kept.sort(key=lambda r: -r["lift"])
    return kept


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: mine + score rules on the SG retail baskets
# ════════════════════════════════════════════════════════════════════════

transactions = generate_transactions(n=2500, seed=42)
print_transaction_summary(transactions)

MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.3

print("\n=== Mining frequent itemsets ===")
frequent = _apriori(transactions, min_support=MIN_SUPPORT)
print(f"  Frequent itemsets: {len(frequent)}")

print("\n=== Generating association rules ===")
rules = generate_rules(frequent, min_confidence=MIN_CONFIDENCE)
print(f"  Rules at min_confidence={MIN_CONFIDENCE}: {len(rules)}")

actionable = filter_actionable(
    rules, min_support=0.03, min_confidence=0.4, min_lift=1.5
)
print(f"  Actionable (supp>=0.03, conf>=0.4, lift>1.5): {len(actionable)}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(rules) > 0, "At least one rule should clear min_confidence"
for rule in rules[:10]:
    assert 0 <= rule["confidence"] <= 1.0, "confidence must be a probability"
    assert rule["lift"] > 0, "lift must be positive"
assert len(actionable) > 0, "At least one rule should clear the three thresholds"
assert actionable[0]["lift"] > 1.5, "Top actionable rule should have lift > 1.5"
print("\n[ok] Checkpoint passed — rule metrics valid and actionable set non-empty\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: top rules + category breakdown
# ════════════════════════════════════════════════════════════════════════

print("Top 15 actionable rules by lift:")
header = (
    f"  {'Antecedent':<28} {'->':>3} {'Consequent':<20} "
    f"{'Supp':>6} {'Conf':>6} {'Lift':>6} {'Conv':>7}"
)
print(header)
print("  " + "-" * 88)
for rule in actionable[:15]:
    ant = format_itemset(rule["antecedent"])
    con = format_itemset(rule["consequent"])
    conv = (
        f"{rule['conviction']:.2f}" if rule["conviction"] != float("inf") else "   inf"
    )
    print(
        f"  {ant:<28} {'->':>3} {con:<20} "
        f"{rule['support']:>6.3f} {rule['confidence']:>6.3f} "
        f"{rule['lift']:>6.2f} {conv:>7}"
    )

# Category breakdown
cross = 0
within = 0
for rule in actionable:
    _, _, rel = categorise_rule(rule["antecedent"], rule["consequent"])
    if rel.startswith("within-category"):
        within += 1
    else:
        cross += 1

print("\n=== Category Breakdown ===")
print(f"  Cross-category rules: {cross}")
print(f"  Within-category rules: {within}")

scatter_df = rules_to_polars(rules).sort("lift", descending=True).head(100)
scatter_df.write_csv(OUTPUT_DIR / "top_rules_scatter.csv")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Watsons cart-page recommender
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Watsons SG (~110 stores + online) wants a cart-page
# recommender that surfaces ONE high-lift next-product suggestion.
# Product spec:
#   reliable  -> high confidence (doesn't annoy shoppers)
#   surprising -> high lift (not just "shampoo -> soap")
#   popular    -> high support (enough stock to fulfil)
#
# These ARE the three-threshold filter inputs.
#
# BUSINESS IMPACT: Industry A/B tests show 2-4% conversion lift and 5-9%
# AOV lift on recommended items. Watsons SG online GMV ~S$200M/year;
# a 6% AOV lift on recommendations is ~S$3-6M/year in pure margin.


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log the rule-quality distribution to ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Same shared experiment as 01/02. Series = sorted lift / confidence /
# support arrays so the M4 dashboard can plot the rule-quality
# distribution alongside Apriori's ladder + FP-Growth's runtime curve.

import math  # noqa: E402

lifts = [float(r["lift"]) for r in rules]
confs = [float(r["confidence"]) for r in rules]
supps = [float(r["support"]) for r in rules]
finite_convs = [
    float(r["conviction"]) for r in rules if not math.isinf(r["conviction"])
]
mean_finite_conv = sum(finite_convs) / len(finite_convs) if finite_convs else 0.0
n_inf_conv = sum(1 for r in rules if math.isinf(r["conviction"]))
top_lift = max(float(r["lift"]) for r in actionable) if actionable else 0.0

# TODO: pick run_name="rule_evaluation_three_threshold" and fill in the
# two cross-category counters you computed above + the actionable rate
# (= n_actionable / n_rules).
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "algorithm": "association_rules",
        "implementation": "from_scratch",
        "n_transactions": len(transactions),
        "min_support_mining": MIN_SUPPORT,
        "min_confidence_filter": MIN_CONFIDENCE,
    },
    scalar_metrics={
        "n_frequent_itemsets": float(len(frequent)),
        "n_rules_generated": float(len(rules)),
        "n_rules_actionable": float(len(actionable)),
        "actionable_rate": ____,
        "cross_category_rules": ____,
        "within_category_rules": ____,
        "top_lift": float(top_lift),
        "n_inf_conviction": float(n_inf_conv),
        "mean_finite_conviction": float(mean_finite_conv),
    },
    series_metrics={
        "lift_sorted_desc": sorted(lifts, reverse=True)[:50],
        "confidence_sorted_desc": sorted(confs, reverse=True)[:50],
        "support_sorted_desc": sorted(supps, reverse=True)[:50],
    },
)
print(f"  [tracked] Rule-quality distribution logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — mlxtend.frequent_patterns.association_rules
# ════════════════════════════════════════════════════════════════════════
# The hand-rolled pipeline made every metric explicit. The production
# destination is two library calls (same as lesson 02) — fpgrowth +
# association_rules — and the SAME three-threshold filter you just
# implemented.

# TODO: import mlxtend's fpgrowth + association_rules; mine + score on
# the one-hot frame; apply the three-threshold filter and print how many
# actionable rules mlxtend produces. Confirm it matches your hand-rolled
# count (modulo floating-point edge cases right at the threshold).
from mlxtend.frequent_patterns import association_rules as mlx_rules  # noqa: E402
from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth  # noqa: E402

onehot_pd = transactions_to_onehot(transactions).to_pandas().astype(bool)
mlx_freq = mlx_fpgrowth(onehot_pd, min_support=MIN_SUPPORT, use_colnames=True)
mlx_rules_df = mlx_rules(mlx_freq, metric="confidence", min_threshold=MIN_CONFIDENCE)
mlx_actionable = mlx_rules_df[
    (mlx_rules_df["support"] >= 0.03)
    & (mlx_rules_df["confidence"] >= 0.4)
    & (mlx_rules_df["lift"] > ____)
]
print(f"  mlxtend rules : {len(mlx_rules_df)}  hand-rolled : {len(rules)}")
print(
    f"  mlxtend actionable: {len(mlx_actionable)}  "
    f"hand-rolled actionable: {len(actionable)}"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Generated directional association rules from frequent itemsets
  [x] Computed support, confidence, lift, and conviction
  [x] Applied the three-threshold filter
  [x] Separated cross-category rules from within-category rules
  [x] Reproduced the entire pipeline via two mlxtend calls

  Next: 04_rule_features.py — use the rules as features for a supervised
  classifier and compare against a raw product-presence baseline.
"""
)

# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
