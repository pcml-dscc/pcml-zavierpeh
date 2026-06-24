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
#   - Rank actionable rules for business stakeholders
#
# PREREQUISITES:
#   - 01_apriori_from_scratch.py (frequent itemset mining)
#   - Basic probability (conditional probability, independence)
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — what each rule metric means mathematically and in business terms
#   2. Build — implement `generate_rules()` and the three-threshold filter
#   3. Train — mine + score rules on the Singapore retail basket
#   4. Visualise — top rules table + support/confidence/lift scatter
#   5. Apply — Watsons "buy-together" personalisation engine
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

# ── Kailash-ML ExperimentTracker — every association-rules run logs here ─
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Four Rule Quality Metrics
# ════════════════════════════════════════════════════════════════════════
# A rule "X -> Y" says: baskets containing X tend to also contain Y.
# It is a statistical summary, not a causal claim. We score it with four
# metrics because each one catches a different failure mode.
#
#   SUPPORT     supp(X u Y) = count(X u Y) / N
#               How often the pair appears. Low support = not enough data
#               to trust the rule; rules with support < 1% are noise on
#               most retail datasets.
#
#   CONFIDENCE  conf(X -> Y) = supp(X u Y) / supp(X)
#               P(Y given X). How RELIABLE the rule is when X is present.
#               A 95% confidence rule fires ~19 out of 20 times X appears.
#
#   LIFT        lift(X -> Y) = conf(X -> Y) / supp(Y)
#               = P(Y|X) / P(Y). How SURPRISING the rule is. Lift = 1
#               means X and Y are independent. Lift > 1 is positive
#               association — X raises the probability of Y. Lift < 1 is
#               negative association (substitutes, not complements).
#
#   CONVICTION  conv(X -> Y) = (1 - supp(Y)) / (1 - conf(X -> Y))
#               Directional strength. Unlike lift (symmetric), conviction
#               asymmetrically measures how much X IMPLIES Y. Conviction
#               of 1 = independence; conviction of infinity = perfect
#               implication (conf = 1, never violated).
#
# THE THREE-THRESHOLD FILTER (the only one that matters in practice)
#
#   keep the rule if:
#     support >= s_min        (enough baskets -> statistical power)
#     confidence >= c_min     (reliable enough to act on)
#     lift > 1                (surprising enough to be informative)
#
# Any ONE of these alone is a trap: high-confidence low-lift rules are
# just popularity (everyone buys milk, so any X -> milk has high
# confidence but zero insight). Low-support high-lift rules are often
# just statistical artefacts.


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
    """Generate all association rules that clear ``min_confidence``.

    For every frequent itemset of size >= 2, every non-empty proper subset
    is a candidate antecedent; its complement is the consequent. We score
    each candidate with support / confidence / lift / conviction and
    drop those below ``min_confidence``.
    """
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

                confidence = support / supp_ant
                if confidence < min_confidence:
                    continue

                lift = confidence / supp_con
                conviction = (
                    (1 - supp_con) / (1 - confidence)
                    if confidence < 1.0
                    else float("inf")
                )

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
    kept = [
        r
        for r in rules
        if r["support"] >= min_support
        and r["confidence"] >= min_confidence
        and r["lift"] > min_lift
    ]
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
print(
    "\n[ok] Checkpoint passed — rule metrics are valid and actionable set is non-empty\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: top rules + category breakdown + polars scatter
# ════════════════════════════════════════════════════════════════════════

print("Top 15 actionable rules by lift:")
header = f"  {'Antecedent':<28} {'->':>3} {'Consequent':<20} "
header += f"{'Supp':>6} {'Conf':>6} {'Lift':>6} {'Conv':>7}"
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

# Polars scatter-ready frame (for ModelVisualizer or plotly in notebooks)
scatter_df = rules_to_polars(rules).sort("lift", descending=True).head(100)
scatter_df.write_csv(OUTPUT_DIR / "top_rules_scatter.csv")
print(f"\n  Saved: {OUTPUT_DIR / 'top_rules_scatter.csv'}")

# INTERPRETATION: Cross-category rules are the commercially interesting
# ones. Within-category rules (shampoo + soap) mostly restate what a
# store manager already knows — they both sit in the personal-care
# aisle. Cross-category rules (breakfast items -> beverages) are where
# Association Rules pay for themselves: they surface adjacencies you
# did not explicitly plan for.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Watsons personalisation engine
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Watsons operates ~110 stores across Singapore plus an online
# channel. The CRM team wants a "buy-together" recommender that shows
# a shopper exactly ONE extra product on the cart page, chosen to
# maximise incremental basket value. The recommendation MUST be:
#   - Reliable  (high confidence — doesn't annoy shoppers with misses)
#   - Surprising (high lift — not just "you bought shampoo, try soap")
#   - Popular   (high support — enough stock to fulfil)
#
# The three-threshold filter above is literally the product spec for
# this recommender: support gates inventory, confidence gates annoyance,
# lift gates relevance.
#
# BUSINESS IMPACT: Industry A/B tests on cart-page recommenders typically
# show 2-4% conversion lift and 5-9% AOV lift when the recommendation is
# driven by high-lift association rules rather than "most popular."
# Watsons SG's reported online GMV is on the order of S$200M/year; a
# 6% AOV lift on the recommended items alone is ~S$3-6M/year in pure
# margin — on top of brand uplift from "the app knows what I need."
#
# LIMITATIONS:
#   - Association rules are backward-looking; seasonal or new-launch items
#     need a cold-start fallback (Ex 7 matrix factorisation helps here)
#   - A rule is a correlation, not a preference — always pair with
#     business rules (don't recommend a more expensive substitute to a
#     shopper who just demonstrated price sensitivity)


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log rule-quality metrics to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Logs the four-metric distribution across all rules + the actionable cut +
# the cross-vs-within category split. Series = sorted-lift / sorted-conf /
# sorted-support arrays so the M4 dashboard can plot the rule-quality
# distribution alongside the other techniques in this experiment.

import math  # noqa: E402

lifts = [float(r["lift"]) for r in rules]
confs = [float(r["confidence"]) for r in rules]
supps = [float(r["support"]) for r in rules]
finite_convs = [
    float(r["conviction"]) for r in rules if not math.isinf(r["conviction"])
]
mean_finite_conv = sum(finite_convs) / len(finite_convs) if finite_convs else 0.0
n_inf_conv = sum(1 for r in rules if math.isinf(r["conviction"]))

actionable_lifts = [float(r["lift"]) for r in actionable]
top_lift = max(actionable_lifts) if actionable_lifts else 0.0
mean_act_lift = (
    sum(actionable_lifts) / len(actionable_lifts) if actionable_lifts else 0.0
)

track_run(
    tracker,
    exp_name,
    run_name="rule_evaluation_three_threshold",
    params={
        "algorithm": "association_rules",
        "implementation": "from_scratch",
        "n_transactions": len(transactions),
        "min_support_mining": MIN_SUPPORT,
        "min_confidence_filter": MIN_CONFIDENCE,
        "actionable_min_support": 0.03,
        "actionable_min_confidence": 0.4,
        "actionable_min_lift": 1.5,
    },
    scalar_metrics={
        "n_frequent_itemsets": float(len(frequent)),
        "n_rules_generated": float(len(rules)),
        "n_rules_actionable": float(len(actionable)),
        "actionable_rate": float(len(actionable) / max(len(rules), 1)),
        "cross_category_rules": float(cross),
        "within_category_rules": float(within),
        "top_lift": float(top_lift),
        "mean_actionable_lift": float(mean_act_lift),
        "mean_lift_all": float(sum(lifts) / max(len(lifts), 1)),
        "mean_confidence_all": float(sum(confs) / max(len(confs), 1)),
        "mean_support_all": float(sum(supps) / max(len(supps), 1)),
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
# This lesson hand-rolled rule generation, all four quality metrics, and
# the three-threshold filter — ~210 lines of structure to internalise WHY
# each metric catches a different failure mode. The production destination
# is the same one you used in lesson 02:
#
#   from mlxtend.frequent_patterns import fpgrowth, association_rules
#
# It runs FP-Growth + every metric (support, confidence, lift, conviction,
# leverage, zhang's metric) in two calls and returns a pandas frame ready
# for the same three-threshold filter you just implemented.

from mlxtend.frequent_patterns import association_rules as mlx_rules  # noqa: E402
from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth  # noqa: E402

onehot_pd = transactions_to_onehot(transactions).to_pandas().astype(bool)
mlx_freq = mlx_fpgrowth(onehot_pd, min_support=MIN_SUPPORT, use_colnames=True)
mlx_rules_df = mlx_rules(mlx_freq, metric="confidence", min_threshold=MIN_CONFIDENCE)
mlx_actionable = mlx_rules_df[
    (mlx_rules_df["support"] >= 0.03)
    & (mlx_rules_df["confidence"] >= 0.4)
    & (mlx_rules_df["lift"] > 1.5)
]
print(f"  mlxtend rules : {len(mlx_rules_df)}  hand-rolled : {len(rules)}")
print(
    f"  mlxtend actionable: {len(mlx_actionable)}  "
    f"hand-rolled actionable: {len(actionable)}"
)
print()
print("  Same four metrics, same three-threshold filter — the engine ships")
print("  what you just built. Take the contract; ship the model card.\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Generated directional association rules from frequent itemsets
  [x] Computed support, confidence, lift, and conviction for every rule
  [x] Applied the three-threshold filter (support + confidence + lift)
  [x] Separated cross-category rules from within-category rules
  [x] Identified a production scenario (Watsons cart-page recommender)
      where the three-threshold filter IS the product spec
  [x] Reproduced the entire pipeline via two mlxtend calls — fpgrowth +
      association_rules — confirming the production destination

  KEY INSIGHT: Confidence alone is popularity. Lift alone is noise.
  Support alone is volume. The three together are actionability.

  Next: 04_rule_features.py — use the discovered rules as features for a
  supervised classifier and measure whether they beat raw product presence.
"""
)

# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
