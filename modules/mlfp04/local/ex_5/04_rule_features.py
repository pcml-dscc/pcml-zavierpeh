# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 5.4: Rule-Based Features for Supervised Classification
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Engineer features from discovered association rules
#   - Compare a product-presence baseline against a rule-enhanced model
#   - Attribute feature importance across product vs rule groups
#
# PREREQUISITES:
#   - 01_apriori_from_scratch.py
#   - 03_rule_evaluation.py
#   - MLFP03 Exercise 1 (feature engineering)
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — rules as handcrafted co-occurrence features
#   2. Build — turn actionable rules into numeric feature columns
#   3. Train — LR + RF, baseline vs rules-only vs combined
#   4. Visualise — metric comparison + feature importance attribution
#   5. Apply — NTUC Link high-value shopper scoring
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shared.mlfp04.ex_5 import (
    OUTPUT_DIR,
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
# THEORY — Rules as Handcrafted Co-Occurrence Features
# ════════════════════════════════════════════════════════════════════════
# A linear model on raw one-hot product features sees each product as
# independent. Association rules inject interaction structure that the
# linear model cannot represent on its own.
#
# The forward connection:
#   Manual rules  ->  Linear factorisation (Ex 7)  ->  Neural nets (Ex 8)
# Each step replaces more of the human steering with learned structure.


# ════════════════════════════════════════════════════════════════════════
# MINI-APRIORI + RULE GENERATION (inline, file stands alone)
# ════════════════════════════════════════════════════════════════════════


def _apriori(
    transactions: list[set[str]], min_support: float
) -> dict[frozenset[str], float]:
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


def _rules_from_itemsets(
    freq: dict[frozenset[str], float],
    min_confidence: float = 0.4,
    min_lift: float = 1.5,
) -> list[dict]:
    rules: list[dict] = []
    for itemset, support in freq.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for ant_tuple in combinations(items, r):
                antecedent = frozenset(ant_tuple)
                consequent = itemset - antecedent
                supp_ant = freq.get(antecedent)
                supp_con = freq.get(consequent)
                if supp_ant is None or supp_con is None:
                    continue
                confidence = support / supp_ant
                lift = confidence / supp_con
                if confidence >= min_confidence and lift > min_lift:
                    rules.append(
                        {
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "support": support,
                            "confidence": confidence,
                            "lift": lift,
                        }
                    )
    rules.sort(key=lambda r: -r["lift"])
    return rules


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: rule-based feature engineer
# ════════════════════════════════════════════════════════════════════════


def engineer_rule_features(
    transactions: list[set[str]],
    rules: list[dict],
) -> pl.DataFrame:
    """Turn each transaction into rule-derived numeric features.

    Columns produced per rule:
      - rule{i}_antecedent — 1 if antecedent present
      - rule{i}_full       — 1 if antecedent AND consequent present

    Plus aggregates: n_rules_triggered, total_rule_lift_x100,
    max_rule_lift_x100.
    """
    rows: list[dict[str, int]] = []
    for txn in transactions:
        txn_set = frozenset(txn)
        row: dict[str, int] = {}
        total_lift = 0.0
        n_triggered = 0
        max_lift = 0.0

        for idx, rule in enumerate(rules):
            # TODO: compute ant_present (is the antecedent a subset of txn?)
            # and full_present (is antecedent UNION consequent a subset?).
            # Hint: frozenset.issubset(txn_set); union with the | operator.
            ant_present = ____
            full_present = ____

            row[f"rule{idx}_antecedent"] = ant_present
            row[f"rule{idx}_full"] = full_present

            if full_present:
                n_triggered += 1
                total_lift += float(rule["lift"])
                if float(rule["lift"]) > max_lift:
                    max_lift = float(rule["lift"])

        row["n_rules_triggered"] = n_triggered
        row["total_rule_lift_x100"] = int(total_lift * 100)
        row["max_rule_lift_x100"] = int(max_lift * 100)
        rows.append(row)

    return pl.DataFrame(rows).fill_null(0)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: baseline vs rules-only vs combined
# ════════════════════════════════════════════════════════════════════════

transactions = generate_transactions(n=2500, seed=42)
print_transaction_summary(transactions)

basket_sizes = np.array([len(t) for t in transactions])
HIGH_VALUE_THRESHOLD = 6
y = (basket_sizes >= HIGH_VALUE_THRESHOLD).astype(int)
print(f"\n  Target: high-value shopper (basket >= {HIGH_VALUE_THRESHOLD} items)")
print(f"  Positive rate: {y.mean():.1%}")

# Baseline features
onehot = transactions_to_onehot(transactions)
X_baseline = onehot.to_numpy().astype(np.float64)
print(f"\n  Baseline features (product presence): {X_baseline.shape[1]}")

# Mine rules and engineer rule features
print("\n=== Mining rules for feature engineering ===")
freq = _apriori(transactions, min_support=0.03)
rules = _rules_from_itemsets(freq, min_confidence=0.4, min_lift=1.5)
top_rules = rules[:20]
print(f"  Actionable rules found: {len(rules)}")
print(f"  Using top {len(top_rules)} rules as features")

rule_df = engineer_rule_features(transactions, top_rules)
X_rules = rule_df.to_numpy().astype(np.float64)

# TODO: stack baseline + rule features horizontally into X_combined.
# Hint: np.hstack([...])
X_combined = ____

print(f"  Rule features:     {X_rules.shape[1]}")
print(f"  Combined features: {X_combined.shape[1]}")


def _split_and_scale(X: np.ndarray, y: np.ndarray, scale: bool):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    if scale:
        s = StandardScaler()
        X_tr = s.fit_transform(X_tr)
        X_te = s.transform(X_te)
    return X_tr, X_te, y_tr, y_te


results: dict[str, dict[str, float]] = {}

print("\n=== Model: Logistic Regression ===")
for name, X in [
    ("Baseline", X_baseline),
    ("Rules only", X_rules),
    ("Combined", X_combined),
]:
    X_tr, X_te, y_tr, y_te = _split_and_scale(X, y, scale=True)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)
    y_proba = lr.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    results[f"LR: {name}"] = {"accuracy": acc, "f1": f1, "auc_roc": auc}
    print(f"  {name:<12} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

print("\n=== Model: Random Forest ===")
for name, X in [
    ("Baseline", X_baseline),
    ("Rules only", X_rules),
    ("Combined", X_combined),
]:
    X_tr, X_te, y_tr, y_te = _split_and_scale(X, y, scale=False)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    y_proba = rf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    results[f"RF: {name}"] = {"accuracy": acc, "f1": f1, "auc_roc": auc}
    print(f"  {name:<12} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    X_combined.shape[1] > X_baseline.shape[1]
), "Combined should add columns to the baseline, not replace them"
assert X_rules.shape[1] > 0, "Should have produced at least one rule feature"
lr_baseline = results["LR: Baseline"]["auc_roc"]
lr_combined = results["LR: Combined"]["auc_roc"]
rf_baseline = results["RF: Baseline"]["auc_roc"]
assert lr_baseline > 0.5, "Baseline LR should beat random"
assert rf_baseline > 0.5, "Baseline RF should beat random"
assert (
    lr_combined >= lr_baseline - 0.05
), "Adding rule features should not significantly regress LR"
print("\n[ok] Checkpoint passed — rule-enhanced model trained + compared\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: feature importance attribution
# ════════════════════════════════════════════════════════════════════════

X_tr, X_te, y_tr, y_te = _split_and_scale(X_combined, y, scale=False)
rf_combined = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
rf_combined.fit(X_tr, y_tr)
importances = rf_combined.feature_importances_

# TODO: split the RF importance vector into the product-feature sum and
# the rule-feature sum. The first X_baseline.shape[1] entries are
# product features; everything after is rule features.
product_importance = ____
rule_importance = ____
total = product_importance + rule_importance

print("=== Feature Importance Attribution (RF combined) ===")
print(f"  Product features contribute: {product_importance / total:.1%}")
print(f"  Rule features contribute:    {rule_importance / total:.1%}")

all_feature_names = list(onehot.columns) + list(rule_df.columns)
top_idx = np.argsort(importances)[::-1][:15]
print("\n  Top 15 features in the combined model:")
for idx in top_idx:
    fname = all_feature_names[idx]
    ftype = "product" if idx < X_baseline.shape[1] else "rule"
    print(f"    [{ftype:>7}] {fname:<40} {importances[idx]:.4f}")

metric_rows = [{"model": k, **v} for k, v in results.items()]
pl.DataFrame(metric_rows).write_csv(OUTPUT_DIR / "rule_features_metrics.csv")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: NTUC Link high-value shopper scoring
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: NTUC Link (~1.7M members) needs an auditable, reproducible
# model to flag the top 20% of baskets as high-value for targeted
# next-trip offers. Rule-based features give the CRM team the per-feature
# interpretability that pure matrix factorisation would hide.
#
# BUSINESS IMPACT: 3-5% precision improvement at the top 20% cutoff
# reaches 25K-42K extra members/month; S$4-6 offer ROI per member
# ~= S$100K-250K/month in incremental campaign contribution.


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log baseline-vs-rule-enhanced classifier metrics
# ════════════════════════════════════════════════════════════════════════
# Same shared experiment as 01-03. Logs all four model variants side by
# side (LR Baseline / LR Combined / RF Baseline / RF Combined) plus the
# product-vs-rule importance split. Series = top-15 RF importances.

# TODO: pick run_name="rule_features_vs_baseline" and fill in the two
# AUC-lift scalars (combined - baseline) for LR and RF.
lr_combined_lift = ____
rf_combined_lift = ____

track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "algorithm": "rule_features",
        "implementation": "lr_plus_rf_baseline_vs_combined",
        "n_transactions": len(transactions),
        "n_baseline_features": int(X_baseline.shape[1]),
        "n_rule_features": int(X_rules.shape[1]),
        "n_combined_features": int(X_combined.shape[1]),
        "rf_n_estimators": 200,
        "rf_max_depth": 10,
    },
    scalar_metrics={
        "lr_baseline_auc": float(results["LR: Baseline"]["auc_roc"]),
        "lr_combined_auc": float(results["LR: Combined"]["auc_roc"]),
        "lr_auc_lift_combined_vs_baseline": float(lr_combined_lift),
        "rf_baseline_auc": float(results["RF: Baseline"]["auc_roc"]),
        "rf_combined_auc": float(results["RF: Combined"]["auc_roc"]),
        "rf_auc_lift_combined_vs_baseline": float(rf_combined_lift),
        "product_importance_share": float(product_importance / total),
        "rule_importance_share": float(rule_importance / total),
    },
    series_metrics={
        "rf_top15_importances": [float(importances[i]) for i in top_idx],
    },
)
print(f"  [tracked] Baseline-vs-combined classifier metrics logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — kailash-ml FeatureEngineer + TrainingPipeline
# ════════════════════════════════════════════════════════════════════════
# The hand-built rule-feature columns made every interaction explicit
# (auditability win). The Kailash destination stacks two engines: ship
# the rule columns alongside auto-discovered co-occurrence features from
# FeatureEngineer, then let TrainingPipeline train + rank LR vs RF vs
# gradient-boosted variants in one call. Manual rules become one feature
# group among many — the bottom rung of the discovery spectrum.

print("  Destination contract:")
print("    FeatureEngineer().generate(df)             -> auto co-occurrence cols")
print("    TrainingPipeline().train(schema, model, eval) -> ranked classifiers")
print(
    f"  Today's run: {X_rules.shape[1]} hand-built rule features + "
    f"4 sklearn variants — top AUC = {max(v['auc_roc'] for v in results.values()):.4f}"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Engineered numeric features from discovered association rules
  [x] Compared baseline vs rules-only vs combined models
  [x] Attributed feature importance across product and rule groups
  [x] Pointed at the Kailash destination — FeatureEngineer +
      TrainingPipeline — that ships rule features alongside auto-
      discovered co-occurrence structure in one call

  KEY INSIGHT: Manual rules are the MANUAL end of a spectrum that runs
  all the way to deep learning.
    Manual rules  ->  Linear factorisation (Ex 7)  ->  Neural nets (Ex 8)

  Next: Exercise 6 moves to UNSTRUCTURED text — TF-IDF from scratch,
  NMF topic modelling, and NPMI coherence.
"""
)

# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
