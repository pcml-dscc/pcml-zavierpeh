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
#   - Measure whether explicit rules add signal over raw one-hot features
#   - Attribute model importance across product vs rule feature groups
#   - See the forward connection to matrix factorisation and neural nets
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
#   3. Train — logistic regression + random forest, baseline vs combined
#   4. Visualise — metric comparison + feature importance attribution
#   5. Apply — NTUC Link high-value shopper scoring for CRM
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

# ── Kailash-ML ExperimentTracker — every association-rules run logs here ─
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Rules as Handcrafted Co-Occurrence Features
# ════════════════════════════════════════════════════════════════════════
# A linear model trained only on product-presence one-hot features sees
# each product as independent. It cannot represent "this basket contains
# the breakfast bundle" as a single signal — at best it learns a weight
# per product and adds them up.
#
# Association rules give you that missing structure for free. Every
# actionable rule X -> Y becomes one or two new binary features:
#
#   ant_present[r]   = 1 if the basket contains X
#   full_present[r]  = 1 if the basket contains X AND Y
#
# Plus a handful of aggregates:
#
#   n_rules_triggered  = how many rules fire on this basket
#   total_rule_lift    = sum of lifts of the rules that fire
#   max_rule_lift      = strongest rule that fires
#
# This is the MANUAL version of what matrix factorisation (Ex 7) and
# neural networks (Ex 8) will do AUTOMATICALLY. The progression is:
#
#   Manual rules  ->  Linear factorisation  ->  Nonlinear neural nets
#   explicit         compressed latent         learned non-linearity
#   interpretable    partially interpretable   least interpretable
#   cheap            medium                    expensive
#
# Ex 5.4 is the bottom rung. Every step above this file is a different
# way to discover co-occurrence structure without hand-writing rules.


# ════════════════════════════════════════════════════════════════════════
# MINI-APRIORI + RULE GENERATION (inline, so this file stands alone)
# ════════════════════════════════════════════════════════════════════════


def _apriori(
    transactions: list[set[str]], min_support: float
) -> dict[frozenset[str], float]:
    """Inline Apriori — same contract as 01_apriori_from_scratch.apriori()."""
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
    """Turn each transaction into a row of rule-derived numeric features.

    Columns produced:
      - ``rule{i}_antecedent`` — 1 if the antecedent is present
      - ``rule{i}_full``       — 1 if antecedent AND consequent are present
      - ``n_rules_triggered``  — count of fully-matched rules
      - ``total_rule_lift_x100`` — sum of matched rule lifts (scaled int)
      - ``max_rule_lift_x100``   — max matched rule lift (scaled int)
    """
    rows: list[dict[str, int]] = []
    for txn in transactions:
        txn_set = frozenset(txn)
        row: dict[str, int] = {}
        total_lift = 0.0
        n_triggered = 0
        max_lift = 0.0

        for idx, rule in enumerate(rules):
            ant_present = int(rule["antecedent"].issubset(txn_set))
            full_present = int(
                (rule["antecedent"] | rule["consequent"]).issubset(txn_set)
            )
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
# TASK 3 — TRAIN: baseline vs combined classifiers
# ════════════════════════════════════════════════════════════════════════

transactions = generate_transactions(n=2500, seed=42)
print_transaction_summary(transactions)

# Target = "high-value shopper" (basket size >= 6 items).
basket_sizes = np.array([len(t) for t in transactions])
HIGH_VALUE_THRESHOLD = 6
y = (basket_sizes >= HIGH_VALUE_THRESHOLD).astype(int)
print(f"\n  Target: high-value shopper (basket >= {HIGH_VALUE_THRESHOLD} items)")
print(f"  Positive rate: {y.mean():.1%}")

# --- Baseline features: raw product presence ---
onehot = transactions_to_onehot(transactions)
X_baseline = onehot.to_numpy().astype(np.float64)
print(f"\n  Baseline features (product presence): {X_baseline.shape[1]}")

# --- Mine rules and engineer features ---
print("\n=== Mining rules for feature engineering ===")
freq = _apriori(transactions, min_support=0.03)
rules = _rules_from_itemsets(freq, min_confidence=0.4, min_lift=1.5)
top_rules = rules[:20]
print(f"  Actionable rules found: {len(rules)}")
print(f"  Using top {len(top_rules)} rules as features")

rule_df = engineer_rule_features(transactions, top_rules)
X_rules = rule_df.to_numpy().astype(np.float64)
X_combined = np.hstack([X_baseline, X_rules])
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

product_importance = float(importances[: X_baseline.shape[1]].sum())
rule_importance = float(importances[X_baseline.shape[1] :].sum())
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

# Metric comparison frame (for notebooks to chart)
metric_rows = [{"model": k, **v} for k, v in results.items()]
metric_df = pl.DataFrame(metric_rows)
metric_df.write_csv(OUTPUT_DIR / "rule_features_metrics.csv")
print(f"\n  Saved: {OUTPUT_DIR / 'rule_features_metrics.csv'}")

# ── Visualisation ─────────────────────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# (A) Feature importance: product vs rule groups (pie + top-15 bar)
fig_imp = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "pie"}, {"type": "xy"}]],
    subplot_titles=["Importance by Group", "Top 15 Features"],
)
fig_imp.add_trace(
    go.Pie(
        labels=["Product Features", "Rule Features"],
        values=[product_importance, rule_importance],
        marker_colors=["#636EFA", "#EF553B"],
        textinfo="label+percent",
    ),
    row=1,
    col=1,
)
top_names = [all_feature_names[i] for i in top_idx]
top_vals = [float(importances[i]) for i in top_idx]
top_colors = ["#636EFA" if idx < X_baseline.shape[1] else "#EF553B" for idx in top_idx]
fig_imp.add_trace(
    go.Bar(
        x=top_vals,
        y=top_names,
        orientation="h",
        marker_color=top_colors,
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig_imp.update_layout(
    title="Feature Importance: Product (blue) vs Rule (red) Features",
    height=500,
    width=1000,
)
fig_imp.update_yaxes(autorange="reversed", row=1, col=2)
imp_path = OUTPUT_DIR / "04_feature_importance.html"
fig_imp.write_html(str(imp_path))
print(f"[viz] Feature importance: {imp_path}")

# (B) Accuracy improvement: baseline vs combined across models
model_names = list(results.keys())
acc_vals = [results[k]["accuracy"] for k in model_names]
auc_vals = [results[k]["auc_roc"] for k in model_names]
fig_acc = go.Figure()
fig_acc.add_trace(
    go.Bar(
        x=model_names,
        y=acc_vals,
        name="Accuracy",
        marker_color="#636EFA",
        text=[f"{v:.3f}" for v in acc_vals],
        textposition="outside",
    )
)
fig_acc.add_trace(
    go.Bar(
        x=model_names,
        y=auc_vals,
        name="AUC-ROC",
        marker_color="#EF553B",
        text=[f"{v:.3f}" for v in auc_vals],
        textposition="outside",
    )
)
fig_acc.update_layout(
    title="Model Comparison: Baseline vs Rules-Only vs Combined",
    xaxis_title="Model Variant",
    yaxis_title="Score",
    barmode="group",
    yaxis_range=[0, 1.1],
)
acc_path = OUTPUT_DIR / "04_accuracy_comparison.html"
fig_acc.write_html(str(acc_path))
print(f"[viz] Accuracy comparison: {acc_path}")

# INTERPRETATION: For a simple linear model, rule features typically add
# 2-5 points of AUC because the LR cannot represent "all three breakfast
# items present" without an explicit interaction. For a random forest,
# the lift is smaller (sometimes zero) because the tree already learns
# interactions implicitly via branching. This is the exact reason Ex 7
# (matrix factorisation) and Ex 8 (neural nets) exist — they learn the
# same co-occurrence structure WITHOUT requiring you to pre-specify rules.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: NTUC Link high-value shopper scoring for CRM
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: NTUC Link is Singapore's largest grocery loyalty programme
# (~1.7M members). The CRM team wants a model that scores each basket
# right after checkout, flagging the top 20% of baskets as "high value"
# for targeted next-trip offers (e.g., free delivery, priority slots).
#
# Two constraints drive the design:
#   - The model must be AUDITABLE (regulator + internal governance)
#   - The feature engineering must be REPRODUCIBLE (monthly re-training)
#
# Rule-based features meet both. Every feature column has a business
# name you can trace to a specific association rule, and the mining
# pipeline reruns monthly against the previous 90 days of baskets.
# Matrix factorisation (Ex 7) would score slightly higher on AUC but
# loses the per-feature interpretability that the CRM team needs when
# explaining a segment to senior management.
#
# BUSINESS IMPACT: A 3-5% improvement in high-value precision at the
# top 20% cutoff translates to roughly 25,000-42,000 members receiving
# targeted offers who would otherwise have been missed. Typical offer
# ROI at that scale is S$4-6 per activated member — ~S$100K-250K/month
# in incremental campaign contribution. On top of that, the rule-based
# column names feed directly into the "why did this shopper qualify?"
# explainability panel the CRM team shows regulators.
#
# LIMITATIONS:
#   - Rules captured here are support >= 3%, which is about the floor
#     for a 1.7M-member base; rarer categories (baby care, pet food)
#     need their own mining run with lower thresholds
#   - Target is basket size, a proxy for value — for true $-value
#     segmentation, weight by SKU price (out of scope here)


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log baseline-vs-rule-enhanced metrics to ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Logs all four model variants side by side (LR Baseline / LR Combined /
# RF Baseline / RF Combined) plus the product-vs-rule importance split.
# Series = top-15 RF importances so the M4 dashboard can visualise the
# combined-model ranking alongside the rule-quality distribution from
# lesson 03.

lr_combined_lift = (
    results["LR: Combined"]["auc_roc"] - results["LR: Baseline"]["auc_roc"]
)
rf_combined_lift = (
    results["RF: Combined"]["auc_roc"] - results["RF: Baseline"]["auc_roc"]
)

track_run(
    tracker,
    exp_name,
    run_name="rule_features_vs_baseline",
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
        "lr_baseline_f1": float(results["LR: Baseline"]["f1"]),
        "lr_combined_f1": float(results["LR: Combined"]["f1"]),
        "rf_baseline_f1": float(results["RF: Baseline"]["f1"]),
        "rf_combined_f1": float(results["RF: Combined"]["f1"]),
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
# This lesson hand-rolled the rule-feature builder, the per-rule binary
# columns, the aggregate (n_rules_triggered / total_lift / max_lift) +
# four full sklearn classifier runs — ~340 lines to internalise rules-as-
# features end-to-end. The destination-first surface in Kailash is two
# engines stacked:
#
#   FeatureEngineer.generate(...) -> auto co-occurrence + lag features
#   TrainingPipeline.train(schema, model_spec, eval_spec) -> ranked models
#
# In production you ship the rule-feature columns (kept verbatim for the
# auditability story above) PLUS the FeatureEngineer-generated set, then
# let TrainingPipeline train + rank LR vs RF vs gradient-boosted variants
# in one call. Manual rule features become one feature group among many,
# which is exactly the spectrum-of-discovery story this lesson opened on.

print("  Destination contract:")
print("    FeatureEngineer().generate(df)             -> auto co-occurrence cols")
print("    TrainingPipeline().train(schema, model, eval) -> ranked classifiers")
print(
    f"  Today's run: hand-built {X_rules.shape[1]} rule features, "
    f"trained 4 sklearn variants — top AUC = {max(auc_vals):.4f}"
)
print()
print("  Manual rules are the bottom rung. The engine ships every rung above.\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Engineered numeric features from discovered association rules
  [x] Compared baseline product-presence vs rule-enhanced models
  [x] Attributed feature importance across product and rule groups
  [x] Identified a production scenario (NTUC Link CRM scoring) where
      explicit rule features are preferred over learned embeddings
  [x] Pointed at the Kailash destination — FeatureEngineer +
      TrainingPipeline — that ships rule features alongside auto-discovered
      co-occurrence structure in one call

  KEY INSIGHT: Association rules are the MANUAL end of a spectrum that
  runs all the way to deep learning.

    Manual rules  ->  Linear factorisation  ->  Nonlinear neural nets
    (this file)      (Ex 7)                    (Ex 8)

  Every technique to the right of this one discovers the same
  co-occurrence structure, just with less human steering and less
  interpretability.

  Next: Exercise 6 moves to UNSTRUCTURED text — TF-IDF from scratch,
  NMF topic modelling, and topic quality evaluation via NPMI coherence.
"""
)

# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
