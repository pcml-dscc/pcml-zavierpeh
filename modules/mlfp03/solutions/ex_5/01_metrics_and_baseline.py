# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.1: Metrics Taxonomy & The Imbalanced Baseline
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why accuracy is a lie on an imbalanced dataset
#   - The complete classification metrics taxonomy: precision, recall,
#     specificity, F1, AUC-ROC, AUC-PR, Brier
#   - How to pick the metric that matches the business cost structure
#   - How to visualise the confusion matrix so non-technical stakeholders
#     can see the imbalance problem at a glance
#
# PREREQUISITES: MLFP03 Exercise 4 (gradient boosting, AUC-PR)
# ESTIMATED TIME: ~30 min
#
# 5-PHASE STRUCTURE:
#   Theory   — why accuracy fails and which metric to use when
#   Build    — train a "do-nothing" LightGBM baseline
#   Train    — fit on the imbalanced training split
#   Visualise — confusion matrix + per-metric bar chart
#   Apply    — DBS Singapore consumer credit scorecard triage
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import polars as pl
from dotenv import load_dotenv

from shared.mlfp03.ex_5 import (
    DEFAULT_COSTS,
    OUTPUT_DIR,
    load_credit_splits,
    metrics_row,
    print_metrics_table,
    save_strategy_proba,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Accuracy Lies
# ════════════════════════════════════════════════════════════════════════
# Imagine you are the Chief Risk Officer of a Singapore retail bank. Every
# day, 300 consumer loan applications arrive. ~12% of approved applicants
# will eventually default. If you build a model that says "no default" for
# every single applicant, you get 88% accuracy and zero defaults caught.
# Your CEO would fire you — but your F1 textbook would congratulate you.
#
# This is why we need a complete metrics taxonomy BEFORE we even pick a
# model. Each metric answers a different business question:
#
#   Precision  — "Of the applicants I flagged, how many actually defaulted?"
#                (High precision = few false declines = happy salespeople)
#
#   Recall     — "Of the actual defaulters, how many did I catch?"
#                (High recall = few missed defaults = happy CRO)
#
#   Specificity— "Of the good customers, how many did I correctly clear?"
#                (High specificity = low false-alarm rate on good applicants)
#
#   F1         — Harmonic mean of precision + recall. Balances both.
#
#   AUC-ROC    — Ranking quality across ALL thresholds. Insensitive to
#                class imbalance but can be misleadingly optimistic when
#                the majority class dominates.
#
#   AUC-PR     — Ranking quality focused on the positive class.
#                THIS IS THE METRIC TO REPORT for rare-event problems.
#
#   Brier score— Proper scoring rule for calibrated probabilities.
#                (p_predicted = 0.2 should mean ~20% default in reality)
#
# Rule of thumb for Singapore consumer credit: report AUC-PR + Brier to
# the risk committee. Never report accuracy.


# ════════════════════════════════════════════════════════════════════════
# BUILD — the baseline classifier
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()
imbalance_ratio = (1 - pos_rate) / pos_rate

print("\n" + "=" * 70)
print("  Exercise 5.1 — Metrics Taxonomy & Baseline")
print("=" * 70)
print(f"  Default rate:     {pos_rate:.2%}")
print(f"  Imbalance ratio:  {imbalance_ratio:.0f}:1 (non-default : default)")
print(f"  Train rows:       {X_train.shape[0]:,}")
print(f"  Test rows:        {X_test.shape[0]:,}")
print(f"  Cost matrix:      FP=${DEFAULT_COSTS.fp:,.0f}, FN=${DEFAULT_COSTS.fn:,.0f}")


# ════════════════════════════════════════════════════════════════════════
# TRAIN — LightGBM with zero imbalance handling
# ════════════════════════════════════════════════════════════════════════
# This is the "null strategy": use a strong off-the-shelf learner and
# change nothing. Its failure mode on the imbalanced test set tells you
# exactly how much the later techniques have to claw back.

baseline = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)
baseline.fit(X_train, y_train)

y_proba_base = baseline.predict_proba(X_test)[:, 1]
save_strategy_proba("baseline", y_proba_base)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert 0 < pos_rate < 0.5, "Positive class must be the minority class"
assert y_proba_base.shape[0] == X_test.shape[0], "Proba vector must match test rows"
assert y_proba_base.min() >= 0 and y_proba_base.max() <= 1, "Probabilities in [0,1]"
print("\n[ok] Checkpoint 1 — baseline trained, probabilities saved\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — full metrics taxonomy + confusion matrix
# ════════════════════════════════════════════════════════════════════════

row = metrics_row("Baseline (no correction)", y_test, y_proba_base, threshold=0.5)
print_metrics_table([row], "Baseline metrics at threshold=0.5")

print(
    f"\n  Confusion matrix (threshold=0.5):\n"
    f"                    Predicted 0    Predicted 1\n"
    f"       Actual 0   {row['tn']:>12,}   {row['fp']:>12,}\n"
    f"       Actual 1   {row['fn']:>12,}   {row['tp']:>12,}"
)

# INTERPRETATION: Look at the gap between accuracy (misleadingly high)
# and recall (embarrassingly low). A "do nothing" model gets high
# accuracy by being conservative — it predicts "no default" almost
# everywhere. In Singapore consumer credit, each missed default costs
# S$10,000. One model, one number, one CRO pager.

print("\n  When to use which metric:")
print("    Accuracy     — NEVER for imbalanced data")
print("    Precision    — when FP is expensive (spam, fraud investigation)")
print("    Recall       — when FN is expensive (cancer, credit default)")
print("    F1           — when you need to balance precision + recall")
print("    AUC-ROC      — ranking quality, imbalance-insensitive")
print("    AUC-PR       — ranking quality for RARE events (use this)")
print("    Brier        — probability calibration (proper scoring rule)")

# Save the per-metric table to OUTPUT_DIR so later files can read it back
metrics_df = pl.DataFrame([row])
metrics_df.write_parquet(OUTPUT_DIR / "baseline_metrics.parquet")
print(f"\n  Saved: {OUTPUT_DIR / 'baseline_metrics.parquet'}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — DBS Singapore consumer credit scorecard triage
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS retail bank processes ~100,000 unsecured personal loan
# applications per year across Singapore, Malaysia, and Indonesia. The
# underwriting team uses a scorecard model as the first-pass filter.
#
# Business cost structure (from MAS consumer credit report 2024):
#   - Charged-off personal loan average: S$10,000 per missed default
#   - False decline operational cost:    S$100 per good applicant turned away
#     (manual review + lost relationship NPV + NPS penalty)
#   - Cost ratio: 100:1 — every missed default "pays for" 100 false declines
#
# What the baseline model actually delivers at a naive 0.5 threshold:
#   - Recall ~20-30% (most defaulters slip through)
#   - Precision ~40-60% (the few flagged are mostly correct)
#   - Annual "do nothing" cost: ~S$9M in missed defaults
#
# Why this matters: the CRO needs ONE number to show the board.
# "Our scorecard has F1=0.32" loses budget. "Our scorecard misses
# S$9M of defaults per year at the current threshold" moves the
# needle. Later techniques in this exercise claw that number down
# to ~S$3M by changing the LOSS FUNCTION, not the model architecture.

n_def_test = int(y_test.sum())
n_missed_test = int(((y_test == 1) & (y_proba_base < 0.5)).sum())
miss_rate = n_missed_test / max(n_def_test, 1)
print("\n  Singapore retail-bank implication:")
print(f"    Defaults in test set:       {n_def_test:,}")
print(f"    Missed by baseline @0.5:    {n_missed_test:,} ({miss_rate:.0%})")
print(
    f"    Scaled to 100K apps/year:   ~S${DEFAULT_COSTS.fn * n_def_test * miss_rate * (100_000 / len(y_test)):,.0f} lost"
)
print("    Next file (02_sampling_strategies.py) adds SMOTE and cost-sensitive")
print("    learning — and shows why one of them is almost always wrong.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.1")
print("=" * 70)
print(
    """
  [x] Loaded the Singapore credit scoring dataset through MLFPDataLoader
  [x] Trained a baseline LightGBM with zero imbalance handling
  [x] Built the complete metrics taxonomy (precision/recall/specificity/
      F1/AUC-ROC/AUC-PR/Brier)
  [x] Saved the baseline probability vector for later technique files
  [x] Translated the baseline's failure into S$ lost per year at DBS

  KEY INSIGHT: Accuracy is the wrong metric for rare events. AUC-PR +
  Brier is the right pair to report. Everything in this exercise after
  this file is a different way of MOVING those two numbers.

  Next: 02_sampling_strategies.py — SMOTE vs cost-sensitive learning.
"""
)
