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
# A model that says "no default" for every Singapore consumer loan
# applicant gets 88% accuracy and catches ZERO defaults. This is why
# we need the full metrics taxonomy before we pick a model:
#
#   Precision  — of flagged applicants, how many actually defaulted
#   Recall     — of actual defaulters, how many did we catch
#   AUC-PR     — ranking quality focused on the minority class
#   Brier      — probability calibration (proper scoring rule)
#
# Rule: for Singapore consumer credit, report AUC-PR + Brier. Never accuracy.


# ════════════════════════════════════════════════════════════════════════
# BUILD — load the splits
# ════════════════════════════════════════════════════════════════════════

# TODO: Call load_credit_splits() from shared.mlfp03.ex_5
# Hint: it returns (X_train, y_train, X_test, y_test, pos_rate)
X_train, y_train, X_test, y_test, pos_rate = ____

imbalance_ratio = (1 - pos_rate) / pos_rate

print("\n" + "=" * 70)
print("  Exercise 5.1 — Metrics Taxonomy & Baseline")
print("=" * 70)
print(f"  Default rate:     {pos_rate:.2%}")
print(f"  Imbalance ratio:  {imbalance_ratio:.0f}:1 (non-default : default)")
print(f"  Cost matrix:      FP=${DEFAULT_COSTS.fp:,.0f}, FN=${DEFAULT_COSTS.fn:,.0f}")


# ════════════════════════════════════════════════════════════════════════
# TRAIN — LightGBM with ZERO imbalance handling
# ════════════════════════════════════════════════════════════════════════

# TODO: Build a default LGBMClassifier with n_estimators=300, random_state=42
# Hint: verbose=-1 suppresses LightGBM's chatter
baseline = lgb.LGBMClassifier(____)

# TODO: Fit the baseline on X_train, y_train
____

# TODO: Predict POSITIVE-CLASS probabilities on X_test
# Hint: predict_proba returns shape (n, 2); you want column index 1
y_proba_base = ____

save_strategy_proba("baseline", y_proba_base)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert 0 < pos_rate < 0.5, "Positive class must be the minority class"
assert y_proba_base.shape[0] == X_test.shape[0], "Proba vector must match test rows"
assert y_proba_base.min() >= 0 and y_proba_base.max() <= 1, "Probabilities in [0,1]"
print("\n[ok] Checkpoint 1 — baseline trained, probabilities saved\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — full metrics taxonomy + confusion matrix
# ════════════════════════════════════════════════════════════════════════

# TODO: Call metrics_row() with name="Baseline (no correction)" and threshold=0.5
row = ____

print_metrics_table([row], "Baseline metrics at threshold=0.5")

print(
    f"\n  Confusion matrix (threshold=0.5):\n"
    f"                    Predicted 0    Predicted 1\n"
    f"       Actual 0   {row['tn']:>12,}   {row['fp']:>12,}\n"
    f"       Actual 1   {row['fn']:>12,}   {row['tp']:>12,}"
)

# INTERPRETATION: The gap between accuracy (high) and recall (low) is
# the imbalance problem made visible. In Singapore consumer credit,
# every missed default costs ~S$10,000.

print("\n  When to use which metric:")
print("    Accuracy     — NEVER for imbalanced data")
print("    AUC-PR       — ranking quality for RARE events (use this)")
print("    Brier        — probability calibration (proper scoring rule)")

metrics_df = pl.DataFrame([row])
metrics_df.write_parquet(OUTPUT_DIR / "baseline_metrics.parquet")
print(f"\n  Saved: {OUTPUT_DIR / 'baseline_metrics.parquet'}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — DBS Singapore consumer credit scorecard triage
# ════════════════════════════════════════════════════════════════════════
# DBS retail bank processes ~100,000 unsecured personal loan applications
# per year. A "do-nothing" baseline at t=0.5 misses most defaults. Every
# missed default = S$10,000 charge-off. Your next technique files will
# claw this back.

n_def_test = int(y_test.sum())
n_missed_test = int(((y_test == 1) & (y_proba_base < 0.5)).sum())
miss_rate = n_missed_test / max(n_def_test, 1)
print("\n  Singapore retail-bank implication:")
print(f"    Defaults in test set:       {n_def_test:,}")
print(f"    Missed by baseline @0.5:    {n_missed_test:,} ({miss_rate:.0%})")
print(
    f"    Scaled to 100K apps/year:   "
    f"~S${DEFAULT_COSTS.fn * n_def_test * miss_rate * (100_000 / len(y_test)):,.0f} lost"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.1")
print("=" * 70)
print(
    """
  [x] Loaded SG credit scoring data via MLFPDataLoader
  [x] Trained a baseline LightGBM with zero imbalance handling
  [x] Built the complete metrics taxonomy
  [x] Translated the baseline failure into S$ lost per year

  Next: 02_sampling_strategies.py — SMOTE vs cost-sensitive learning.
"""
)
