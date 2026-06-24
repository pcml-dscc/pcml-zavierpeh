# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.4: Threshold Optimisation from a Cost Matrix
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why threshold=0.5 is wrong for asymmetric costs
#   - Bayes-optimal threshold: t* = cost_FP / (cost_FP + cost_FN)
#   - Empirical threshold sweep on a fixed probability vector
#   - Translating the chosen threshold into annual S$ savings
#
# PREREQUISITES: 02_sampling_strategies.py (cost-sensitive proba saved)
# ESTIMATED TIME: ~25 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from dotenv import load_dotenv

from shared.mlfp03.ex_5 import (
    ANNUAL_APPLICATIONS,
    DEFAULT_COSTS,
    OUTPUT_DIR,
    annual_roi,
    load_credit_splits,
    load_strategy_proba,
    metrics_row,
    print_metrics_table,
    print_roi,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Decision theory for asymmetric costs
# ════════════════════════════════════════════════════════════════════════
# Expected cost at threshold t minimises at:
#     t* = cost_FP / (cost_FP + cost_FN)
# For our 100:1 cost matrix: t* = 100 / 10100 ≈ 0.0099. Decline any
# applicant with >=1% default probability. The default 0.5 is wrong by
# two orders of magnitude.


# ════════════════════════════════════════════════════════════════════════
# BUILD — load saved probabilities, sweep thresholds
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()

# TODO: Load the cost-sensitive probabilities saved in exercise 5.2
# Hint: use load_strategy_proba("cost_sensitive_scale")
y_proba = ____

print("\n" + "=" * 70)
print("  Exercise 5.4 — Threshold Optimisation")
print("=" * 70)
print(f"  Cost matrix: FP=${DEFAULT_COSTS.fp:,.0f}, FN=${DEFAULT_COSTS.fn:,.0f}")
print(f"  Bayes-optimal theoretical threshold: {DEFAULT_COSTS.optimal_threshold:.4f}")


# ════════════════════════════════════════════════════════════════════════
# TRAIN — tune decision rule (no model training — just a threshold sweep)
# ════════════════════════════════════════════════════════════════════════

thresholds = np.arange(0.005, 0.500, 0.005)
sweep_rows: list[dict] = []
for t in thresholds:
    # TODO: Predict labels at threshold t
    # Hint: (y_proba >= t).astype(int)
    y_pred = ____

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())

    # TODO: Compute total_cost = fp * cost_FP + fn * cost_FN
    # Hint: use DEFAULT_COSTS.fp and DEFAULT_COSTS.fn
    cost = ____

    sweep_rows.append(
        {
            "threshold": float(t),
            "total_cost_usd": float(cost),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    )

# TODO: Find the row with the minimum total_cost_usd
# Hint: min(sweep_rows, key=lambda r: r["total_cost_usd"])
best_row = ____
best_t = best_row["threshold"]

# Compute the t=0.5 reference cost
y_pred_default = (y_proba >= 0.5).astype(int)
fp_d = int(((y_pred_default == 1) & (y_test == 0)).sum())
fn_d = int(((y_pred_default == 0) & (y_test == 1)).sum())
default_cost = fp_d * DEFAULT_COSTS.fp + fn_d * DEFAULT_COSTS.fn


# ── Checkpoint 4 ────────────────────────────────────────────────────────
assert 0.0 < best_t < 1.0, "Best threshold must be in (0,1)"
assert (
    best_row["total_cost_usd"] <= default_cost
), "Optimised threshold should not cost more than t=0.5"
print("[ok] Checkpoint 4 — threshold sweep and argmin computed\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE
# ════════════════════════════════════════════════════════════════════════

print(f"\n  {'threshold':>10} {'total_cost':>14} {'fp':>6} {'fn':>6}")
print("  " + "─" * 44)
for r in sweep_rows[::10]:
    print(
        f"  {r['threshold']:>10.3f} ${r['total_cost_usd']:>12,.0f} "
        f"{r['fp']:>6} {r['fn']:>6}"
    )

print(f"\n  Cost at t=0.5:       ${default_cost:,.0f}")
print(f"  Cost at t*={best_t:.3f}: ${best_row['total_cost_usd']:,.0f}")
print(f"  Savings: ${default_cost - best_row['total_cost_usd']:,.0f}")

pl.DataFrame(sweep_rows).write_parquet(OUTPUT_DIR / "threshold_sweep.parquet")

row = metrics_row("Cost-sens @ t*", y_test, y_proba, threshold=best_t)
print_metrics_table([row], f"Metrics at optimised threshold t*={best_t:.3f}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Maybank Singapore unsecured-loan ROI
# ════════════════════════════════════════════════════════════════════════
# Scale the test confusion matrix to Maybank's 100K annual applications.

# TODO: Compute annual ROI at the default t=0.5 and at the optimised t*
# Hint: annual_roi(y_test, y_proba, threshold=..., annual_volume=ANNUAL_APPLICATIONS)
roi_default = ____
roi_best = ____

print_roi("Annual ROI @ t=0.5", roi_default)
print_roi(f"Annual ROI @ t*={best_t:.3f}", roi_best)

delta = roi_best["annual_savings_usd"] - roi_default["annual_savings_usd"]
print(f"\n  Threshold tuning alone adds: ${delta:,.0f}/year in savings")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.4")
print("=" * 70)
print(
    """
  [x] Derived the Bayes-optimal threshold t*
  [x] Ran an empirical sweep and found the argmin of total cost
  [x] Quantified dollar savings of threshold tuning alone

  Next: 05_calibration.py — Platt/Isotonic calibration + final comparison.
"""
)
