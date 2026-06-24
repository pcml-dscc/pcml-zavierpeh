# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.2: Sampling Strategies — SMOTE vs Cost-Sensitive
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - How SMOTE generates synthetic samples and where it breaks
#   - Cost-sensitive learning via scale_pos_weight + sample_weight
#   - Why cost-sensitive dominates SMOTE on tabular finance data
#
# PREREQUISITES: 01_metrics_and_baseline.py
# ESTIMATED TIME: ~30 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE

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
# THEORY — SMOTE failure modes
# ════════════════════════════════════════════════════════════════════════
# SMOTE creates synthetic minority samples by k-NN interpolation. It fails in
# production for three reasons: (1) Lipschitz violation — the midpoint between
# two real defaulters may not resemble any real applicant; (2) Noise — errors
# in minority rows get amplified; (3) Dimensionality — in 45-feature space,
# nearest neighbours are nearly equidistant so interpolation is meaningless.
#
# Cost-sensitive learning changes the LOSS instead of the data:
#   - scale_pos_weight = n_neg / n_pos (class-balanced)
#   - sample_weight from the dollar cost matrix (most general)


# ════════════════════════════════════════════════════════════════════════
# BUILD — three strategies
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()

print("\n" + "=" * 70)
print("  Exercise 5.2 — Sampling Strategies")
print("=" * 70)

# --- SMOTE --------------------------------------------------------------
# TODO: Fit imblearn SMOTE on X_train, y_train with random_state=42
# Hint: smote.fit_resample(X, y) returns (X_resampled, y_resampled)
smote = SMOTE(random_state=42)
X_smote, y_smote = ____

smote_model = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)

# --- Cost-sensitive (A) scale_pos_weight --------------------------------
# TODO: Derive scale_weight from pos_rate
# Hint: (1 - p) / p — this equals n_neg / n_pos
scale_weight = ____
cost_a_model = lgb.LGBMClassifier(
    n_estimators=300,
    scale_pos_weight=____,
    random_state=42,
    verbose=-1,
)

# --- Cost-sensitive (B) sample_weight -----------------------------------
# TODO: Build sample_weights from the business cost matrix
# Hint: np.where(y_train == 1, DEFAULT_COSTS.fn, DEFAULT_COSTS.fp)
sample_weights = ____
cost_b_model = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)


# ════════════════════════════════════════════════════════════════════════
# TRAIN — fit all three
# ════════════════════════════════════════════════════════════════════════

# TODO: Fit smote_model on the SMOTE-resampled data
____

# TODO: Fit cost_a_model on the ORIGINAL X_train, y_train
____

# TODO: Fit cost_b_model on X_train, y_train with sample_weight=sample_weights
____

y_proba_smote = smote_model.predict_proba(X_test)[:, 1]
y_proba_cost_a = cost_a_model.predict_proba(X_test)[:, 1]
y_proba_cost_b = cost_b_model.predict_proba(X_test)[:, 1]

save_strategy_proba("smote", y_proba_smote)
save_strategy_proba("cost_sensitive_scale", y_proba_cost_a)
save_strategy_proba("cost_sensitive_matrix", y_proba_cost_b)


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert len(y_smote) > len(y_train), "SMOTE must increase dataset size"
assert y_smote.mean() > pos_rate, "SMOTE must rebalance minority class"
assert scale_weight > 1.0, "scale_pos_weight must up-weight the minority class"
print("[ok] Checkpoint 2 — three imbalance strategies trained\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE
# ════════════════════════════════════════════════════════════════════════

rows = [
    metrics_row("SMOTE", y_test, y_proba_smote),
    metrics_row("Cost-sens (scale_pos)", y_test, y_proba_cost_a),
    metrics_row("Cost-sens (matrix)", y_test, y_proba_cost_b),
]
print_metrics_table(rows, "Sampling strategy comparison (threshold=0.5)")

pl.DataFrame(rows).write_parquet(OUTPUT_DIR / "sampling_metrics.parquet")

# INTERPRETATION: Watch the Brier column. Cost-sensitive usually wins on
# calibration; SMOTE often damages it. Credit scoring cares about
# calibration because loans are priced from the predicted probability.


# ════════════════════════════════════════════════════════════════════════
# APPLY — UOB card-fraud detection
# ════════════════════════════════════════════════════════════════════════
# UOB SG processes ~S$28B in card transactions per year. A SMOTE fraud
# model would produce synthetic rows in 45-D feature space that don't
# match any real cardholder. Precision collapses, legitimate customers
# get declined at Chanel/Marina Bay Sands, and the model is rolled back.
# Cost-sensitive learning delivers the same recall with better Brier.

worst_brier = max(r["brier"] for r in rows)
best_brier = min(r["brier"] for r in rows)
print(f"\n  Brier gap between best/worst: {worst_brier - best_brier:+.4f}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.2")
print("=" * 70)
print(
    """
  [x] Ran SMOTE and observed class-balance change
  [x] Trained two cost-sensitive variants (scale_pos_weight, sample_weight)
  [x] Compared all three on the full metrics taxonomy
  [x] Saw why cost-sensitive beats SMOTE on calibration

  Next: 03_loss_functions.py — focal loss sweeps alpha to down-weight
  easy examples automatically.
"""
)
