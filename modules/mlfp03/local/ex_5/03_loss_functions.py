# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.3: Loss Functions — Focal Loss & Alpha Sweep
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Focal loss equation: FL(p_t) = -(1-p_t)^gamma * log(p_t)
#   - How gamma/alpha down-weight easy examples automatically
#   - How to sweep alpha and read the sensitivity curve
#
# PREREQUISITES: 02_sampling_strategies.py
# ESTIMATED TIME: ~25 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import polars as pl
from dotenv import load_dotenv

from shared.mlfp03.ex_5 import (
    OUTPUT_DIR,
    load_credit_splits,
    metrics_row,
    print_metrics_table,
    save_strategy_proba,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Focal Loss
# ════════════════════════════════════════════════════════════════════════
# Focal Loss (Lin et al., 2017) adds a modulating factor (1-p_t)^gamma to
# cross-entropy. When the model is confident and correct on an example,
# the modulating factor is tiny and the example nearly vanishes from the
# loss. Hard examples (p_t around 0.5) still contribute fully. Gamma=0 is
# ordinary cross-entropy; gamma=2 is canonical.
#
# LightGBM doesn't expose focal loss directly. We approximate it by
# sweeping an alpha multiplier on top of scale_pos_weight.


# ════════════════════════════════════════════════════════════════════════
# BUILD + TRAIN — alpha sweep
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()

# TODO: Derive the base scale_pos_weight from (y_train == 0).sum() / (y_train == 1).sum()
base_weight = ____

print("\n" + "=" * 70)
print("  Exercise 5.3 — Focal Loss Alpha Sweep")
print("=" * 70)
print(f"  Base scale_pos_weight (class-balanced): {base_weight:.2f}")

alpha_multipliers = [0.5, 1.0, 2.0, 5.0, 10.0]
metric_rows: list[dict] = []

for alpha in alpha_multipliers:
    # TODO: Compute pos_w for this alpha (alpha * base_weight)
    pos_w = ____

    model = lgb.LGBMClassifier(
        n_estimators=300,
        scale_pos_weight=pos_w,
        random_state=42,
        verbose=-1,
    )
    # TODO: Fit model on X_train, y_train
    ____

    y_proba = model.predict_proba(X_test)[:, 1]
    name = f"focal_alpha_{alpha:.1f}"
    save_strategy_proba(name, y_proba)

    row = metrics_row(f"alpha={alpha:.1f}", y_test, y_proba)
    row["alpha"] = float(alpha)
    metric_rows.append(row)


# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert len(metric_rows) == len(alpha_multipliers), "Must sweep every alpha"
assert all(0 <= r["auc_pr"] <= 1 for r in metric_rows), "AUC-PR in [0,1]"
print("[ok] Checkpoint 3 — alpha sweep complete\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE
# ════════════════════════════════════════════════════════════════════════

print_metrics_table(metric_rows, "Focal loss alpha sweep")

# TODO: Find the alpha with the best AUC-PR (max) and the best Brier (min)
# Hint: max(..., key=lambda r: r["auc_pr"]) and min(..., key=lambda r: r["brier"])
best_by_pr = ____
best_by_brier = ____

print(f"\n  Best AUC-PR: alpha={best_by_pr['alpha']:.1f}")
print(f"  Best Brier:  alpha={best_by_brier['alpha']:.1f}")

pl.DataFrame(metric_rows).write_parquet(OUTPUT_DIR / "focal_sweep_metrics.parquet")

# INTERPRETATION: AUC-PR and Brier usually peak at DIFFERENT alphas. This
# is the ranking-vs-calibration trade-off that the loss function alone
# cannot solve — hence the explicit calibration in 05_calibration.py.


# ════════════════════════════════════════════════════════════════════════
# APPLY — OCBC SME early-warning default detection
# ════════════════════════════════════════════════════════════════════════
# OCBC SG's ~18,000 SME loans have ~3% default rate. The hard cases are
# SMEs that look healthy until the final month. Focal loss shifts gradient
# onto these borderline borrowers, picking up subtle distress signals
# that flat cost-sensitive training would miss. Early warning 90 days
# before default = +29pp recovery rate = ~S$30K per loan recovered.

print(
    f"\n  OCBC SME implication: best recall at alpha={best_by_pr['alpha']:.1f}"
    f" = {best_by_pr['recall']:.2%}"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.3")
print("=" * 70)
print(
    """
  [x] Derived focal loss FL(p_t) = -(1-p_t)^gamma * log(p_t)
  [x] Swept alpha and recorded the sensitivity curve
  [x] Observed the AUC-PR vs Brier trade-off

  Next: 04_threshold_optimisation.py — tune the DECISION threshold from
  the cost matrix directly.
"""
)
