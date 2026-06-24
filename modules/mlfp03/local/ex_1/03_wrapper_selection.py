# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.3: Wrapper Feature Selection (RFE + Random Forest)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Run Recursive Feature Elimination (RFE) with a Random Forest
#   - Understand how wrappers capture feature INTERACTIONS
#   - Apply to NHCS heart-failure readmission scoring
#
# PREREQUISITES: 02_filter_selection.py
# ESTIMATED TIME: ~25 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

from shared.mlfp03.ex_1 import (
    build_full_feature_frame,
    load_icu_tables,
    log_selection_run,
    prepare_selection_inputs,
    setup_tracking,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Wrappers Capture Interactions
# ════════════════════════════════════════════════════════════════════════
# A wrapper trains a model, asks it which features matter, drops the
# weakest, and repeats. Because the model sees all features together,
# it can learn that feature A is only useful in the presence of B —
# an interaction a filter method ignores.
#
# Random Forest is a strong RFE estimator because its splits natively
# capture non-linear interactions.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: assemble the estimator and RFE
# ════════════════════════════════════════════════════════════════════════

tables = load_icu_tables()
features = build_full_feature_frame(tables)
feature_cols, X_sel, y_binary = prepare_selection_inputs(features)

print("\n" + "=" * 70)
print("  Wrapper Selection — RFE with Random Forest")
print("=" * 70)
print(f"  Features: {len(feature_cols)}")
print(f"  Samples:  {X_sel.shape[0]}")

# TODO: Build a Random Forest estimator for RFE.
# Hint: RandomForestClassifier(n_estimators=100, max_depth=5,
#                              random_state=42, n_jobs=-1)
rf_estimator = ____

N_FEATURES_TO_SELECT = 15

# TODO: Wrap the estimator in RFE with n_features_to_select and step.
# Hint: RFE(estimator=rf_estimator, n_features_to_select=N_FEATURES_TO_SELECT, step=5)
rfe = ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit the RFE loop
# ════════════════════════════════════════════════════════════════════════

# TODO: Fit rfe on X_sel, y_binary.
# Hint: rfe.fit(X_sel, y_binary)
____

rfe_selected = [name for name, selected in zip(feature_cols, rfe.support_) if selected]
rfe_ranking = sorted(
    [(name, int(rank)) for name, rank in zip(feature_cols, rfe.ranking_)],
    key=lambda x: x[1],
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert (
    len(rfe_selected) == N_FEATURES_TO_SELECT
), f"Task 3: RFE must select {N_FEATURES_TO_SELECT}, got {len(rfe_selected)}"
assert all(
    f in feature_cols for f in rfe_selected
), "Task 3: invalid feature in RFE output"
print("\n[ok] Checkpoint 1 passed — RFE fit complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the ranking
# ════════════════════════════════════════════════════════════════════════

print("\n--- RFE Ranking (top 20; rank=1 means SELECTED) ---")
print(f"{'Feature':<35} {'Rank':>6}")
print("-" * 44)
for name, rank in rfe_ranking[:20]:
    marker = "  <-- selected" if rank == 1 else ""
    print(f"  {name:<33} {rank:>6}{marker}")

print(f"\n  Total RFE-selected features: {len(rfe_selected)}")
print(f"  Selected: {rfe_selected}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert rfe_ranking[0][1] == 1, "Task 4: top-ranked features should have rank=1"
print("\n[ok] Checkpoint 2 passed — RFE ranking is well-formed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4b — LOG the wrapper run
# ════════════════════════════════════════════════════════════════════════


async def log_wrapper() -> str:
    conn, tracker, exp_id = await setup_tracking()
    run_id = await log_selection_run(
        tracker,
        exp_id,
        run_name="wrapper_rfe_rf15",
        method="wrapper",
        selected_features=rfe_selected,
        total_features=len(feature_cols),
        extra_params={
            "estimator": "RandomForestClassifier",
            "n_estimators": "100",
            "max_depth": "5",
            "n_features_to_select": str(N_FEATURES_TO_SELECT),
            "step": "5",
        },
        extra_metrics={},
    )
    await conn.close()
    return run_id


run_id = asyncio.run(log_wrapper())
print(f"\n  ExperimentTracker run: {run_id}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: National Heart Centre Singapore
# ════════════════════════════════════════════════════════════════════════
# NHCS wants a 30-day readmission risk model for heart-failure
# patients. Interactions DOMINATE: ejection_fraction * diuretic_dose,
# creatinine * ACE_inhibitor, BNP * beta_blocker_dose.
#
# BUSINESS IMPACT: ~S$2.35M/year saved (3,600 x 0.04 x S$16,300).
# Selection cost ~S$24K. ~95x ROI in year one.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Configured a Random Forest estimator inside sklearn's RFE loop
  [x] Understood how wrappers promote interaction-rich features

  Next: 04_embedded_selection.py — Lasso regularisation for single-pass
  embedded selection.
"""
)
