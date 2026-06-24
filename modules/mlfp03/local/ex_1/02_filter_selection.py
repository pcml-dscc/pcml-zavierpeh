# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.2: Filter Feature Selection
#                         (Mutual Information + Chi-Squared)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Use mutual information to rank features by any dependency
#   - Use chi-squared to test statistical independence
#   - Intersect top-k rankings to find ROBUST features
#   - Apply filter selection to SingHealth radiology triage
#
# PREREQUISITES: 01_feature_engineering.py (feature matrix built)
# ESTIMATED TIME: ~25 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from shared.mlfp03.ex_1 import (
    build_full_feature_frame,
    load_icu_tables,
    log_selection_run,
    prepare_selection_inputs,
    print_ranking,
    save_ranking_csv,
    setup_tracking,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — What "Filter" Selection Means
# ════════════════════════════════════════════════════════════════════════
# Filter methods score each feature INDEPENDENTLY of the downstream
# model. Fast, parallelisable, blind to interactions.
#   - Mutual information (MI): any functional dependency
#   - Chi-squared: independence test; requires NON-NEGATIVE features


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: feature matrix + numeric inputs
# ════════════════════════════════════════════════════════════════════════

tables = load_icu_tables()
features = build_full_feature_frame(tables)
feature_cols, X_sel, y_binary = prepare_selection_inputs(features)

print("\n" + "=" * 70)
print("  Filter Selection — Mutual Information + Chi-Squared")
print("=" * 70)
print(f"  Features: {len(feature_cols)}")
print(f"  Samples:  {X_sel.shape[0]}")
print(f"  Positive class rate: {y_binary.mean():.3f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (score): run MI and chi-squared
# ════════════════════════════════════════════════════════════════════════

# TODO: Score every feature with mutual information.
# Hint: mutual_info_classif(X_sel, y_binary, random_state=42)
mi_scores = ____

mi_ranking = sorted(
    [
        (name, float(score) if np.isfinite(score) else 0.0)
        for name, score in zip(feature_cols, mi_scores)
    ],
    key=lambda x: x[1],
    reverse=True,
)

# Chi-squared requires non-negative features — scale to [0, 1] first.
# TODO: Apply MinMaxScaler to X_sel before chi2.
# Hint: MinMaxScaler().fit_transform(X_sel)
X_chi2 = ____

# TODO: Run chi2 on the scaled X and unpack (scores, pvalues).
# Hint: chi2(X_chi2, y_binary)
chi2_scores, chi2_pvalues = ____

chi2_ranking = sorted(
    [
        (name, float(score) if np.isfinite(score) else 0.0)
        for name, score in zip(feature_cols, chi2_scores)
    ],
    key=lambda x: x[1],
    reverse=True,
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(mi_ranking) == len(feature_cols), "Task 3: MI must score every feature"
assert len(chi2_ranking) == len(feature_cols), "Task 3: chi2 must score every feature"
assert mi_ranking[0][1] > 0, "Task 3: top MI feature should have a positive score"
print("\n[ok] Checkpoint 1 passed — filter scoring complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the rankings
# ════════════════════════════════════════════════════════════════════════

print_ranking("Mutual Information (top 15)", mi_ranking, top=15)
print_ranking("Chi-Squared (top 15)", chi2_ranking, top=15)

# TODO: Build top-20 sets for each method and compute the intersection.
# Hint: mi_top20 = {name for name, _ in mi_ranking[:20]}
mi_top20 = ____
chi2_top20 = ____
filter_consensus = sorted(mi_top20 & chi2_top20)

print("\n--- Filter Consensus (top-20 intersection) ---")
print(f"  MI top-20:           {len(mi_top20)} features")
print(f"  Chi-squared top-20:  {len(chi2_top20)} features")
print(f"  Intersection:        {len(filter_consensus)} features")
for f in filter_consensus:
    print(f"    - {f}")

save_ranking_csv(mi_ranking, "filter_mi_ranking.csv", score_col="mi_score")
save_ranking_csv(chi2_ranking, "filter_chi2_ranking.csv", score_col="chi2_score")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(filter_consensus) >= 3, "Task 4: expected at least 3 consensus features"
print("\n[ok] Checkpoint 2 passed — filter consensus found\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4b — LOG the filter run to ExperimentTracker
# ════════════════════════════════════════════════════════════════════════


async def log_filter() -> str:
    conn, tracker, exp_id = await setup_tracking()
    run_id = await log_selection_run(
        tracker,
        exp_id,
        run_name="filter_mi_chi2_top20",
        method="filter",
        selected_features=filter_consensus,
        total_features=len(feature_cols),
        extra_params={"top_k": "20", "scorers": "mutual_info_classif,chi2"},
        extra_metrics={"top1_mi": mi_ranking[0][1], "top1_chi2": chi2_ranking[0][1]},
    )
    await conn.close()
    return run_id


run_id = asyncio.run(log_filter())
print(f"\n  ExperimentTracker run: {run_id}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: SingHealth Radiology Triage
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: SingHealth's tele-radiology network processes 4,000
# chest X-rays/day. Filter scoring runs in milliseconds per feature —
# the ranking can refresh nightly, and clinicians can veto features
# they don't trust.
#
# BUSINESS IMPACT: ~S$246K/year in earlier interventions + S$1.2M/year
# in avoided missed findings. ~30x ROI in year one.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Scored every engineered feature with mutual information + chi2
  [x] Found the robust top-20 intersection as a model-free shortlist
  [x] Logged the run to ExperimentTracker

  Next: 03_wrapper_selection.py — Recursive Feature Elimination with
  a Random Forest to capture interactions.
"""
)
