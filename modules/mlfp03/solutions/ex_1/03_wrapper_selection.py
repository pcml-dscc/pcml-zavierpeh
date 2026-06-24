# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.3: Wrapper Feature Selection (Recursive Feature
#                         Elimination with Random Forest)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Run Recursive Feature Elimination (RFE) around a Random Forest
#   - Understand how wrapper methods capture feature INTERACTIONS
#   - Compare RFE's selection against the filter consensus
#   - Apply wrapper selection in a setting where interactions matter
#     (cardiology risk models)
#
# PREREQUISITES: 02_filter_selection.py (filter consensus built)
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — why wrappers see what filters miss
#   2. Build — assemble estimator + RFE
#   3. Train — fit RFE, get ranking + support mask
#   4. Visualise — ranked table, marker for selected features
#   5. Apply — National Heart Centre Singapore risk scoring (S$ impact)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score

from shared.mlfp03.ex_1 import (
    OUTPUT_DIR,
    build_full_feature_frame,
    load_icu_tables,
    log_selection_run,
    prepare_selection_inputs,
    setup_tracking,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Wrappers Capture Interactions
# ════════════════════════════════════════════════════════════════════════
# A wrapper method trains a MODEL, asks the model which features
# mattered, removes the weakest, and repeats. Because the model sees all
# features together, it can learn that feature A is only useful in the
# presence of feature B — an interaction a filter method ignores.
#
# Recursive Feature Elimination (RFE) is the canonical wrapper:
#     1. Fit a Random Forest on all features.
#     2. Rank features by the forest's feature_importances_.
#     3. Drop the lowest-ranked k features.
#     4. Refit, re-rank, repeat until we hit the target feature count.
#
# The Random Forest is a strong default inside RFE because it captures
# non-linear dependencies and interactions out of the box — linear
# wrappers (LogReg RFE) miss exactly the interactions we care about.
#
# COST TRADE-OFF:
#   + captures interactions
#   + works with any model that exposes feature importances
#   - much slower than filter methods (train N models, not one score)
#   - selection is specific to the estimator — an RFE-chosen set may
#     help a Random Forest but confuse a linear model


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: assemble the estimator and RFE object
# ════════════════════════════════════════════════════════════════════════

tables = load_icu_tables()
features = build_full_feature_frame(tables)
feature_cols, X_sel, y_binary = prepare_selection_inputs(features)

print("\n" + "=" * 70)
print("  Wrapper Selection — RFE with Random Forest")
print("=" * 70)
print(f"  Features: {len(feature_cols)}")
print(f"  Samples:  {X_sel.shape[0]}")

rf_estimator = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1,
)

N_FEATURES_TO_SELECT = 15
rfe = RFE(estimator=rf_estimator, n_features_to_select=N_FEATURES_TO_SELECT, step=5)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit the RFE loop
# ════════════════════════════════════════════════════════════════════════

rfe.fit(X_sel, y_binary)

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

# --- RFE elimination curve: accuracy vs number of features ---
n_features_range = [5, 8, 10, 12, 15, 18, 20, 25]
n_features_range = [n for n in n_features_range if n <= len(feature_cols)]
elim_scores = []
for n_feat in n_features_range:
    rfe_curve = RFE(
        estimator=RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
        ),
        n_features_to_select=n_feat,
        step=5,
    )
    rfe_curve.fit(X_sel, y_binary)
    X_reduced = X_sel[:, rfe_curve.support_]
    cv_acc = cross_val_score(
        RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
        ),
        X_reduced,
        y_binary,
        cv=3,
        scoring="accuracy",
    ).mean()
    elim_scores.append(cv_acc)
    print(f"  n_features={n_feat:<3}  CV accuracy={cv_acc:.4f}")

fig_rfe = go.Figure()
fig_rfe.add_trace(
    go.Scatter(
        x=n_features_range,
        y=elim_scores,
        mode="lines+markers",
        marker=dict(size=10, color="#2563eb"),
        line=dict(width=3),
        name="CV Accuracy",
    )
)
best_idx = int(np.argmax(elim_scores))
fig_rfe.add_annotation(
    x=n_features_range[best_idx],
    y=elim_scores[best_idx],
    text=f"Best: {n_features_range[best_idx]} features",
    showarrow=True,
    arrowhead=2,
)
fig_rfe.update_layout(
    title="RFE Elimination Curve — Accuracy vs Number of Features",
    xaxis_title="Number of Features Selected",
    yaxis_title="3-Fold CV Accuracy",
    height=450,
)
rfe_path = OUTPUT_DIR / "ex1_03_rfe_elimination_curve.html"
fig_rfe.write_html(str(rfe_path))
print(f"\n  Saved: {rfe_path}")


# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert rfe_ranking[0][1] == 1, "Task 4: top-ranked features should have rank=1"
print("\n[ok] Checkpoint 2 passed — RFE ranking is well-formed\n")

# INTERPRETATION: compare this list to filter consensus from 02. RFE will
# typically promote interaction-rich features (shock_index,
# treatment_burden_score) that the filter methods under-rank because
# each interaction factor alone is only weakly related to the target.


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
# TASK 5 — APPLY: National Heart Centre Singapore Risk Stratification
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: National Heart Centre Singapore (NHCS) wants a 30-day
# re-admission risk model for heart-failure patients. The training set
# has ~220 candidate features across demographics, lab panels, medication
# history, and procedure codes. Interactions are KNOWN to dominate:
#   - ejection_fraction * diuretic_dose (under-diuresed weak heart)
#   - creatinine * ACE_inhibitor (renal contraindication)
#   - BNP * beta_blocker_dose (titration window)
# Filter methods rank each factor individually and miss every one of
# these combinations.
#
# Why RFE + Random Forest is the right tool:
#   - Random Forest captures interactions natively through its splits
#   - RFE iteratively removes the weakest factor, giving the remaining
#     features a chance to re-combine in the retained subset
#   - The ranking is stable enough that NHCS cardiologists can audit
#     the top 15 against clinical guidelines
#
# BUSINESS IMPACT: NHCS estimates each prevented 30-day readmission
# saves S$12,500 in avoided ICU bed-days, plus ~S$3,800 in avoided
# follow-up imaging. The baseline readmission rate is 23%; a model
# that cuts readmissions by 4 percentage points on ~3,600
# heart-failure discharges per year saves:
#     3,600 x 0.04 x (S$12,500 + S$3,800) ~ S$2.35M/year
# RFE + RF training cost: one data scientist x two weeks = ~S$24K.
# First-year ROI: ~95x.
#
# LIMITATIONS:
#   - RFE is estimator-specific: the 15 features that help a Random
#     Forest may not transfer cleanly to a logistic regression
#   - Compute cost scales with dataset size; at 500K rows and 500
#     features, RFE becomes impractical and you drop to embedded
#     methods (04_embedded_selection.py)
#   - The Random Forest's feature_importances_ are biased toward
#     high-cardinality features; permutation importance is more robust
#     if cardinality varies wildly across candidates


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Configured a Random Forest estimator inside sklearn's RFE loop
  [x] Fit RFE and extracted the selected-feature mask + ranking
  [x] Understood how wrappers promote interaction-rich features
  [x] Logged the wrapper run to ExperimentTracker
  [x] Applied RFE to NHCS heart-failure readmission scoring

  KEY INSIGHT: Wrappers see interactions but pay a compute tax. Use them
  when you can afford the training time AND when domain knowledge tells
  you interactions matter.

  Next: 04_embedded_selection.py — Lasso regularisation, which bakes
  feature selection into the model-fitting step itself.
"""
)
