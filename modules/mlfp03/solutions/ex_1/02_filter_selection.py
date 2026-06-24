# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.2: Filter Feature Selection
#                         (Mutual Information + Chi-Squared)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Use mutual information to rank features by any (linear or non-linear)
#     dependency with the target
#   - Use chi-squared to rank features by statistical independence
#   - Intersect top-k rankings to find ROBUST features that survive both
#     methods
#   - Apply filter selection to high-dimensional clinical data in a
#     cost-sensitive healthcare setting
#
# PREREQUISITES: 01_feature_engineering.py (feature matrix built)
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — what "filter" means and when to use it
#   2. Build — prepare X, y_binary from the feature matrix
#   3. Train — score every feature with MI and chi-squared
#   4. Visualise — ranked bar chart + top-20 intersection
#   5. Apply — SingHealth radiology triage (S$ impact)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import plotly.graph_objects as go
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from shared.mlfp03.ex_1 import (
    OUTPUT_DIR,
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
# Filter methods score each feature INDEPENDENTLY of the model that will
# eventually consume them. They are cheap, parallelisable, and stable.
# They are also blind to interactions: a feature that is useless alone
# but informative in combination with another feature will score poorly.
#
# Two filter methods cover the common cases:
#   - Mutual information (MI): captures any functional dependency, linear
#     or non-linear. Continuous features encouraged.
#   - Chi-squared: tests independence between a feature and a categorical
#     target. Requires NON-NEGATIVE features (MinMax-scale first).
#
# WHEN TO USE: first-pass pruning before expensive wrapper methods, or
# whenever you need a stable, explainable ranking that a clinician can
# sign off on.
#
# WHEN TO AVOID: feature engineering where interactions dominate — e.g.,
# "fever AND tachycardia" together matter even though each alone is
# common. For interaction-heavy domains, jump to wrapper selection (RFE).


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

# --- Mutual Information ---
mi_scores = mutual_info_classif(X_sel, y_binary, random_state=42)
# kNN-based MI / chi2 can return NaN for features that are constant or tie
# heavily on duplicate rows; treat an undefined score as "no information" (0.0).
mi_ranking = sorted(
    [
        (name, float(score) if np.isfinite(score) else 0.0)
        for name, score in zip(feature_cols, mi_scores)
    ],
    key=lambda x: x[1],
    reverse=True,
)

# --- Chi-Squared (requires non-negative features) ---
X_chi2 = MinMaxScaler().fit_transform(X_sel)
chi2_scores, chi2_pvalues = chi2(X_chi2, y_binary)
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

# Intersection: features that pass BOTH filter methods are robust —
# their relevance is not an artefact of one particular score.
mi_top20 = {name for name, _ in mi_ranking[:20]}
chi2_top20 = {name for name, _ in chi2_ranking[:20]}
filter_consensus = sorted(mi_top20 & chi2_top20)

print("\n--- Filter Consensus (top-20 intersection) ---")
print(f"  MI top-20:           {len(mi_top20)} features")
print(f"  Chi-squared top-20:  {len(chi2_top20)} features")
print(f"  Intersection:        {len(filter_consensus)} features")
for f in filter_consensus:
    print(f"    - {f}")

# --- Mutual information + chi-squared bar charts (top 15) ---
top_n = 15
mi_names = [name for name, _ in mi_ranking[:top_n]]
mi_vals = [score for _, score in mi_ranking[:top_n]]
chi2_names = [name for name, _ in chi2_ranking[:top_n]]
chi2_vals = [score for _, score in chi2_ranking[:top_n]]

fig_mi = go.Figure(
    go.Bar(x=mi_vals[::-1], y=mi_names[::-1], orientation="h", marker_color="#2563eb")
)
fig_mi.update_layout(
    title="Mutual Information Scores — Top 15 Features",
    xaxis_title="MI Score",
    yaxis_title="Feature",
    height=500,
    margin=dict(l=200),
)
mi_path = OUTPUT_DIR / "ex1_02_mi_scores.html"
fig_mi.write_html(str(mi_path))

fig_chi2 = go.Figure(
    go.Bar(
        x=chi2_vals[::-1], y=chi2_names[::-1], orientation="h", marker_color="#dc2626"
    )
)
fig_chi2.update_layout(
    title="Chi-Squared Scores — Top 15 Features",
    xaxis_title="Chi2 Score",
    yaxis_title="Feature",
    height=500,
    margin=dict(l=200),
)
chi2_path = OUTPUT_DIR / "ex1_02_chi2_scores.html"
fig_chi2.write_html(str(chi2_path))

# --- Selected vs dropped features comparison ---
mi_top20 = {name for name, _ in mi_ranking[:20]}
chi2_top20 = {name for name, _ in chi2_ranking[:20]}
consensus_features = sorted(mi_top20 & chi2_top20)
only_mi = sorted(mi_top20 - chi2_top20)
only_chi2 = sorted(chi2_top20 - mi_top20)

fig_venn = go.Figure()
categories = (
    ["Both"] * len(consensus_features)
    + ["MI only"] * len(only_mi)
    + ["Chi2 only"] * len(only_chi2)
)
feat_names = consensus_features + only_mi + only_chi2
fig_venn.add_trace(
    go.Bar(
        y=feat_names,
        x=[1] * len(feat_names),
        orientation="h",
        marker_color=[
            "#10b981" if c == "Both" else "#2563eb" if c == "MI only" else "#dc2626"
            for c in categories
        ],
        text=categories,
        textposition="inside",
    )
)
fig_venn.update_layout(
    title="Filter Consensus: MI vs Chi-Squared Top-20 Overlap",
    xaxis=dict(showticklabels=False),
    height=max(400, 25 * len(feat_names)),
    margin=dict(l=200),
    showlegend=False,
)
venn_path = OUTPUT_DIR / "ex1_02_filter_consensus.html"
fig_venn.write_html(str(venn_path))
print(f"\n  Saved: {mi_path}")
print(f"  Saved: {chi2_path}")
print(f"  Saved: {venn_path}")

# Persist the rankings so downstream technique files can re-use them
save_ranking_csv(mi_ranking, "filter_mi_ranking.csv", score_col="mi_score")
save_ranking_csv(chi2_ranking, "filter_chi2_ranking.csv", score_col="chi2_score")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert (
    len(filter_consensus) >= 3
), f"Task 4: expected at least 3 consensus features, got {len(filter_consensus)}"
print("\n[ok] Checkpoint 2 passed — filter consensus found\n")

# INTERPRETATION: A feature that ranks high under BOTH mutual information
# and chi-squared is almost certainly informative — the two methods
# disagree on noise but agree on signal.


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
# SCENARIO: SingHealth runs a tele-radiology network across seven
# Singapore hospitals. ~4,000 chest X-rays/day land in a shared queue;
# radiologists want a priority score that surfaces the likely-abnormal
# studies to the top of the queue within two minutes of upload.
#
# The production feature matrix has ~180 candidate features — patient
# demographics, prior admissions, current vitals, medication flags,
# referring specialty. The data-science team cannot afford to run a
# wrapper method on every new patient cohort (too slow) and a CNN on
# the image alone is over-indexed on the pixel distribution.
#
# Why filter selection is the right tool:
#   - Filter scoring runs in milliseconds per feature, so the ranking
#     can be refreshed nightly without infrastructure investment
#   - The MI + chi-squared intersection is explainable — a clinician
#     reviewing the feature list can veto any feature they don't trust
#   - Filters are model-agnostic: the downstream priority model can be
#     swapped (LogReg today, GBM tomorrow) without re-running selection
#
# BUSINESS IMPACT: SingHealth estimates a one-minute reduction in
# radiologist time-to-read for urgent studies is worth S$45 per study
# (through earlier downstream interventions: thrombolytics, surgery
# prep, ICU bed allocation). At 4,000 studies/day and a conservative
# 15% urgent rate, a 90-second improvement yields:
#     600 urgent/day x 1.5 min x S$0.75/min x 365 days ~ S$246K/year
# Plus an estimated S$1.2M/year in avoided missed findings on
# high-priority cases. Filter-selection infra cost: one analyst-week
# to wire + nightly Airflow job. ~30x ROI in year one.
#
# LIMITATIONS:
#   - Filter methods miss INTERACTION effects (the whole point of
#     wrapper selection in 03_wrapper_selection.py)
#   - Chi-squared requires non-negative features — the MinMax step
#     silently flattens scale, which can hurt features with natural
#     zeros
#   - MI is sensitive to bandwidth choice on continuous features;
#     random_state=42 makes the ranking deterministic but not
#     necessarily "correct"


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Scored every engineered feature with mutual information
  [x] Scored every engineered feature with chi-squared (MinMax-scaled)
  [x] Found the ROBUST top-20 intersection as a model-free shortlist
  [x] Logged the filter run to ExperimentTracker for later comparison
  [x] Applied filter selection to SingHealth radiology triage at scale

  KEY INSIGHT: Filters are the cheapest, fastest, most explainable
  feature-selection tool. Reach for them first. Only graduate to
  wrapper or embedded methods when the filter shortlist is not enough
  or when interactions dominate.

  Next: 03_wrapper_selection.py — use Recursive Feature Elimination
  with a Random Forest to capture interactions that filters miss.
"""
)
