# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 4.1: Statistical Outlier Detection (Z-score + IQR)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Apply the 3-sigma rule with Z-scores for outlier detection
#   - Apply the 1.5*IQR rule without assuming normality
#   - Winsorise extreme values to reduce skewness without losing rows
#   - Score and compare both methods with AUC-ROC and AUC-PR
#
# PREREQUISITES: MLFP02 (distributions, percentiles).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why statistical outliers matter for rare-event detection
#   2. Build — compute Z-scores and IQR bounds from standardised features
#   3. Train — score every row (unsupervised — no parameter fitting)
#   4. Visualise — distribution of flagged rows vs true anomalies
#   5. Apply — Singapore NETS chargeback review queue prioritisation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import skew

from shared.mlfp04.ex_4 import (
    _finite,
    load_dataset,
    print_metrics,
    score_metrics,
    setup_engines,
    teardown_engines,
    track_run,
)

# ── Kailash-ML ExperimentTracker — anomaly zoo shared store ──────────────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Statistical Outlier Rules Still Matter
# ════════════════════════════════════════════════════════════════════════
# Z-score and IQR are the cheapest anomaly detectors on the planet. They
# run in a single pass, need zero training, and produce a score that a
# non-technical analyst can explain ("this account's return rate is 4.2
# standard deviations above the average"). That explainability is worth
# more than +2% AUC-ROC in regulated industries — the model's answer is
# trivially defensible in a compliance audit.
#
# Z-score:  flag if |x - mean| / std  > 3
#           Assumes roughly-normal features. Fails on skewed or
#           multi-modal distributions (the mean and std get pulled by
#           the very tails you're trying to detect).
#
# IQR:      flag if x < Q1 - 1.5*IQR  or  x > Q3 + 1.5*IQR
#           Distribution-free. Works on skewed data because quartiles
#           are robust to the extreme tails.
#
# Winsorisation: instead of dropping outliers, CLIP them to the IQR
# bounds. Preserves sample size (no data loss) while pulling the mean
# and variance closer to the bulk of the distribution. This is the
# right move when you suspect a minority of rows is genuinely extreme
# but you still need to model the majority cleanly.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the Z-score and IQR detectors
# ════════════════════════════════════════════════════════════════════════

X, y, feature_cols, _frame = load_dataset()
n_samples, n_features = X.shape
print("\n" + "=" * 70)
print("  Statistical Outlier Detection — Z-score and IQR")
print("=" * 70)
print(
    f"Rows: {n_samples:,} | Features: {n_features} | "
    f"Anomalies: {int(y.sum()):,} ({y.mean():.2%})"
)


def zscore_anomaly_scores(X_scaled: np.ndarray) -> np.ndarray:
    """Return the per-row maximum absolute Z-score across features.

    X is already standardised (mean=0, std=1) by `build_features`, so
    |X| IS the Z-score. The max across features is the "worst" Z-score
    per row — the one that would trigger the 3-sigma rule first.
    """
    z = np.abs(X_scaled)
    return z.max(axis=1)


def iqr_outlier_counts(
    X_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (outlier count per row, lower bound, upper bound) using 1.5*IQR."""
    Q1 = np.percentile(X_scaled, 25, axis=0)
    Q3 = np.percentile(X_scaled, 75, axis=0)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    counts = ((X_scaled < lower) | (X_scaled > upper)).sum(axis=1).astype(np.float64)
    return counts, lower, upper


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (score every row — no fitting required)
# ════════════════════════════════════════════════════════════════════════

z_scores = zscore_anomaly_scores(X)
iqr_scores, lower_bound, upper_bound = iqr_outlier_counts(X)

# Sweep Z-score thresholds so the student sees the precision/coverage trade-off
print("\nZ-score threshold sweep (how many rows each threshold flags):")
for threshold in [2.0, 2.5, 3.0, 3.5]:
    flagged = z_scores > threshold
    n_flagged = int(flagged.sum())
    precision = float(y[flagged].mean()) if n_flagged else 0.0
    print(
        f"  |z| > {threshold}: flagged={n_flagged:>5,}  "
        f"({n_flagged / n_samples:.1%})  precision={precision:.3f}"
    )

# Headline metrics
print("\nPer-method scores:")
z_metrics = print_metrics("Z-score (max)", y, z_scores)
iqr_metrics = print_metrics("IQR (outlier count)", y, iqr_scores)

# Winsorisation — clip to IQR bounds and measure skewness reduction
X_winsorised = np.clip(X, lower_bound, upper_bound)
n_clipped = int((X != X_winsorised).sum())
skew_before = float(np.mean(np.abs(skew(X, axis=0))))
skew_after = float(np.mean(np.abs(skew(X_winsorised, axis=0))))
print(
    f"\nWinsorisation: clipped {n_clipped:,} values "
    f"({n_clipped / X.size:.2%} of the matrix)"
)
print(f"  Mean |skewness| before: {skew_before:.4f}")
print(f"  Mean |skewness| after:  {skew_after:.4f}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert z_metrics["auc_roc"] > 0.4, "Z-score AUC should beat random floor"
assert iqr_metrics["auc_roc"] > 0.4, "IQR AUC should beat random floor"
assert z_scores.min() >= 0, "Max |Z| scores must be non-negative"
assert skew_after <= skew_before + 1e-2, "Winsorisation should not increase skew"
print("\n[ok] Checkpoint passed — Z-score and IQR detectors scored\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── (A) Z-score distribution with threshold lines ──────────────────────
fig_z = go.Figure()
fig_z.add_trace(
    go.Histogram(
        x=z_scores[y == 0],
        name="Normal",
        opacity=0.7,
        nbinsx=60,
        marker_color="#636EFA",
    )
)
fig_z.add_trace(
    go.Histogram(
        x=z_scores[y == 1],
        name="Anomaly",
        opacity=0.7,
        nbinsx=60,
        marker_color="#EF553B",
    )
)
for thresh in [2.0, 3.0]:
    fig_z.add_vline(
        x=thresh,
        line_dash="dash",
        line_color="black",
        annotation_text=f"|z|={thresh}",
        annotation_position="top right",
    )
fig_z.update_layout(
    title="Z-Score Distribution: Normal vs Anomaly",
    xaxis_title="Max |Z-score| Across Features",
    yaxis_title="Count",
    barmode="overlay",
)
z_path = Path("outputs") / "ex4_anomaly" / "01_zscore_distribution.html"
z_path.parent.mkdir(parents=True, exist_ok=True)
fig_z.write_html(str(z_path))
print(f"[viz] Z-score distribution: {z_path}")

# ── (B) IQR box plots per feature (first 6 features) ──────────────────
n_show = min(6, X.shape[1])
fig_box = make_subplots(
    rows=1, cols=n_show, subplot_titles=[f"Feature {i}" for i in range(n_show)]
)
for i in range(n_show):
    fig_box.add_trace(
        go.Box(
            y=X[y == 0, i], name="Normal", marker_color="#636EFA", showlegend=(i == 0)
        ),
        row=1,
        col=i + 1,
    )
    fig_box.add_trace(
        go.Box(
            y=X[y == 1, i], name="Anomaly", marker_color="#EF553B", showlegend=(i == 0)
        ),
        row=1,
        col=i + 1,
    )
    fig_box.add_hline(
        y=float(upper_bound[i]), line_dash="dot", line_color="orange", row=1, col=i + 1
    )
    fig_box.add_hline(
        y=float(lower_bound[i]), line_dash="dot", line_color="orange", row=1, col=i + 1
    )
fig_box.update_layout(
    title="IQR Box Plots per Feature (orange = 1.5*IQR bounds)",
    height=400,
    width=250 * n_show,
)
box_path = Path("outputs") / "ex4_anomaly" / "01_iqr_boxplots.html"
fig_box.write_html(str(box_path))
print(f"[viz] IQR box plots: {box_path}")

print("\nInterpretation:")
print("  Z-score finds rows that are extreme on at least ONE feature.")
print("  IQR counts HOW MANY features are extreme — rewards multi-dim outliers.")
print("  For rare-event detection (<2% anomaly rate), AUC-PR is the honest")
print("  metric: AUC-ROC looks healthy even when precision is near zero.")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: NETS Chargeback Review Queue Prioritisation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: NETS (Network for Electronic Transfers, Singapore) processes
# ~12 million e-payments per day. Its fraud operations team manually
# reviews a queue of flagged transactions each morning; a typical reviewer
# can look at about 400 cases before fatigue and false-positive blindness
# set in.
#
# Why statistical outliers are the right tool FIRST:
#   - Explainable to compliance ("this merchant's chargeback rate is 4.3
#     standard deviations above normal") — no "the model said so"
#   - Zero training data required — the rule runs against live features
#   - Sub-millisecond scoring — fits inside the 120ms payment SLA
#
# BUSINESS IMPACT: If the Z-score + IQR pre-filter trims the daily queue
# from 5,000 flagged cases to the 800 with the highest blended outlier
# rank, reviewers see the top 400 in half the time. Industry benchmarks
# from MAS-licensed issuers put each caught chargeback at S$180 recovered
# and S$40 saved on dispute processing. Catching 30 extra chargebacks per
# day that the prior rule-only system missed = ~S$6,600/day in net
# recovery, or ~S$1.6M/year — for a detector that took three hours to
# build and zero dollars to run.
#
# LIMITATIONS: Statistical rules miss COORDINATED outliers (rings of
# accounts whose individual features look normal but whose joint pattern
# is suspicious). Exercise 4.2 (Isolation Forest) and 4.3 (LOF) catch
# those. Exercise 4.4 blends all four into the production score.

# Simple queue-prioritisation demo
reviewer_budget = 400
blended = (z_scores - z_scores.min()) + (iqr_scores - iqr_scores.min())
queue_order = np.argsort(-blended)[:reviewer_budget]
queue_precision = float(y[queue_order].mean())
queue_recall = float(y[queue_order].sum() / max(y.sum(), 1))
print(f"\nQueue-prioritisation demo " f"(reviewer budget = {reviewer_budget}):")
print(
    f"  Precision in top-{reviewer_budget}: {queue_precision:.3f}  "
    f"(fraction of reviewed cases that are true anomalies)"
)
print(
    f"  Recall in top-{reviewer_budget}:    {queue_recall:.3f}  "
    f"(fraction of ALL anomalies the reviewer sees)"
)


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Method names (zscore_max / iqr_count / queue) are simple snake_case so
# they match the tracker key regex [a-zA-Z_][a-zA-Z0-9_.\-]* directly —
# no _slug() needed. AUC-ROC/AP can be NaN if the sweep ever produces a
# single-class slice; _finite() guards every emit.

track_run(
    tracker,
    exp_name,
    run_name="statistical_zscore_iqr",
    params={
        "n_samples": n_samples,
        "n_features": n_features,
        "anomaly_rate": float(y.mean()),
        "reviewer_budget": reviewer_budget,
        "n_winsorised": n_clipped,
    },
    scalar_metrics={
        "zscore_auc_roc": _finite(z_metrics["auc_roc"]),
        "zscore_avg_precision": _finite(z_metrics["avg_precision"]),
        "iqr_auc_roc": _finite(iqr_metrics["auc_roc"]),
        "iqr_avg_precision": _finite(iqr_metrics["avg_precision"]),
        "queue_precision": _finite(queue_precision),
        "queue_recall": _finite(queue_recall),
        "skew_before": _finite(skew_before),
        "skew_after": _finite(skew_after),
    },
)
print(
    f"\n  [tracked] zscore + IQR + queue prioritisation logged to "
    f"{exp_name} run='statistical_zscore_iqr'\n"
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — AnomalyDetectionEngine.detect()
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1's AnomalyDetectionEngine.detect() does NOT support
# zscore/IQR (those are pure statistical primitives — too simple to
# merit an engine). It does support isolation_forest / lof / one_class_svm.
# This close shows where the engine path begins — Lesson 02 onwards uses
# the same engine surface for the harder algorithms this lesson's
# statistical rules cannot catch (coordinated outliers, density-based
# anomalies in feature subspaces).

import polars as pl

from kailash_ml.engines.anomaly_detection import AnomalyDetectionEngine

anomaly_df = pl.from_numpy(X, schema=feature_cols)
det = AnomalyDetectionEngine()
preview = det.detect(anomaly_df, algorithm="isolation_forest", contamination=0.01)
preview_metrics = score_metrics(y, np.asarray(preview.scores))
print(
    f"  AnomalyDetectionEngine.detect(isolation_forest): "
    f"AUC-ROC={preview_metrics['auc_roc']:.4f}  "
    f"AP={preview_metrics['avg_precision']:.4f}  "
    f"n_anomalies={preview.n_anomalies}"
)
print(
    f"  Hand-rolled Z-score AUC-ROC (Task 3): {z_metrics['auc_roc']:.4f}  "
    f"— statistical is the cheap-and-explainable primitive layer; the engine"
    " path begins next lesson with isolation_forest.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Z-score outlier detection (the 3-sigma rule) on standardised features
  [x] IQR outlier detection (the 1.5*IQR rule) without assuming normality
  [x] Winsorisation as a non-destructive alternative to dropping outliers
  [x] AUC-ROC vs AUC-PR on a <2% anomaly rate dataset
  [x] Framed a NETS Singapore payments scenario with concrete dollar impact

  KEY INSIGHT: Statistical rules are the CHEAPEST and MOST EXPLAINABLE
  anomaly detectors. Use them as your first filter, then layer
  Isolation Forest / LOF / ensembles on top for the hard cases.

  Next: 02_isolation_forest.py — path-length isolation finds anomalies
  that Z-score and IQR miss because it considers feature interactions.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
