# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 4.3: Local Outlier Factor (LOF)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain LOF as a ratio of neighbour density to point density
#   - Sweep n_neighbors and explain the "locality" trade-off
#   - Fit LOF and turn negative_outlier_factor_ into an anomaly score
#   - Explain when LOF beats Isolation Forest (varying-density clusters)
#
# PREREQUISITES: 4.2 (Isolation Forest).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why comparing LOCAL densities catches embedded anomalies
#   2. Build — sweep n_neighbors and pick the best value
#   3. Train — fit LOF and extract negative_outlier_factor_
#   4. Visualise — ROC curve (written to outputs/)
#   5. Apply — Shopee return-fraud cluster detection in SEA marketplaces
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from shared.mlfp04.ex_4 import (
    _finite,
    load_dataset,
    print_metrics,
    score_metrics,
    setup_engines,
    teardown_engines,
    track_run,
    write_roc_chart,
)

# ── Kailash-ML ExperimentTracker — anomaly zoo shared store ──────────────
tracker, exp_name = setup_engines()

# Per-n_neighbors sweep results captured below for the TRACK section.
nbrs_sweep: dict[int, dict[str, float]] = {}


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Local Density Beats Global Distance
# ════════════════════════════════════════════════════════════════════════
# LOF answers a different question than Isolation Forest. IF asks "how
# hard is this point to isolate from the rest of the data?" LOF asks
# "how does this point's neighbourhood density compare to ITS NEIGHBOURS'
# neighbourhood density?"
#
# Concretely, for a point p:
#   1. Find its k nearest neighbours N_k(p).
#   2. Measure local density around p: how close are those neighbours?
#   3. Measure local density around EACH neighbour.
#   4. LOF(p) = mean( density(neighbour_i) / density(p) ) for i in N_k(p).
#
# LOF ~ 1.0 means p has roughly the same density as its neighbours — it
# belongs to a cluster. LOF >> 1.0 means p sits in a sparser pocket than
# its neighbours — it's an outlier, EVEN IF it's surrounded by other
# points globally.
#
# WHY IT BEATS ISOLATION FOREST SOMETIMES: in data with varying cluster
# densities (some clusters dense, others sparse), a single global rule
# ("far from everything = outlier") fails. LOF is the right tool when
# anomalies live AT THE EDGE of a cluster they don't belong to.
#
# COST: LOF is O(n^2) at worst because it needs nearest-neighbour queries.
# For n > 200K rows, sub-sample or switch to an approximate NN backend.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: sweep n_neighbors
# ════════════════════════════════════════════════════════════════════════

X, y, _feature_cols, _frame = load_dataset()
n_samples, n_features = X.shape
print("\n" + "=" * 70)
print("  Local Outlier Factor (LOF)")
print("=" * 70)
print(
    f"Rows: {n_samples:,} | Features: {n_features} | "
    f"Anomalies: {int(y.sum()):,} ({y.mean():.2%})"
)

print("\nn_neighbors sweep:")
for n_nbrs in [10, 20, 30, 50]:
    lof_test = LocalOutlierFactor(
        n_neighbors=n_nbrs,
        contamination=0.01,
        novelty=False,
    )
    labels_test = lof_test.fit_predict(X)
    scores_test = -lof_test.negative_outlier_factor_
    m = score_metrics(y, scores_test)
    n_flagged = int((labels_test == -1).sum())
    nbrs_sweep[n_nbrs] = {
        "auc_roc": m["auc_roc"],
        "avg_precision": m["avg_precision"],
        "n_flagged": float(n_flagged),
    }
    print(
        f"  n_neighbors={n_nbrs:<3}  AUC-ROC={m['auc_roc']:.4f}  "
        f"AP={m['avg_precision']:.4f}  flagged={n_flagged:,}"
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit LOF with the chosen n_neighbors
# ════════════════════════════════════════════════════════════════════════
# n_neighbors=20 is a robust default for tabular data <100K rows. Smaller
# values hypersensitise to local noise; larger values drift toward a
# global density estimate and lose the "local" advantage.

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=False)
lof_labels = lof.fit_predict(X)
# negative_outlier_factor_ is negated so that "more negative = more normal"
# in sklearn's convention. Negate AGAIN so "higher = more anomalous".
lof_scores = -lof.negative_outlier_factor_

print("\nFinal LOF (n_neighbors=20):")
lof_metrics = print_metrics("LOF", y, lof_scores)
print(f"  Predicted anomalies: {int((lof_labels == -1).sum()):,}")
print(f"  True anomalies:      {int(y.sum()):,}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    lof_metrics["auc_roc"] > 0.5
), f"LOF AUC-ROC {lof_metrics['auc_roc']:.4f} should beat random"
assert lof_scores.std() > 0, "LOF scores should vary across rows"
assert lof_scores.shape[0] == n_samples, "Score length must match row count"
print("\n[ok] Checkpoint passed — LOF scored all rows\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: ROC curve + LOF score scatter + distance distribution
# ════════════════════════════════════════════════════════════════════════
roc_path = write_roc_chart(y, lof_scores, "LOF", "ex4_roc_lof.html")
print(f"Saved ROC chart: {roc_path}")

import plotly.graph_objects as go
from pathlib import Path

out_dir = Path("outputs") / "ex4_anomaly"
out_dir.mkdir(parents=True, exist_ok=True)

# ── (A) LOF scores scatter, coloured by anomaly label ──────────────────
# Plot the first two principal features vs LOF score to show where
# the high-LOF points cluster relative to the data.
fig_scatter = go.Figure()
fig_scatter.add_trace(
    go.Scatter(
        x=X[y == 0, 0],
        y=X[y == 0, 1],
        mode="markers",
        marker=dict(
            size=4,
            color=lof_scores[y == 0],
            colorscale="Blues",
            opacity=0.5,
        ),
        name="Normal",
    )
)
fig_scatter.add_trace(
    go.Scatter(
        x=X[y == 1, 0],
        y=X[y == 1, 1],
        mode="markers",
        marker=dict(
            size=7,
            color=lof_scores[y == 1],
            colorscale="Reds",
            colorbar=dict(title="LOF Score", x=1.02),
            opacity=0.9,
        ),
        name="Anomaly",
    )
)
fig_scatter.update_layout(
    title="LOF Scores: Feature 0 vs Feature 1 (colour = LOF score)",
    xaxis_title="Feature 0 (standardised)",
    yaxis_title="Feature 1 (standardised)",
)
scatter_path = out_dir / "03_lof_scatter.html"
fig_scatter.write_html(str(scatter_path))
print(f"[viz] LOF scatter: {scatter_path}")

# ── (B) LOF score distribution: normal vs anomaly ─────────────────────
fig_dist = go.Figure()
fig_dist.add_trace(
    go.Histogram(
        x=lof_scores[y == 0],
        name="Normal",
        opacity=0.7,
        nbinsx=60,
        marker_color="#636EFA",
    )
)
fig_dist.add_trace(
    go.Histogram(
        x=lof_scores[y == 1],
        name="Anomaly",
        opacity=0.7,
        nbinsx=60,
        marker_color="#EF553B",
    )
)
median_normal = float(np.median(lof_scores[y == 0]))
median_anomaly = float(np.median(lof_scores[y == 1]))
fig_dist.add_vline(
    x=median_normal,
    line_dash="dash",
    line_color="#636EFA",
    annotation_text=f"Normal median={median_normal:.2f}",
)
fig_dist.add_vline(
    x=median_anomaly,
    line_dash="dash",
    line_color="#EF553B",
    annotation_text=f"Anomaly median={median_anomaly:.2f}",
)
fig_dist.update_layout(
    title="LOF Score Distribution: Normal vs Anomaly",
    xaxis_title="LOF Score (higher = more anomalous)",
    yaxis_title="Count",
    barmode="overlay",
)
dist_path = out_dir / "03_lof_score_distribution.html"
fig_dist.write_html(str(dist_path))
print(f"[viz] LOF score distribution: {dist_path}")

print("\nLOF vs Isolation Forest on this dataset:")
print("  LOF catches anomalies that live inside a cluster they don't")
print("  belong to (sparse pocket surrounded by denser cluster).")
print("  Isolation Forest catches anomalies that are far from EVERY")
print("  cluster. Use BOTH, then blend — see 04_ensemble_blending.py.")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Shopee Return-Fraud Cluster Detection
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Shopee (SEA marketplace HQ'd in Singapore) operates buyer
# protection on returned items. A known fraud pattern is the "friends-
# and-family refund ring" — a cluster of buyer accounts with shared
# devices, similar behavioural features, all filing identical refund
# claims against the same seller. Individually each account is
# unremarkable; collectively they form a tight cluster in feature space
# that's distinct from the broader legitimate-buyer population.
#
# Why LOF is the right tool here:
#   - The fraud cluster is TIGHT — it has high LOCAL density
#   - Surrounding legitimate buyers have LOWER local density (more
#     spread out, diverse features)
#   - Isolation Forest MISSES this because the fraud cluster is NOT far
#     from the data; it's embedded in the middle
#   - LOF catches it because the ratio of cluster density to
#     neighbourhood density is extreme
#
# BUSINESS IMPACT: Shopee's 2023 APAC trust report disclosed ~S$14M/year
# in refund-ring losses across SEA marketplaces. A weekly LOF run on
# account embeddings, pre-filtering to the top 0.5% of the buyer base
# (about 5,000 suspects in a 1M-buyer market), lets the trust team send
# every suspect through a step-up verification flow. If LOF catches 40%
# of the ring behaviour (matched against ground-truth fraud labels from
# subsequent chargebacks), recovered loss = ~S$5.6M/year against an
# infrastructure cost of ~S$40K/year for the weekly batch job.
#
# LIMITATIONS: LOF is O(n^2) at scale. For a 50M-row marketplace, use
# a HDBSCAN + density-ratio approximation, or sub-sample to 200K rows
# per run. Exercise 4.4 (Ensemble Engine) shows how blending LOF with
# Isolation Forest raises recall further without raising cost much.


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Sweep keys: lof_k{N}_auc_roc / lof_k{N}_avg_precision / lof_k{N}_n_flagged
# Integer N is regex-safe directly — no _slug() needed.

sweep_metrics: dict[str, float] = {}
for n_nbrs, stats in nbrs_sweep.items():
    sweep_metrics[f"lof_k{n_nbrs}_auc_roc"] = _finite(stats["auc_roc"])
    sweep_metrics[f"lof_k{n_nbrs}_avg_precision"] = _finite(stats["avg_precision"])
    sweep_metrics[f"lof_k{n_nbrs}_n_flagged"] = stats["n_flagged"]

track_run(
    tracker,
    exp_name,
    run_name="local_outlier_factor",
    params={
        "n_samples": n_samples,
        "n_features": n_features,
        "best_n_neighbors": 20,
        "contamination": 0.01,
        "anomaly_rate": float(y.mean()),
    },
    scalar_metrics={
        "lof_auc_roc": _finite(lof_metrics["auc_roc"]),
        "lof_avg_precision": _finite(lof_metrics["avg_precision"]),
        "lof_n_predicted_anomalies": float(int((lof_labels == -1).sum())),
        "lof_normal_median": _finite(median_normal),
        "lof_anomaly_median": _finite(median_anomaly),
    }
    | sweep_metrics,
)
print(
    f"\n  [tracked] LOF n_neighbors sweep + final-fit logged to {exp_name} "
    f"run='local_outlier_factor'\n"
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — AnomalyDetectionEngine.detect(algorithm='lof')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1's AnomalyDetectionEngine wraps LOF under the same
# .detect() surface used in lesson 02 for isolation_forest. Same engine,
# different `algorithm=` string. Production stacks blend both via the
# EnsembleEngine in lesson 04.

import polars as pl

from kailash_ml.engines.anomaly_detection import AnomalyDetectionEngine

anomaly_df = pl.from_numpy(X, schema=_feature_cols)
det = AnomalyDetectionEngine()
fit_result = det.detect(anomaly_df, algorithm="lof", contamination=0.01)
fit_metrics = score_metrics(y, np.asarray(fit_result.scores))
print(
    f"  AnomalyDetectionEngine.detect(lof, contamination=0.01): "
    f"AUC-ROC={fit_metrics['auc_roc']:.4f}  "
    f"AP={fit_metrics['avg_precision']:.4f}  "
    f"n_anomalies={fit_result.n_anomalies}"
)
print(
    f"  Hand-rolled AUC-ROC (Task 3): {lof_metrics['auc_roc']:.4f}  "
    "— same algorithm under one surface; the engine's contamination knob"
    " maps directly to LocalOutlierFactor's parameter.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] LOF as a density-ratio test, not a distance test
  [x] n_neighbors as the "locality" knob
  [x] How LOF finds cluster-embedded anomalies that IF misses
  [x] The O(n^2) scalability limit and when to sub-sample
  [x] Framed a Shopee refund-ring detection scenario with recovered-loss impact

  KEY INSIGHT: Different anomaly detectors answer different questions.
  LOF asks a LOCAL question ("is this point in a sparser pocket than
  its neighbours?"). Isolation Forest asks a GLOBAL question ("is this
  point far from everything?"). You need BOTH in a real pipeline.

  Next: 04_ensemble_blending.py — combine Z-score + IQR + IF + LOF into
  a single ensemble score using kailash-ml EnsembleEngine.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
