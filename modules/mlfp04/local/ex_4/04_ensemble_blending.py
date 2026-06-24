# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 4.4: Ensemble Blending with kailash-ml EnsembleEngine
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Normalise anomaly scores across detectors with different scales
#   - Blend detectors with equal weights, AUC weights, and rank weights
#   - Use kailash-ml EnsembleEngine.blend() for a unified ensemble API
#   - Compare all methods on AUC-ROC, AUC-PR, precision-at-recall
#   - Monitor ensemble anomaly rate over time for drift detection
#
# PREREQUISITES: 4.1, 4.2, 4.3.
#
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — why blending beats any single detector
#   2. Build — re-fit the four detectors and normalise their scores
#   3. Train — build three blends + EnsembleEngine.blend()
#   4. Visualise — comparison chart + monitoring chart
#   5. Apply — MAS-aligned production monitoring for a SG neobank
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from kailash_ml import EnsembleEngine

from shared.mlfp04.ex_4 import (
    AnomalyScoreEstimator,
    _finite,
    load_dataset,
    normalise_scores,
    precision_at_recall,
    print_metrics,
    rank_normalise,
    score_metrics,
    setup_engines,
    teardown_engines,
    track_run,
    write_comparison_chart,
    write_monitoring_chart,
)

# ── Kailash-ML ExperimentTracker — anomaly zoo shared store ──────────────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Ensemble Always Wins
# ════════════════════════════════════════════════════════════════════════
# Different detectors have different blind spots. Averaging their scores
# means the errors intersect, not union — the blend's error set is
# strictly smaller than any single detector's error set.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: re-fit the four detectors
# ════════════════════════════════════════════════════════════════════════

X, y, _feature_cols, _frame = load_dataset()
n_samples, _n_features = X.shape

print("\n" + "=" * 70)
print("  Ensemble Blending — Z-score + IQR + IF + LOF")
print("=" * 70)
print(f"Rows: {n_samples:,} | Anomalies: {int(y.sum()):,} ({y.mean():.2%})")

z_scores = np.abs(X).max(axis=1)

Q1 = np.percentile(X, 25, axis=0)
Q3 = np.percentile(X, 75, axis=0)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
iqr_scores = ((X < lower) | (X > upper)).sum(axis=1).astype(np.float64)

iso_forest = IsolationForest(
    n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1
).fit(X)
iso_scores = -iso_forest.score_samples(X)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=False)
lof.fit_predict(X)
lof_scores = -lof.negative_outlier_factor_

print("\nPer-detector baseline:")
z_m = print_metrics("Z-score", y, z_scores)
iqr_m = print_metrics("IQR", y, iqr_scores)
iso_m = print_metrics("Isolation Forest", y, iso_scores)
lof_m = print_metrics("LOF", y, lof_scores)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: three manual blends + EnsembleEngine.blend()
# ════════════════════════════════════════════════════════════════════════

z_norm = normalise_scores(z_scores)
iqr_norm = normalise_scores(iqr_scores)
iso_norm = normalise_scores(iso_scores)
lof_norm = normalise_scores(lof_scores)

# TODO: Build the equal-weight blend (simple average of the four normalised scores)
equal_blend = ____

# AUC-weighted blend — each detector contributes proportional to its AUC-ROC
aucs = {
    "z": z_m["auc_roc"],
    "iqr": iqr_m["auc_roc"],
    "iso": iso_m["auc_roc"],
    "lof": lof_m["auc_roc"],
}
total_auc = sum(aucs.values())
weights = {k: v / total_auc for k, v in aucs.items()}

# TODO: Build the AUC-weighted blend using weights["z"], weights["iqr"], etc.
weighted_blend = ____

# TODO: Rank-based blend. Hint: use rank_normalise(z_scores) etc. then average.
z_rank = ____
iqr_rank = ____
iso_rank = ____
lof_rank = ____
rank_blend = (z_rank + iqr_rank + iso_rank + lof_rank) / 4.0

# kailash-ml EnsembleEngine.blend()
estimators = [
    AnomalyScoreEstimator(iso_scores),
    AnomalyScoreEstimator(lof_scores),
    AnomalyScoreEstimator(z_scores),
    AnomalyScoreEstimator(iqr_scores),
]
engine = EnsembleEngine()
try:
    # TODO: Call engine.blend(estimators=estimators, X=X, weights=[...])
    # Weights must match the estimator order above.
    blended_proba = ____
    engine_blend = blended_proba[:, 1]
except (TypeError, AttributeError):
    engine_blend = weighted_blend

print("\nEnsemble blends:")
equal_m = print_metrics("Equal-weight blend", y, equal_blend)
weighted_m = print_metrics("AUC-weighted blend", y, weighted_blend)
rank_m = print_metrics("Rank blend", y, rank_blend)
engine_m = print_metrics("EnsembleEngine blend", y, engine_blend)
print(
    f"  AUC weights: z={weights['z']:.3f} iqr={weights['iqr']:.3f} "
    f"iso={weights['iso']:.3f} lof={weights['lof']:.3f}"
)


# ── Checkpoint ──────────────────────────────────────────────────────────
best_single_auc = max(
    z_m["auc_roc"], iqr_m["auc_roc"], iso_m["auc_roc"], lof_m["auc_roc"]
)
best_ensemble_auc = max(
    equal_m["auc_roc"], weighted_m["auc_roc"], rank_m["auc_roc"], engine_m["auc_roc"]
)
assert weighted_m["auc_roc"] > 0.5, "AUC-weighted blend should beat random"
assert abs(sum(weights.values()) - 1.0) < 1e-6, "AUC weights should sum to 1"
assert (
    best_ensemble_auc >= best_single_auc - 0.05
), "Ensembles should not significantly underperform best single detector"
print("\n[ok] Checkpoint passed — all four blends computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: comparison chart + monitoring
# ════════════════════════════════════════════════════════════════════════

comparison = {
    "Z-score": {"AUC_ROC": z_m["auc_roc"], "Avg_Precision": z_m["avg_precision"]},
    "IQR": {"AUC_ROC": iqr_m["auc_roc"], "Avg_Precision": iqr_m["avg_precision"]},
    "Isolation Forest": {
        "AUC_ROC": iso_m["auc_roc"],
        "Avg_Precision": iso_m["avg_precision"],
    },
    "LOF": {"AUC_ROC": lof_m["auc_roc"], "Avg_Precision": lof_m["avg_precision"]},
    "Equal Blend": {
        "AUC_ROC": equal_m["auc_roc"],
        "Avg_Precision": equal_m["avg_precision"],
    },
    "AUC-Weighted": {
        "AUC_ROC": weighted_m["auc_roc"],
        "Avg_Precision": weighted_m["avg_precision"],
    },
    "Rank Blend": {
        "AUC_ROC": rank_m["auc_roc"],
        "Avg_Precision": rank_m["avg_precision"],
    },
    "EnsembleEngine": {
        "AUC_ROC": engine_m["auc_roc"],
        "Avg_Precision": engine_m["avg_precision"],
    },
}
comparison_path = write_comparison_chart(comparison, "ex4_anomaly_comparison.html")
print(f"Saved comparison chart: {comparison_path}")

print("\nPrecision at key recall levels (AUC-weighted blend):")
for target_recall in [0.50, 0.70, 0.80, 0.90]:
    p, t = precision_at_recall(y, weighted_blend, target_recall)
    print(f"  Recall={target_recall:.0%}  precision={p:.4f}  threshold={t:.4f}")

# Production monitoring demo
window_size = max(1, n_samples // 10)
_, decision_threshold = precision_at_recall(y, weighted_blend, 0.80)
anomaly_rates: list[float] = []
for i in range(10):
    start = i * window_size
    end = start + window_size
    window = weighted_blend[start:end]
    anomaly_rates.append(float((window > decision_threshold).mean()))
monitoring_path = write_monitoring_chart(anomaly_rates, "ex4_monitoring.html")
print(f"Saved monitoring chart: {monitoring_path}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS-Aligned Production Monitoring for a SG Neobank
# ════════════════════════════════════════════════════════════════════════
# A MAS-licensed Singapore neobank runs real-time txn monitoring against
# its TRM guidelines. Under the 250ms auth SLA, no single detector
# achieves both <0.2% FPR and >80% recall. A blended EnsembleEngine
# ships under SLA, is MAS-explainable, and delivers:
#
#   +S$1.6M/year recovered fraud
#   -39,000 false reviews/day (at S$30/hour reviewer cost)
#   = ~S$4.6M/year net impact
#
# The monitoring chart above is itself a drift signal: a sudden jump
# in the anomaly rate means the feature distribution moved, not that
# fraudsters got faster. Retrain or page the on-call ML engineer.


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's leaderboard to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Eight detectors (4 base + 4 blends) plus AUC weights and the monitoring
# windows. EnsembleEngine.blend() above IS the destination-first close;
# this section persists the leaderboard so runs 01-04 of m4_anomaly_zoo
# are comparable via tracker.list_runs().

precision_at_recall_metrics: dict[str, float] = {}
for tr in (0.50, 0.70, 0.80, 0.90):
    p, _t = precision_at_recall(y, weighted_blend, tr)
    label = str(tr).replace(".", "_")
    precision_at_recall_metrics[f"weighted_p_at_r_{label}"] = _finite(p)

monitoring_metrics: dict[str, float] = {
    f"monitor_window_{i:02d}_rate": _finite(rate)
    for i, rate in enumerate(anomaly_rates, start=1)
}

# TODO: call track_run with run_name="ensemble_blending". Headline scalars:
# every per-detector AUC-ROC + AP wrapped in _finite(), every blend variant
# AUC-ROC + AP, the four AUC weights, plus best_single_auc_roc /
# best_ensemble_auc_roc / ensemble_lift_over_best_single. |-merge with
# precision_at_recall_metrics and monitoring_metrics.
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "n_samples": n_samples,
        "n_estimators_iso": 200,
        "n_neighbors_lof": 20,
        "contamination": 0.01,
        "anomaly_rate": float(y.mean()),
        "decision_threshold": float(decision_threshold),
        "monitoring_windows": len(anomaly_rates),
    },
    scalar_metrics={
        "z_auc_roc": _finite(z_m["auc_roc"]),
        "z_avg_precision": _finite(z_m["avg_precision"]),
        "iqr_auc_roc": _finite(iqr_m["auc_roc"]),
        "iqr_avg_precision": _finite(iqr_m["avg_precision"]),
        "iso_auc_roc": _finite(iso_m["auc_roc"]),
        "iso_avg_precision": _finite(iso_m["avg_precision"]),
        "lof_auc_roc": _finite(lof_m["auc_roc"]),
        "lof_avg_precision": _finite(lof_m["avg_precision"]),
        "equal_blend_auc_roc": _finite(equal_m["auc_roc"]),
        "equal_blend_avg_precision": _finite(equal_m["avg_precision"]),
        "weighted_blend_auc_roc": _finite(weighted_m["auc_roc"]),
        "weighted_blend_avg_precision": _finite(weighted_m["avg_precision"]),
        "rank_blend_auc_roc": _finite(rank_m["auc_roc"]),
        "rank_blend_avg_precision": _finite(rank_m["avg_precision"]),
        "engine_blend_auc_roc": _finite(engine_m["auc_roc"]),
        "engine_blend_avg_precision": _finite(engine_m["avg_precision"]),
        "weight_z": _finite(weights["z"]),
        "weight_iqr": _finite(weights["iqr"]),
        "weight_iso": _finite(weights["iso"]),
        "weight_lof": _finite(weights["lof"]),
        "best_single_auc_roc": _finite(best_single_auc),
        "best_ensemble_auc_roc": ____,
        "ensemble_lift_over_best_single": _finite(best_ensemble_auc - best_single_auc),
    }
    | precision_at_recall_metrics
    | monitoring_metrics,
)
print(f"\n  [tracked] 8-detector leaderboard logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Score normalisation across detectors with different scales
  [x] Equal, AUC-weighted, and rank blends (manual)
  [x] kailash-ml EnsembleEngine.blend() with a custom adapter
  [x] 8-detector AUC-ROC / AUC-PR comparison
  [x] Precision-at-recall as the operator-facing metric
  [x] Production monitoring as a drift signal
  [x] MAS-aligned SG neobank deployment scenario

  Exercise 4 complete. Next: Exercise 5 — association rule mining.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
