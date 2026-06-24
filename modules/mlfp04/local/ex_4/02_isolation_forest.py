# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 4.2: Isolation Forest Anomaly Detection
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain path-length isolation as an anomaly score
#   - Fit an Isolation Forest with the right contamination setting
#   - Sweep contamination to visualise precision/recall trade-off
#   - Compare tree-based isolation against statistical baselines
#
# PREREQUISITES: 4.1 (Z-score + IQR baselines).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — "rare points get isolated faster"
#   2. Build — fit IsolationForest with a contamination sweep
#   3. Train — score every row with the best-performing fit
#   4. Visualise — ROC curve (written to outputs/)
#   5. Apply — GrabPay merchant risk scoring
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

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

# Per-contamination sweep results captured below for the TRACK section.
sweep_results: dict[float, dict[str, float]] = {}


# ════════════════════════════════════════════════════════════════════════
# THEORY — Path Length as an Anomaly Score
# ════════════════════════════════════════════════════════════════════════
# Isolation Forest builds random binary trees with random feature splits.
# Anomalies need fewer splits to isolate (shallow leaf = anomalous).
# Score: higher = more anomalous. Contamination sets the expected
# fraction of anomalies — it's a domain assumption, not a hyperparameter.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: contamination sweep
# ════════════════════════════════════════════════════════════════════════

X, y, _feature_cols, _frame = load_dataset()
n_samples, n_features = X.shape
print("\n" + "=" * 70)
print("  Isolation Forest Anomaly Detection")
print("=" * 70)
print(
    f"Rows: {n_samples:,} | Features: {n_features} | "
    f"Anomalies: {int(y.sum()):,} ({y.mean():.2%})"
)

print("\nContamination sweep:")
contamination_grid = [0.001, 0.005, 0.01, 0.02, 0.05]
for contam in contamination_grid:
    # TODO: Instantiate IsolationForest with n_estimators=200,
    # contamination=contam, random_state=42, n_jobs=-1
    model = ____

    # TODO: Call model.fit_predict(X) to get labels (-1 = anomaly, 1 = normal)
    preds = ____

    n_flagged = int((preds == -1).sum())
    flagged = preds == -1
    precision = float(y[flagged].mean()) if n_flagged else 0.0
    sweep_results[contam] = {
        "n_flagged": float(n_flagged),
        "precision": precision,
    }
    print(
        f"  contamination={contam:<6}  flagged={n_flagged:>5,}  "
        f"precision={precision:.3f}"
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit the best-performing contamination
# ════════════════════════════════════════════════════════════════════════

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1,
)
# TODO: Call iso_forest.fit(X)
____

# TODO: Extract anomaly scores. score_samples returns HIGHER for normal;
# negate so HIGHER means more anomalous.
iso_scores = ____
iso_labels = iso_forest.predict(X)

print("\nFinal Isolation Forest (contamination=0.01):")
iso_metrics = print_metrics("Isolation Forest", y, iso_scores)
print(f"  Predicted anomalies: {int((iso_labels == -1).sum()):,}")
print(f"  True anomalies:      {int(y.sum()):,}")


# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    iso_metrics["auc_roc"] > 0.5
), f"Isolation Forest AUC-ROC {iso_metrics['auc_roc']:.4f} should beat random"
assert iso_metrics["avg_precision"] > 0.0, "AP should be positive"
assert (iso_labels == -1).sum() > 0, "Should flag at least one anomaly"
print("\n[ok] Checkpoint passed — Isolation Forest scored all rows\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: ROC curve
# ════════════════════════════════════════════════════════════════════════
roc_path = write_roc_chart(
    y, iso_scores, "Isolation Forest", "ex4_roc_isolation_forest.html"
)
print(f"Saved ROC chart: {roc_path}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: GrabPay Merchant Payout Risk Scoring
# ════════════════════════════════════════════════════════════════════════
# GrabPay runs nightly payouts for merchants across SEA. Refund-ring
# fraud looks normal on single features but unusual in 40+ feature joint
# space. Isolation Forest catches it; statistical rules don't.
#
# Impact: ~S$1.2M/year recovered against <S$20K/year IT cost. 60x ROI.


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Sweep keys use underscores in place of dots ("0.001" → "0_001") to
# stay inside the tracker key regex [a-zA-Z_][a-zA-Z0-9_.\-]*.

sweep_metrics: dict[str, float] = {}
for contam, stats in sweep_results.items():
    label = str(contam).replace(".", "_")
    sweep_metrics[f"iso_contam_{label}_n_flagged"] = stats["n_flagged"]
    sweep_metrics[f"iso_contam_{label}_precision"] = _finite(stats["precision"])

# TODO: call track_run with run_name="isolation_forest". Headline scalars:
# iso_auc_roc + iso_avg_precision (from iso_metrics — wrap in _finite),
# iso_n_predicted_anomalies (count of iso_labels == -1). |-merge with
# sweep_metrics.
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "n_samples": n_samples,
        "n_features": n_features,
        "n_estimators": 200,
        "best_contamination": 0.01,
        "anomaly_rate": float(y.mean()),
    },
    scalar_metrics={
        "iso_auc_roc": _finite(iso_metrics["auc_roc"]),
        "iso_avg_precision": ____,
        "iso_n_predicted_anomalies": float(int((iso_labels == -1).sum())),
    }
    | sweep_metrics,
)
print(f"\n  [tracked] isolation_forest sweep + final-fit logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — AnomalyDetectionEngine.detect()
# ════════════════════════════════════════════════════════════════════════
# AnomalyDetectionEngine wraps the same Isolation Forest you just fit by
# hand under one .detect() surface. The tracker leaderboard is the
# natural next step — engine + leaderboard = production anomaly stack.

import polars as pl

from kailash_ml.engines.anomaly_detection import AnomalyDetectionEngine

anomaly_df = pl.from_numpy(X, schema=_feature_cols)

# TODO: Instantiate AnomalyDetectionEngine and call .detect on anomaly_df
# with algorithm='isolation_forest' and contamination=0.01.
det = ____
fit_result = ____
fit_metrics = score_metrics(y, np.asarray(fit_result.scores))
print(
    f"  AnomalyDetectionEngine.detect(isolation_forest, contamination=0.01): "
    f"AUC-ROC={fit_metrics['auc_roc']:.4f}  "
    f"AP={fit_metrics['avg_precision']:.4f}  "
    f"n_anomalies={fit_result.n_anomalies}"
)
print(
    f"  Hand-rolled AUC-ROC (Task 3): {iso_metrics['auc_roc']:.4f} "
    "— same algorithm, one-line vs ten-line interface.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Path-length isolation explained
  [x] Contamination sweep and precision trade-off
  [x] IsolationForest on 40+ feature tabular data
  [x] ROC chart via ModelVisualizer
  [x] GrabPay SEA merchant fraud scenario

  Next: 03_local_outlier_factor.py — LOF finds cluster-embedded anomalies
  that Isolation Forest misses.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
