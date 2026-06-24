# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1.1: K-means with k-means++ Initialisation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Apply K-means with k-means++ initialisation and understand why it
#     converges faster than random initialisation
#   - Use the elbow method and silhouette score to select K objectively
#   - Read per-sample silhouette to spot mis-assigned points
#   - Interpret inertia (within-cluster sum of squares) as a loss value
#
# PREREQUISITES: MLFP03 complete (supervised ML, feature scaling).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why K-means works and how k-means++ fixes its weakness
#   2. Build — the elbow + silhouette sweep across K
#   3. Train — fit K-means with k-means++ vs random and compare
#   4. Visualise — silhouette curves vs K + per-sample silhouette
#   5. Apply — Singapore Shopee loyalty segmentation, $ impact per tier
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_1 import (
    RANDOM_STATE,
    load_customers,
    out_path,
    setup_engines,
    standardise,
    teardown_engines,
    track_run,
)

load_dotenv()

# ── Kailash-ML ExperimentTracker — every clustering run logs here ─────────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why K-means Works and Why k-means++ Matters
# ════════════════════════════════════════════════════════════════════════
# K-means minimises the within-cluster sum of squares:
#     J = Σ_k Σ_{x in C_k} ||x - μ_k||²
# It alternates two steps: assign each point to the nearest centroid,
# then recompute each centroid as the mean of its assigned points.
# k-means++ seeds centroids far apart to avoid poor local minima.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Load data and sweep K with silhouette scoring
# ════════════════════════════════════════════════════════════════════════

customers, feature_cols = load_customers()
X_scaled, _ = standardise(customers, feature_cols)
n_samples, n_features = X_scaled.shape

print("=" * 70)
print("  K-means on Singapore E-commerce Customers")
print("=" * 70)
print(f"  Samples={n_samples:,}  features={n_features}")


def sweep_k(X: np.ndarray, k_values: range) -> dict[str, list[float]]:
    """Fit K-means for each K and return per-K inertia + validity metrics."""
    inertias, sils, chs, dbs = [], [], [], []
    print(f"\n  {'K':>3} {'Inertia':>12} {'Silhouette':>12} {'CH':>10} {'DB':>8}")
    print("  " + "─" * 50)
    for k in k_values:
        # TODO: Build a KMeans instance with n_clusters=k, init='k-means++',
        # n_init=10, random_state=RANDOM_STATE. Fit_predict on X.
        # Hint: km = KMeans(n_clusters=____, random_state=____, n_init=10, init="k-means++")
        km = ____
        labels = ____

        # TODO: Append km.inertia_ to inertias and the three sklearn metrics
        # (silhouette_score, calinski_harabasz_score, davies_bouldin_score)
        # computed on (X, labels).
        inertias.append(____)
        sils.append(____)
        chs.append(____)
        dbs.append(____)
        print(
            f"  {k:>3} {km.inertia_:>12.0f} {sils[-1]:>12.4f} "
            f"{chs[-1]:>10.0f} {dbs[-1]:>8.4f}"
        )
    return {"inertia": inertias, "silhouette": sils, "ch": chs, "db": dbs}


K_RANGE = range(2, 11)
sweep = sweep_k(X_scaled, K_RANGE)

# TODO: Pick best_k as the K that MAXIMISES silhouette. Use np.argmax.
best_k = ____
print(f"\n  Best K by silhouette: {best_k} (score={max(sweep['silhouette']):.4f})")


# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert 2 <= best_k <= 10, "Task 2: best_k must be in the tested range"
assert max(sweep["silhouette"]) > 0, "Task 2: best silhouette should be positive"
assert len(sweep["inertia"]) == len(list(K_RANGE)), "Task 2: sweep size mismatch"
print("\n  [ok] Checkpoint 1 passed — silhouette sweep complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: k-means++ vs random initialisation head-to-head
# ════════════════════════════════════════════════════════════════════════

# TODO: Build TWO KMeans models with the SAME n_clusters=best_k and
# n_init=10, but different init values: "k-means++" and "random".
km_plus = ____
km_random = ____

t0 = time.perf_counter()
km_plus.fit(X_scaled)
t_plus = time.perf_counter() - t0

t0 = time.perf_counter()
km_random.fit(X_scaled)
t_random = time.perf_counter() - t0

print(f"  k-means++ vs Random Initialisation (K={best_k}):")
print(
    f"    k-means++: inertia={km_plus.inertia_:12.0f}  iters={km_plus.n_iter_:>3}  time={t_plus:.3f}s"
)
print(
    f"    Random:    inertia={km_random.inertia_:12.0f}  iters={km_random.n_iter_:>3}  time={t_random:.3f}s"
)

km_labels = km_plus.predict(X_scaled)


# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert (
    km_plus.inertia_ <= km_random.inertia_ + 1
), "Task 3: k-means++ should achieve inertia at least as good as random"
assert len(set(km_labels.tolist())) == best_k, "Task 3: wrong cluster count"
print("\n  [ok] Checkpoint 2 passed — k-means++ confirmed superior\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Silhouette curves and per-sample silhouette
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
history = {
    "Silhouette": sweep["silhouette"],
    "Inertia (scaled)": [i / max(sweep["inertia"]) for i in sweep["inertia"]],
}
fig = viz.training_history(history, x_label="K")
fig.update_layout(title=f"K-means: Silhouette and Inertia vs K (best K={best_k})")
fig.write_html(str(out_path("01_kmeans_elbow.html")))
print(f"  Saved: {out_path('01_kmeans_elbow.html')}")

# TODO: Compute per-sample silhouette using sklearn.metrics.silhouette_samples
# on (X_scaled, km_labels). Then for each cluster id, print its size, mean
# silhouette, and the number of points with s(i) < 0 (mis-assigned).
sil_samples = ____

print(f"\n  Per-Sample Silhouette (K={best_k}):")
for cid in range(best_k):
    mask = km_labels == cid
    s = sil_samples[mask]
    n_neg = int((s < 0).sum())
    print(
        f"    Cluster {cid}: n={int(mask.sum()):>5}  mean_sil={s.mean():.4f}  "
        f"mis-assigned={n_neg} ({n_neg/len(s):.1%})"
    )


# ── Checkpoint 3 ──────────────────────────────────────────────────────────
assert sil_samples.shape[0] == n_samples, "Task 4: per-sample silhouette missing points"
print("\n  [ok] Checkpoint 3 passed — visualisation and per-sample audit done\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Shopee Singapore Loyalty Segmentation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Shopee SG replaces its hand-coded Bronze/Silver/Gold tiers
# with data-driven K-means segments. K-means is the right tool: small K,
# roughly spherical clusters, centroids ARE the segment profiles.
#
# BUSINESS IMPACT: On a 3M buyer base, a 20% lift on S$20/buyer
# incremental campaign revenue is ~S$12M / year. Training cost: 2 seconds.

print("  APPLY — Shopee SG Loyalty Segmentation")
print("  ─────────────────────────────────────────────────────────────────")

# TODO: Compute segment sizes via np.bincount(km_labels). Print each
# segment's size and its percentage of n_samples.
segment_sizes = ____
for i, n in enumerate(segment_sizes):
    pct = n / n_samples * 100
    print(f"    Segment {i}: {n:>5,} customers ({pct:5.1f}%)")
print("    Estimated annual lift: S$12M (3M buyers × S$20 × 20%).")


# ── Checkpoint 4 ──────────────────────────────────────────────────────────
assert segment_sizes.min() > 0, "Task 5: every segment must have at least one customer"
assert int(segment_sizes.sum()) == n_samples, "Task 5: counts must sum to n_samples"
print("\n  [ok] Checkpoint 4 passed — segment sizes valid\n")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Every M4 ex_1 lesson logs into the SAME experiment ('m4_clustering_zoo')
# so you can compare K-means against hierarchical / DBSCAN / spectral /
# GMM later from one SQLite store. Sweep series = per-K curves; scalar
# metrics = the final-fit numbers.

# TODO: call track_run with the tracker + exp_name from setup_engines().
# run_name should identify the technique ("kmeans_pp" matches the solution
# leaderboard). Fill in the scalar_metrics value for the best silhouette
# from sweep["silhouette"], and the series_metrics dict with sweep_silhouette
# and sweep_inertia (already collected in `sweep`).
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "init": "k-means++",
        "n_init": 10,
        "random_state": RANDOM_STATE,
        "best_k": best_k,
        "n_features": n_features,
        "n_samples": n_samples,
    },
    scalar_metrics={
        "best_silhouette": ____,
        "kmeans_pp_inertia": float(km_plus.inertia_),
        "kmeans_random_inertia": float(km_random.inertia_),
        "kmeans_pp_iters": float(km_plus.n_iter_),
        "kmeans_random_iters": float(km_random.n_iter_),
        "kmeans_pp_time_s": float(t_plus),
        "kmeans_random_time_s": float(t_random),
    },
    series_metrics={
        "sweep_silhouette": ____,
        "sweep_inertia": ____,
    },
)
print(f"  [tracked] sweep + final-fit metrics logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — the kailash-ml ClusteringEngine
# ════════════════════════════════════════════════════════════════════════
# You hand-rolled the K sweep, the silhouette/CH/DB metrics, the
# k-means++-vs-random comparison, and the per-sample silhouette audit —
# ~120 lines of structure to internalise the moving parts.
#
# kailash-ml ships a single engine that IS that pipeline. ClusteringEngine.
# `sweep_k` runs the whole K-vs-criterion sweep for any supported algorithm
# (kmeans, hierarchical, dbscan, spectral, gmm) and `fit` returns a
# ClusterResult with labels + silhouette + Calinski-Harabasz + inertia.

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

cluster_df = pl.from_numpy(X_scaled, schema=feature_cols)

# TODO: instantiate ClusteringEngine and call .sweep_k on cluster_df with
# range(2, 11), algorithm='kmeans', criterion='silhouette'. Print the
# returned sweep_result.optimal_k.
clustering = ____
sweep_result = ____
print(f"  ClusteringEngine.sweep_k(): optimal_k={sweep_result.optimal_k}")

# TODO: call clustering.fit on cluster_df with algorithm='kmeans' and
# n_clusters=best_k. The returned ClusterResult exposes .silhouette_score,
# .calinski_harabasz_score, and .inertia.
fit_result = ____
print(
    f"  ClusteringEngine.fit(K={best_k}): "
    f"silhouette={fit_result.silhouette_score:.4f}  "
    f"CH={fit_result.calinski_harabasz_score:.0f}  "
    f"inertia={fit_result.inertia:.0f}"
)
print()
print("  Every metric the lesson printed by hand — silhouette, CH, inertia,")
print("  cluster sizes — is a field on ClusterResult. ClusteringEngine IS")
print("  the destination this lesson walked you toward.\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] K-means minimises within-cluster sum of squares via alternating
      assignment/update steps — guaranteed to converge
  [x] k-means++ seeding beats random init on speed and final inertia
  [x] Silhouette score gives an objective criterion for choosing K
  [x] Per-sample silhouette exposes mis-assigned points for re-review
  [x] Mapped K={best_k} clusters onto a Shopee SG loyalty tier system

  KEY INSIGHT: K-means gives you the centroids for free — the centroids
  ARE the segment profiles, ready for the marketing team.

  Next: 02_hierarchical.py — when you need a dendrogram instead of a K.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
