# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1.4: Spectral Clustering via the Graph Laplacian
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build an RBF affinity graph and the graph Laplacian
#   - Embed points via the smallest k eigenvectors of L
#   - Cluster the spectral embedding with K-means
#   - Recognise non-convex cluster shapes K-means cannot separate
#
# PREREQUISITES: 01_kmeans.py.
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — affinity, Laplacian, eigenvectors, embedding
#   2. Build — subsample + fit spectral for a few K values
#   3. Train — pick the best K
#   4. Visualise — silhouette vs K
#   5. Apply — SMRT Singapore train-line community detection
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_1 import (
    RANDOM_STATE,
    load_customers,
    out_path,
    setup_engines,
    standardise,
    subsample,
    teardown_engines,
    track_run,
)

load_dotenv()

# ── Kailash-ML ExperimentTracker — every clustering run logs here ─────────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Graph Laplacian Embedding
# ════════════════════════════════════════════════════════════════════════
# A_ij = exp(-||x_i - x_j||² / 2σ²)    (RBF affinity)
# D_ii = Σ_j A_ij                      (degree matrix)
# L = D - A                            (graph Laplacian)
# The smallest k eigenvectors of L embed the graph in R^k; points that
# are connected through dense paths land near each other there. Run
# K-means on the embedding. Price: O(n^3). Small-to-medium data only.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: subsample and fit spectral for K in {3,4,5}
# ════════════════════════════════════════════════════════════════════════

customers, feature_cols = load_customers()
X_scaled, _ = standardise(customers, feature_cols)
n_samples = X_scaled.shape[0]

# TODO: Subsample 2500 rows using the shared subsample() helper.
X_spec, idx_spec = ____
n_spec = X_spec.shape[0]

print("=" * 70)
print("  Spectral Clustering on Singapore E-commerce Customers")
print("=" * 70)
print(f"  Subsample: {n_spec:,} of {n_samples:,}")

K_CANDIDATES = [3, 4, 5]
spectral_results: dict[int, dict] = {}

print(f"\n  {'K':>3} {'Silhouette':>12} {'Time':>8}")
print("  " + "─" * 28)
for k in K_CANDIDATES:
    t0 = time.perf_counter()
    # TODO: Build a SpectralClustering model with n_clusters=k,
    # affinity="rbf", gamma=1.0, random_state=RANDOM_STATE,
    # assign_labels="kmeans". Fit_predict on X_spec.
    spec = ____
    labels = ____
    elapsed = time.perf_counter() - t0
    sil = silhouette_score(X_spec, labels)
    spectral_results[k] = {"labels": labels, "sil": sil, "time": elapsed}
    print(f"  {k:>3} {sil:>12.4f} {elapsed:>7.2f}s")


# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert len(spectral_results) == len(K_CANDIDATES), "Task 2: spectral sweep incomplete"
print("\n  [ok] Checkpoint 1 passed — spectral embeddings fitted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Pick the best K and compare to K-means on the same X
# ════════════════════════════════════════════════════════════════════════

# TODO: Pick the K with the best silhouette from spectral_results.
best_k_spec, best_stats = ____
print(f"  Best spectral K: {best_k_spec}  (silhouette={best_stats['sil']:.4f})")

km_compare = KMeans(n_clusters=best_k_spec, random_state=RANDOM_STATE, n_init=10)
km_labels_sub = km_compare.fit_predict(X_spec)
km_sil = silhouette_score(X_spec, km_labels_sub)
print(f"    K-means silhouette (same subsample): {km_sil:.4f}")
print(f"    Δ (spectral − kmeans) = {best_stats['sil'] - km_sil:+.4f}")

spec_labels = best_stats["labels"]


# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert best_k_spec in K_CANDIDATES, "Task 3: best K selection invalid"
assert len(set(spec_labels.tolist())) == best_k_spec, "Task 3: label count mismatch"
print("\n  [ok] Checkpoint 2 passed — spectral best-K selected\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: silhouette vs K
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
fig = viz.training_history(
    {"Silhouette (spectral)": [spectral_results[k]["sil"] for k in K_CANDIDATES]},
    x_label="K",
)
fig.update_layout(title="Spectral Clustering: Silhouette vs K")
fig.write_html(str(out_path("04_spectral_silhouette.html")))
print(f"  Saved: {out_path('04_spectral_silhouette.html')}")


# ── Checkpoint 3 ──────────────────────────────────────────────────────────
assert out_path("04_spectral_silhouette.html").exists(), "Task 4: viz not saved"
print("\n  [ok] Checkpoint 3 passed — spectral visualisation rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: SMRT Singapore Train-Line Community Detection
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: SMRT's ~200 MRT/LRT stations form a natural graph with edge
# weights = inter-station rider flows. Spectral is the canonical
# community-detection method on graphs.
#
# BUSINESS IMPACT: ~S$29M / year (capacity matching + station-group
# advertising + incident rerouting) on ~S$850M rail revenue.

print("  APPLY — SMRT Train-Line Community Detection")
print("  ─────────────────────────────────────────────────────────────────")

# TODO: Compute sizes = np.bincount(spec_labels) and print each
# community's size + percentage of n_spec.
sizes = ____
for i, n in enumerate(sizes):
    print(f"    Community {i}: {n:>5,} customers ({n/n_spec:6.1%})")
print("    Estimated annual benefit: S$29M.")


# ── Checkpoint 4 ──────────────────────────────────────────────────────────
assert int(sizes.sum()) == n_spec, "Task 5: spectral partition size mismatch"
print("\n  [ok] Checkpoint 4 passed — spectral community partition valid\n")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# scalar_metrics merges a base headline with per-K dicts derived from
# spectral_results — the same |-merge pattern you used in 02_hierarchical.

# TODO: call track_run with run_name "spectral_rbf". Headline scalars are
# spectral_best_silhouette (best_stats["sil"]), kmeans_baseline_silhouette
# (km_sil), and spectral_minus_kmeans_delta. Then |-merge in two per-K
# dicts: {f"spectral_k{k}_silhouette": float(r["sil"]) for k, r in
# spectral_results.items()} and the per-K times.
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "k_candidates": ",".join(str(k) for k in K_CANDIDATES),
        "best_k": best_k_spec,
        "affinity": "rbf",
        "gamma": 1.0,
        "n_subsample": n_spec,
    },
    scalar_metrics={
        "spectral_best_silhouette": float(best_stats["sil"]),
        "kmeans_baseline_silhouette": float(km_sil),
        "spectral_minus_kmeans_delta": float(best_stats["sil"] - km_sil),
    }
    | ____
    | ____,
)
print(f"  [tracked] spectral sweep + K-means baseline logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — ClusteringEngine.fit(algorithm='spectral')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1's ClusteringEngine wraps spectral clustering with the
# same RBF-affinity + Laplacian-embedding + KMeans-on-embedding pipeline
# you just hand-built. The engine handles polars→numpy and returns
# ClusterResult — silhouette, CH, inertia, labels — in one call.

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

spec_df = pl.from_numpy(X_spec, schema=feature_cols)

# TODO: instantiate ClusteringEngine and call .fit on spec_df with
# algorithm='spectral' and n_clusters=best_k_spec.
clustering = ____
fit_result = ____
print(
    f"  ClusteringEngine.fit(spectral, K={best_k_spec}): "
    f"silhouette={(fit_result.silhouette_score or 0.0):.4f}  "
    f"n_clusters={fit_result.n_clusters}"
)
print()
print(
    "  Three lessons, one ClusteringEngine surface — kmeans (lesson 01),"
    " dbscan (lesson 03), spectral (here). Same fit() signature across all"
    " three; the ExperimentTracker leaderboard makes the comparison trivial.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] RBF affinity matrix and the graph Laplacian
  [x] Spectral embedding via the smallest k eigenvectors of L
  [x] K-means in the spectral embedding space
  [x] When spectral beats K-means (non-convex shapes, graph data)
  [x] Mapped to SMRT MRT community detection — ~S$29M / year

  Next: 05_evaluation_profiling.py — pick a winner.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
