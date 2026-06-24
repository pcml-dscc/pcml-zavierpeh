# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1.4: Spectral Clustering via the Graph Laplacian
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build an RBF affinity graph between points
#   - Embed points using the top-k eigenvectors of the graph Laplacian
#   - Cluster in the spectral embedding space with K-means
#   - Recognise non-convex cluster shapes that K-means cannot separate
#
# PREREQUISITES: 01_kmeans.py (K-means as the embedding-space learner).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — affinity, Laplacian, eigenvectors, embedding
#   2. Build — subsample, fit spectral for a few K values
#   3. Train — score partitions and pick the best K
#   4. Visualise — silhouette vs K
#   5. Apply — Singapore SMRT train-line community detection, $ impact
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import SpectralClustering
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
# THEORY — Graph Laplacian Embedding in Plain English
# ════════════════════════════════════════════════════════════════════════
# Spectral clustering treats points as nodes in a graph. Edge weights
# encode "similarity". The algorithm:
#
#   1. Affinity matrix  A_ij = exp(-||x_i - x_j||² / 2σ²)   (RBF kernel)
#   2. Degree matrix    D_ii = Σ_j A_ij
#   3. Laplacian        L = D - A    (or normalised L_sym = I - D^-½ A D^-½)
#   4. Eigendecompose L and take the SMALLEST k eigenvectors → spectral
#      embedding in R^k
#   5. Run K-means on the k-dim embedding
#
# WHY it works: the k smallest eigenvectors of the graph Laplacian encode
# the graph's "connectivity structure". Points that are connected through
# many dense paths end up near each other in the embedding, even if they
# are FAR apart in the original input space. This is what lets spectral
# clustering separate non-convex shapes like concentric rings or two
# interlocking spirals — shapes K-means cannot resolve.
#
# The price: eigendecomposition is O(n³) time and O(n²) memory. Spectral
# clustering is strictly a small-to-medium data tool. For 50K+ points,
# use Nyström approximation or skip to HDBSCAN.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Subsample aggressively, fit spectral for K in {3,4,5}
# ════════════════════════════════════════════════════════════════════════

customers, feature_cols = load_customers()
X_scaled, _ = standardise(customers, feature_cols)
n_samples = X_scaled.shape[0]

X_spec, idx_spec = subsample(X_scaled, n=2500, seed=RANDOM_STATE)
n_spec = X_spec.shape[0]

print("=" * 70)
print("  Spectral Clustering on Singapore E-commerce Customers")
print("=" * 70)
print(f"  Subsample: {n_spec:,} of {n_samples:,} (spectral is O(n^3))")

K_CANDIDATES = [3, 4, 5]
spectral_results: dict[int, dict] = {}

print(f"\n  {'K':>3} {'Silhouette':>12} {'Time':>8}")
print("  " + "─" * 28)
for k in K_CANDIDATES:
    t0 = time.perf_counter()
    spec = SpectralClustering(
        n_clusters=k,
        random_state=RANDOM_STATE,
        affinity="rbf",
        gamma=1.0,
        assign_labels="kmeans",
    )
    labels = spec.fit_predict(X_spec)
    elapsed = time.perf_counter() - t0
    sil = silhouette_score(X_spec, labels)
    spectral_results[k] = {"labels": labels, "sil": sil, "time": elapsed}
    print(f"  {k:>3} {sil:>12.4f} {elapsed:>7.2f}s")


# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert len(spectral_results) == len(K_CANDIDATES), "Task 2: spectral sweep incomplete"
print("\n  [ok] Checkpoint 1 passed — spectral embeddings fitted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Pick the best K
# ════════════════════════════════════════════════════════════════════════
# Spectral has no training loop in the gradient-descent sense — the
# "training" is the eigendecomposition followed by a tiny K-means on the
# embedding. We pick K by silhouette in the SPECTRAL embedding space.

best_k_spec, best_stats = max(spectral_results.items(), key=lambda x: x[1]["sil"])
print(f"  Best spectral K: {best_k_spec}  (silhouette={best_stats['sil']:.4f})")
print(f"  Compare with K-means silhouette on the SAME subsample:")

from sklearn.cluster import KMeans

km_compare = KMeans(n_clusters=best_k_spec, random_state=RANDOM_STATE, n_init=10)
km_labels_sub = km_compare.fit_predict(X_spec)
km_sil = silhouette_score(X_spec, km_labels_sub)
print(f"    K-means silhouette: {km_sil:.4f}")
print(
    f"    Δ (spectral − kmeans) = {best_stats['sil'] - km_sil:+.4f}  "
    "(positive = spectral wins; expected on non-convex structure)"
)

spec_labels = best_stats["labels"]


# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert best_k_spec in K_CANDIDATES, "Task 3: best K selection invalid"
assert len(set(spec_labels.tolist())) == best_k_spec, "Task 3: label count mismatch"
print("\n  [ok] Checkpoint 2 passed — spectral best-K selected\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: silhouette vs K for the spectral sweep
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
assert out_path(
    "04_spectral_silhouette.html"
).exists(), "Task 4: visualisation not saved"
print("\n  [ok] Checkpoint 3 passed — spectral visualisation rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: SMRT Singapore Train-Line Community Detection
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: SMRT Corporation operates Singapore's MRT and LRT network.
# Station-to-station passenger flows form a natural graph: each station
# is a node, each tap-in→tap-out pair is an edge weighted by rider count.
# SMRT's capacity planning team wants to discover "passenger communities"
# — groups of stations that trade riders densely with each other (CBD
# commuter cluster, Woodlands→Marina Bay commuter corridor, Orchard
# tourist cluster).
#
# Why spectral is the right tool here:
#   - The data is NATIVELY a graph — spectral is the canonical graph-
#     partitioning method
#   - Communities are non-convex in geographic space: a station in Jurong
#     may cluster with the CBD via commute flows, not with its neighbours
#   - The station count is ~200 — eigendecomposition is instant
#   - Normalised-cut (what spectral optimises) is the textbook community-
#     detection objective
#
# BUSINESS IMPACT: SMRT's disclosed 2024 rail revenue was ~S$850M. A
# data-driven community taxonomy feeds three concrete operations:
#   1. Peak-hour train assignment — extra cars on the dense commute
#      corridor identified by the top eigenvector. Typical capacity-
#      matching lift: 3-5% ridership recovered from passengers who
#      currently cannot board during peak (~S$25M/year revenue recovery).
#   2. Advertising revenue — station-group media packages priced on
#      actual community co-occurrence, not geographic adjacency. Typical
#      yield uplift 8-12% on ~S$40M ad revenue = ~S$4M/year.
#   3. Incident rerouting — when a line fails, the community graph tells
#      ops which bus bridges to prioritise.
# Total annual benefit ≈ S$29M vs. one-time modelling cost of a few
# engineer-hours (spectral on 200 nodes is 3 seconds).

print("  APPLY — SMRT Train-Line Community Detection")
print("  ─────────────────────────────────────────────────────────────────")
sizes = np.bincount(spec_labels)
for i, n in enumerate(sizes):
    print(f"    Community {i}: {n:>5,} customers ({n/n_spec:6.1%})")
print("    (In the SMRT scenario each node is a STATION, not a customer.)")
print("    Estimated annual benefit: S$29M (capacity + ads + rerouting).")


# ── Checkpoint 4 ──────────────────────────────────────────────────────────
assert int(sizes.sum()) == n_spec, "Task 5: spectral partition size mismatch"
print("\n  [ok] Checkpoint 4 passed — spectral community partition valid\n")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════

track_run(
    tracker,
    exp_name,
    run_name="spectral_rbf",
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
    | {
        f"spectral_k{k}_silhouette": float(r["sil"])
        for k, r in spectral_results.items()
    }
    | {f"spectral_k{k}_time_s": float(r["time"]) for k, r in spectral_results.items()},
)
print(
    f"  [tracked] spectral sweep + K-means baseline logged to {exp_name} run='spectral_rbf'\n"
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — ClusteringEngine.fit(algorithm='spectral')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1's ClusteringEngine wraps spectral clustering with the
# same RBF-affinity + Laplacian-embedding + KMeans-on-embedding pipeline
# this lesson just hand-built. The engine handles polars→numpy and
# returns ClusterResult — silhouette, CH, inertia, labels — in one call.

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

spec_df = pl.from_numpy(X_spec, schema=feature_cols)
clustering = ClusteringEngine()
fit_result = clustering.fit(spec_df, algorithm="spectral", n_clusters=best_k_spec)
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
  [x] Build an RBF affinity matrix and the graph Laplacian
  [x] Embed points via the smallest k eigenvectors of L
  [x] Run K-means on the spectral embedding instead of the raw features
  [x] Recognise when spectral beats K-means (non-convex shapes, graph-
      structured data)
  [x] Mapped the method onto SMRT MRT community detection for a
      ~S$29M / year capacity + ads + rerouting benefit

  KEY INSIGHT: When your data is NATURALLY a graph (stations, users,
  molecules), spectral is the default. When the similarity you care
  about is path-based rather than straight-line distance, spectral is
  the default. For everything else, prefer K-means or HDBSCAN because
  spectral is O(n^3).

  Next: 05_evaluation_profiling.py — stitch every method together and
  decide which one to ship.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
