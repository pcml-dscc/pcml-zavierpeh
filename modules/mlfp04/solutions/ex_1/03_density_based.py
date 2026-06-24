# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1.3: Density-Based Clustering (DBSCAN + HDBSCAN)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Apply DBSCAN with core/border/noise classification and select
#     epsilon via a k-distance plot
#   - Apply HDBSCAN to skip epsilon altogether by building a hierarchy of
#     DBSCAN clusterings and extracting the most stable partition
#   - Decide when "noise" is a feature, not a bug
#
# PREREQUISITES: 01_kmeans.py (feature setup).
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — density-based clustering and the epsilon/minPts contract
#   2. Build — k-distance plot and DBSCAN sweep across epsilon
#   3. Train — HDBSCAN with eom vs leaf cluster selection
#   4. Visualise — k-distance elbow plot
#   5. Apply — Singapore Grab ride-hail hotspot discovery, $ impact
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_1 import (
    setup_engines,
    teardown_engines,
    track_run,
    load_customers,
    out_path,
    standardise,
)

load_dotenv()

# ── Kailash-ML ExperimentTracker — every clustering run logs here ─────────
tracker, exp_name = setup_engines()

try:
    import hdbscan as hdbscan_lib
except ImportError:
    hdbscan_lib = None


# ════════════════════════════════════════════════════════════════════════
# THEORY — Density as the Cluster Definition
# ════════════════════════════════════════════════════════════════════════
# K-means and hierarchical cluster by MINIMISING a distance objective.
# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# redefines a cluster as "a region of high point density, separated from
# other regions by areas of low density". Two hyperparameters:
#
#   epsilon  - neighbourhood radius in input space
#   minPts   - minimum points inside that radius to be "core"
#
# Every point is classified as:
#   Core   - has ≥ minPts neighbours within epsilon
#   Border - not core, but within epsilon of a core point
#   Noise  - neither (label = -1)
#
# Two consequences:
#   1. Clusters can be any SHAPE (moons, rings, arbitrary blobs)
#   2. Points in sparse regions get label -1 ("noise") instead of being
#      forced into a cluster they do not belong to
#
# HDBSCAN fixes DBSCAN's biggest weakness: epsilon is hard to choose and
# changes with the data. HDBSCAN runs DBSCAN for EVERY epsilon at once,
# builds a hierarchy, and extracts the clusters that stay stable across
# the widest range of densities. You specify only min_cluster_size.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Load data, run k-distance plot, sweep DBSCAN epsilon
# ════════════════════════════════════════════════════════════════════════

customers, feature_cols = load_customers()
X_scaled, _ = standardise(customers, feature_cols)
n_samples = X_scaled.shape[0]

print("=" * 70)
print("  Density-Based Clustering on Singapore E-commerce Customers")
print("=" * 70)
print(f"  Samples: {n_samples:,}  features: {X_scaled.shape[1]}")

# k-distance plot: the canonical epsilon-selection tool for DBSCAN
K_NN = 10  # also used as minPts
nn = NearestNeighbors(n_neighbors=K_NN)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_dist = np.sort(distances[:, -1])

# Find the elbow via Kneedle: the point on the sorted k-distance curve that
# is FURTHEST from the chord between (0, k_dist[0]) and (n-1, k_dist[-1])
# AFTER both axes are normalised to [0, 1]. This locates the true point of
# maximum curvature — argmax of the 2nd derivative latches onto the steepest
# tail jump (a single outlier) and over-shoots.
_x = np.linspace(0.0, 1.0, k_dist.size)
_y = (k_dist - k_dist.min()) / (k_dist.max() - k_dist.min())
elbow_idx = int(np.argmax(np.abs(_y - _x)))
eps_suggested = float(k_dist[elbow_idx])

print(f"\n  k-distance elbow at eps ≈ {eps_suggested:.4f}")
print(f"  Range: [{k_dist[0]:.4f}, {k_dist[-1]:.4f}]")

# DBSCAN sweep around the suggested epsilon
print(f"\n  {'eps':>8} {'Clusters':>10} {'Noise %':>10} {'Silhouette':>12}")
print("  " + "─" * 44)
dbscan_results: dict[float, dict] = {}
for eps in (eps_suggested * 0.7, eps_suggested, eps_suggested * 1.3):
    labels = DBSCAN(eps=eps, min_samples=K_NN, n_jobs=-1).fit_predict(X_scaled)
    k = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    noise_pct = float((labels == -1).mean())
    valid = labels != -1
    sil = (
        silhouette_score(X_scaled[valid], labels[valid])
        if valid.sum() >= 2 and k >= 2
        else float("nan")
    )
    dbscan_results[eps] = {"labels": labels, "k": k, "noise_pct": noise_pct, "sil": sil}
    print(f"  {eps:>8.4f} {k:>10} {noise_pct:>9.1%} {sil:>12.4f}")

db_labels = dbscan_results[eps_suggested]["labels"]
n_noise = int((db_labels == -1).sum())
print(f"\n  DBSCAN (eps={eps_suggested:.4f}, minPts={K_NN}):")
print(f"    Non-noise points: {n_samples - n_noise:,}")
print(f"    Noise points:     {n_noise:,} ({n_noise/n_samples:.1%})")
print(f"    Clusters found:   {dbscan_results[eps_suggested]['k']}")


# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert eps_suggested > 0, "Task 2: suggested epsilon should be positive"
assert any(
    r["k"] >= 2 for r in dbscan_results.values()
), "Task 2: DBSCAN should find at least 2 clusters for one eps"
print("\n  [ok] Checkpoint 1 passed — DBSCAN epsilon selection complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: HDBSCAN with eom vs leaf cluster selection
# ════════════════════════════════════════════════════════════════════════

if hdbscan_lib is None:
    raise ImportError(
        "hdbscan package is required for this exercise. "
        "Install with: uv add hdbscan  (or  pip install hdbscan)"
    )

hdb_eom = hdbscan_lib.HDBSCAN(
    min_cluster_size=50, min_samples=10, cluster_selection_method="eom"
)
hdb_eom_labels = hdb_eom.fit_predict(X_scaled)
n_eom = len(set(hdb_eom_labels.tolist())) - (1 if -1 in hdb_eom_labels else 0)

hdb_leaf = hdbscan_lib.HDBSCAN(
    min_cluster_size=50, min_samples=10, cluster_selection_method="leaf"
)
hdb_leaf_labels = hdb_leaf.fit_predict(X_scaled)
n_leaf = len(set(hdb_leaf_labels.tolist())) - (1 if -1 in hdb_leaf_labels else 0)

noise_eom = float((hdb_eom_labels == -1).mean())
valid_eom = hdb_eom_labels != -1
sil_eom = (
    silhouette_score(X_scaled[valid_eom], hdb_eom_labels[valid_eom])
    if valid_eom.sum() >= 2 and n_eom >= 2
    else float("nan")
)

print(f"  HDBSCAN cluster-selection comparison:")
print(
    f"    EOM (Excess of Mass) : {n_eom} clusters  noise={noise_eom:.1%}  sil={sil_eom:.4f}"
)
print(f"    Leaf                 : {n_leaf} clusters  (finest granularity)")
print("  EOM is the production default — it picks the most stable clusters")
print("  across all density levels. Leaf gives the finest partition and is")
print("  useful when you explicitly want small sub-clusters.")


# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert n_eom >= 1, "Task 3: HDBSCAN EOM should find at least 1 cluster"
assert 0 <= noise_eom <= 1, "Task 3: noise fraction must be in [0, 1]"
print("\n  [ok] Checkpoint 2 passed — HDBSCAN auto-discovers clusters\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: the k-distance elbow that drove epsilon selection
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
fig = viz.training_history(
    {"k-distance (sorted)": k_dist.tolist()},
    x_label="Point index (sorted)",
)
fig.update_layout(
    title=f"DBSCAN k-distance plot (k={K_NN})  elbow at eps≈{eps_suggested:.4f}"
)
fig.write_html(str(out_path("03_dbscan_k_distance.html")))
print(f"  Saved: {out_path('03_dbscan_k_distance.html')}")
print("  Read the plot: y-axis is the distance from each point to its k-th")
print("  nearest neighbour, sorted. The 'elbow' (max curvature) is the")
print("  epsilon where density transitions from 'cluster' to 'noise'.")


# ── Checkpoint 3 ──────────────────────────────────────────────────────────
assert out_path("03_dbscan_k_distance.html").exists(), "Task 4: visualisation not saved"
print("\n  [ok] Checkpoint 3 passed — k-distance visualisation rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Grab Singapore Ride-Hail Hotspot Discovery
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Grab Singapore dispatches 400,000+ rides/day. The ops team
# needs to position driver incentives near "hotspots" — dense pickup
# clusters that form and dissolve throughout the day (CBD 8am, Marina
# Bay Sands 11pm, Changi T1 arrivals hall 06:00). K-means would FORCE a
# partition of every pickup in the city, including sparse residential
# areas. Hierarchical would collapse under 400K points.
#
# Why HDBSCAN is the right tool here:
#   - Hotspots have arbitrary SHAPES (Orchard Rd is a long thin strip)
#   - True sparse regions should stay sparse ("noise" is correct for a
#     single lonely pickup in Punggol at 3am — do not build a cluster)
#   - Density VARIES by neighbourhood (CBD is dense; Tuas is sparse) —
#     DBSCAN's single epsilon fails here; HDBSCAN's multi-density
#     hierarchy handles it natively
#   - min_cluster_size maps directly to a business rule: "ignore any
#     hotspot with fewer than N trips/hour"
#
# BUSINESS IMPACT: Grab SEA's 2024 annual report discloses ~US$2.4B in
# mobility gross transaction value for Singapore. Driver-incentive
# mis-targeting (paying surge for areas where demand already exists or
# missing genuine under-served areas) burns an estimated 3-5% of the
# incentive budget. Incentives are ~8% of mobility GTV (~US$192M/year).
# HDBSCAN-based hotspot discovery conservatively recovers ~20% of waste:
#     US$192M × 0.04 average waste × 0.20 recovery = US$1.54M / year
# (roughly S$2.05M/year) — and more importantly, driver satisfaction
# improves because surge lands where actual demand is, not where a
# K-means centroid happened to fall.

print("  APPLY — Grab SG Hotspot Discovery")
print("  ─────────────────────────────────────────────────────────────────")
for cid in sorted(set(int(c) for c in hdb_eom_labels.tolist() if c >= 0)):
    n = int((hdb_eom_labels == cid).sum())
    print(f"    Hotspot {cid}: {n:>5,} customers ({n/n_samples:6.1%})")
print(
    f"    Noise (truly sparse): {int((hdb_eom_labels == -1).sum()):,} customers "
    f"({float((hdb_eom_labels == -1).mean()):6.1%})"
)
print("    Estimated annual incentive waste recovery: S$2.05M")


# ── Checkpoint 4 ──────────────────────────────────────────────────────────
assert (
    int((hdb_eom_labels >= 0).sum() + (hdb_eom_labels == -1).sum()) == n_samples
), "Task 5: HDBSCAN labels must cover every sample"
print("\n  [ok] Checkpoint 4 passed — hotspot partition valid\n")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════

dbscan_best_eps, dbscan_best_stats = max(
    dbscan_results.items(), key=lambda x: x[1]["sil"]
)
track_run(
    tracker,
    exp_name,
    run_name="dbscan_hdbscan",
    params={
        "eps_suggested": eps_suggested,
        "min_pts": K_NN,
        "n_samples": n_samples,
        "hdb_min_cluster_size": 50,
        "hdb_min_samples": 10,
    },
    scalar_metrics={
        "dbscan_best_eps": float(dbscan_best_eps),
        "dbscan_best_silhouette": float(dbscan_best_stats["sil"]),
        "dbscan_best_n_clusters": float(dbscan_best_stats["k"]),
        "dbscan_best_noise_pct": float(dbscan_best_stats["noise_pct"]),
        "hdbscan_eom_n_clusters": float(n_eom),
        "hdbscan_leaf_n_clusters": float(n_leaf),
        "hdbscan_eom_noise_pct": float(noise_eom),
    },
)
print(
    f"  [tracked] DBSCAN sweep + HDBSCAN run logged to {exp_name} run='dbscan_hdbscan'\n"
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — ClusteringEngine.fit(algorithm='dbscan')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1's ClusteringEngine wraps DBSCAN. The engine handles the
# polars→numpy conversion, fits, computes silhouette over the non-noise
# subset, and returns ClusterResult with labels + metrics — the same flow
# this lesson hand-rolled across 60 lines of sklearn glue.
#
# HDBSCAN remains an exception: 1.5.1 does not have an HDBSCAN adapter
# yet. For HDBSCAN, the destination is the ExperimentTracker leaderboard —
# every persistence-vs-eom-vs-leaf comparison this lesson logged is now in
# m4_clustering_zoo.db, queryable side-by-side with kmeans (lesson 01) and
# the linkage zoo (lesson 02).

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

dbscan_df = pl.from_numpy(X_scaled, schema=feature_cols)
clustering = ClusteringEngine()
fit_result = clustering.fit(
    dbscan_df, algorithm="dbscan", eps=eps_suggested, min_samples=K_NN
)
print(
    f"  ClusteringEngine.fit(dbscan, eps={eps_suggested:.4f}): "
    f"n_clusters={fit_result.n_clusters}  "
    f"silhouette={(fit_result.silhouette_score or 0.0):.4f}"
)
print(
    "  ClusteringEngine 1.5.1: kmeans/dbscan/spectral/gmm. HDBSCAN — use the"
    " hdbscan library + tracker until the engine adapter lands.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] DBSCAN defines clusters by local density, not distance to a centroid
  [x] epsilon is chosen via the k-distance elbow, minPts via domain rule
  [x] Noise (label = -1) is a FEATURE: sparse points stay unassigned
  [x] HDBSCAN eliminates epsilon by running DBSCAN at every density and
      extracting the most persistent clusters
  [x] eom (Excess of Mass) vs leaf cluster selection: eom is default;
      leaf is for finest granularity
  [x] Mapped the method onto Grab SG hotspot discovery — S$2.05M/year
      recovered driver-incentive budget

  KEY INSIGHT: If your data has VARIABLE density (CBD vs suburbs) or
  arbitrary cluster SHAPES (strips, rings, moons), force-fitting K-means
  will give you nonsense. Density-based clustering is what you reach for.

  Next: 04_spectral.py — when you know the clusters are non-convex and
  you need the graph structure to find them.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
