# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1.3: Density-Based Clustering (DBSCAN + HDBSCAN)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Apply DBSCAN with core/border/noise and select epsilon via a
#     k-distance plot
#   - Apply HDBSCAN to skip epsilon via the density hierarchy
#   - Decide when "noise" is a feature, not a bug
#
# PREREQUISITES: 01_kmeans.py.
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — density-based clustering
#   2. Build — k-distance plot + DBSCAN epsilon sweep
#   3. Train — HDBSCAN with eom vs leaf
#   4. Visualise — k-distance elbow plot
#   5. Apply — Grab SG hotspot discovery
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

try:
    import hdbscan as hdbscan_lib
except ImportError:
    hdbscan_lib = None


# ════════════════════════════════════════════════════════════════════════
# THEORY — Density as the Cluster Definition
# ════════════════════════════════════════════════════════════════════════
# DBSCAN: a cluster is a dense region separated from other regions by
# sparse gaps. Hyperparameters: epsilon (radius), minPts (density).
# Point types: core / border / noise (label = -1).
# HDBSCAN: runs DBSCAN at every density and picks the most persistent
# clusters. No epsilon required.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: k-distance plot + DBSCAN epsilon sweep
# ════════════════════════════════════════════════════════════════════════

customers, feature_cols = load_customers()
X_scaled, _ = standardise(customers, feature_cols)
n_samples = X_scaled.shape[0]

print("=" * 70)
print("  Density-Based Clustering on Singapore E-commerce Customers")
print("=" * 70)
print(f"  Samples: {n_samples:,}  features: {X_scaled.shape[1]}")

K_NN = 10

# TODO: Fit a NearestNeighbors(n_neighbors=K_NN) on X_scaled and get the
# distances array. Pull distances[:, -1] (distance to the k-th neighbour)
# and np.sort() it ascending — this is the k-distance curve.
nn = ____
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_dist = ____

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

print(f"\n  {'eps':>8} {'Clusters':>10} {'Noise %':>10} {'Silhouette':>12}")
print("  " + "─" * 44)
dbscan_results: dict[float, dict] = {}
for eps in (eps_suggested * 0.7, eps_suggested, eps_suggested * 1.3):
    # TODO: Fit DBSCAN(eps=eps, min_samples=K_NN, n_jobs=-1) on X_scaled
    # and call .fit_predict(X_scaled) to get labels.
    labels = ____

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


# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert eps_suggested > 0, "Task 2: suggested epsilon should be positive"
assert any(r["k"] >= 2 for r in dbscan_results.values()), "Task 2: no clusters found"
print("\n  [ok] Checkpoint 1 passed — DBSCAN epsilon selection complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: HDBSCAN with eom vs leaf cluster selection
# ════════════════════════════════════════════════════════════════════════

if hdbscan_lib is None:
    raise ImportError("hdbscan required: uv add hdbscan")

# TODO: Build TWO HDBSCAN models with min_cluster_size=50, min_samples=10.
# The first uses cluster_selection_method="eom", the second "leaf".
# Fit_predict both on X_scaled.
hdb_eom = ____
hdb_eom_labels = ____

hdb_leaf = ____
hdb_leaf_labels = ____

n_eom = len(set(hdb_eom_labels.tolist())) - (1 if -1 in hdb_eom_labels else 0)
n_leaf = len(set(hdb_leaf_labels.tolist())) - (1 if -1 in hdb_leaf_labels else 0)
noise_eom = float((hdb_eom_labels == -1).mean())

valid_eom = hdb_eom_labels != -1
sil_eom = (
    silhouette_score(X_scaled[valid_eom], hdb_eom_labels[valid_eom])
    if valid_eom.sum() >= 2 and n_eom >= 2
    else float("nan")
)

print(f"  HDBSCAN cluster-selection comparison:")
print(f"    EOM : {n_eom} clusters  noise={noise_eom:.1%}  sil={sil_eom:.4f}")
print(f"    Leaf: {n_leaf} clusters  (finest granularity)")


# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert n_eom >= 1, "Task 3: HDBSCAN should find at least 1 cluster"
assert 0 <= noise_eom <= 1, "Task 3: noise fraction must be in [0, 1]"
print("\n  [ok] Checkpoint 2 passed — HDBSCAN auto-discovers clusters\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: k-distance elbow plot
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
fig = viz.training_history(
    {"k-distance (sorted)": k_dist.tolist()},
    x_label="Point index (sorted)",
)
fig.update_layout(title=f"DBSCAN k-distance plot  elbow at eps≈{eps_suggested:.4f}")
fig.write_html(str(out_path("03_dbscan_k_distance.html")))
print(f"  Saved: {out_path('03_dbscan_k_distance.html')}")


# ── Checkpoint 3 ──────────────────────────────────────────────────────────
assert out_path("03_dbscan_k_distance.html").exists(), "Task 4: viz not saved"
print("\n  [ok] Checkpoint 3 passed — k-distance visualisation rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Grab Singapore Ride-Hail Hotspot Discovery
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Grab SG dispatches 400K+ rides/day. HDBSCAN finds hotspots
# of any shape, keeps genuinely sparse regions as noise, and handles
# variable density (CBD vs Tuas).
#
# BUSINESS IMPACT: Estimated S$2.05M / year driver-incentive waste
# recovery on ~US$192M incentive spend.

print("  APPLY — Grab SG Hotspot Discovery")
print("  ─────────────────────────────────────────────────────────────────")
for cid in sorted(set(int(c) for c in hdb_eom_labels.tolist() if c >= 0)):
    n = int((hdb_eom_labels == cid).sum())
    print(f"    Hotspot {cid}: {n:>5,} customers ({n/n_samples:6.1%})")
print(
    f"    Noise: {int((hdb_eom_labels == -1).sum()):,} customers "
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
# Best DBSCAN epsilon = the one with the highest silhouette in the sweep.

dbscan_best_eps, dbscan_best_stats = max(
    dbscan_results.items(), key=lambda x: x[1]["sil"]
)

# TODO: call track_run with run_name "dbscan_hdbscan". scalar_metrics MUST
# include dbscan_best_eps, dbscan_best_silhouette, dbscan_best_n_clusters,
# hdbscan_eom_n_clusters (use n_eom), hdbscan_leaf_n_clusters (n_leaf),
# hdbscan_eom_noise_pct (noise_eom).
track_run(
    tracker,
    exp_name,
    run_name=____,
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
        "hdbscan_eom_n_clusters": ____,
        "hdbscan_leaf_n_clusters": ____,
        "hdbscan_eom_noise_pct": ____,
    },
)
print(f"  [tracked] DBSCAN sweep + HDBSCAN run logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — ClusteringEngine.fit(algorithm='dbscan')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1's ClusteringEngine wraps DBSCAN. The engine handles the
# polars→numpy conversion, fits, computes silhouette over the non-noise
# subset, and returns ClusterResult — the same flow this lesson hand-rolled
# across 60 lines of sklearn glue.
#
# HDBSCAN remains an exception: 1.5.1 does not have an HDBSCAN adapter
# yet. For HDBSCAN, the destination is the ExperimentTracker leaderboard.

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

dbscan_df = pl.from_numpy(X_scaled, schema=feature_cols)

# TODO: instantiate ClusteringEngine and call .fit on dbscan_df with
# algorithm='dbscan', eps=eps_suggested, min_samples=K_NN. The returned
# ClusterResult exposes .n_clusters and .silhouette_score.
clustering = ____
fit_result = ____
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
  [x] DBSCAN defines clusters by local density
  [x] epsilon selected via k-distance elbow, minPts via domain rule
  [x] Noise (label = -1) is a feature: sparse points stay unassigned
  [x] HDBSCAN eliminates epsilon by searching all density levels
  [x] Mapped to Grab SG hotspot discovery — S$2.05M / year recovery

  Next: 04_spectral.py — non-convex clusters via the graph Laplacian.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
