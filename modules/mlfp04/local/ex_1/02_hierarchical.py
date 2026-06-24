# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1.2: Hierarchical Clustering with Four Linkage Methods
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build an agglomerative hierarchy bottom-up via the linkage matrix
#   - Compare single, complete, average, and Ward's linkage behaviours
#   - Read a dendrogram and cut it at a chosen K
#   - Choose linkage based on cluster shape expectations
#
# PREREQUISITES: 01_kmeans.py.
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — agglomerative merging and what linkage means
#   2. Build — fit four linkage methods on a subsample
#   3. Train — score each linkage partition against the others
#   4. Visualise — the Ward dendrogram
#   5. Apply — Singapore NTUC FairPrice store-cluster taxonomy
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import numpy as np
from dotenv import load_dotenv
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

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
# THEORY — Agglomerative Merging and Linkage
# ════════════════════════════════════════════════════════════════════════
# Each linkage method defines "distance between clusters" differently:
#   single   = min pairwise distance → elongated chains
#   complete = max pairwise distance → compact spheres
#   average  = mean pairwise distance
#   ward     = increase in within-cluster variance → K-means-like
# Ward's is the production default.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Subsample and fit four linkage methods
# ════════════════════════════════════════════════════════════════════════

customers, feature_cols = load_customers()
X_scaled, _ = standardise(customers, feature_cols)
n_samples = X_scaled.shape[0]

# TODO: Subsample 2000 rows using the shared subsample() helper.
# Hint: X_hier, idx_hier = subsample(X_scaled, n=____, seed=RANDOM_STATE)
X_hier, idx_hier = ____
n_hier = X_hier.shape[0]

CUT_K = 5
LINKAGE_METHODS = ["single", "complete", "average", "ward"]

print("=" * 70)
print("  Hierarchical Clustering on Singapore E-commerce Customers")
print("=" * 70)
print(f"  Subsample: {n_hier:,} of {n_samples:,} customers  cut K={CUT_K}")
print(
    f"\n  {'Linkage':<10} {'K':>4} {'Silhouette':>12} {'CH':>10} {'DB':>8} {'Time':>8}"
)
print("  " + "─" * 55)


def fit_linkage(method: str) -> dict:
    """Fit one linkage method and score it at the CUT_K cut."""
    t0 = time.perf_counter()
    # TODO: Compute the scipy linkage matrix Z. Use metric="euclidean".
    # Hint: Z = linkage(X_hier, method=____, metric="euclidean")
    Z = ____
    elapsed = time.perf_counter() - t0

    # TODO: Cut the dendrogram to get CUT_K clusters using
    # fcluster(Z, t=CUT_K, criterion="maxclust"). Subtract 1 for 0-based labels.
    labels = ____

    k_actual = len(set(labels.tolist()))
    if k_actual >= 2:
        sil = silhouette_score(X_hier, labels)
        ch = calinski_harabasz_score(X_hier, labels)
        db = davies_bouldin_score(X_hier, labels)
    else:
        sil, ch, db = -1.0, 0.0, float("inf")
    return {
        "Z": Z,
        "labels": labels,
        "n_clusters": k_actual,
        "silhouette": sil,
        "ch": ch,
        "db": db,
        "time": elapsed,
    }


hier_results = {m: fit_linkage(m) for m in LINKAGE_METHODS}
for m, r in hier_results.items():
    print(
        f"  {m:<10} {r['n_clusters']:>4} {r['silhouette']:>12.4f} "
        f"{r['ch']:>10.0f} {r['db']:>8.4f} {r['time']:>7.2f}s"
    )


# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert len(hier_results) == 4, "Task 2: all four linkage methods required"
assert all("silhouette" in r for r in hier_results.values()), "Task 2: scoring gap"
print("\n  [ok] Checkpoint 1 passed — four linkage methods fitted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Which linkage wins?
# ════════════════════════════════════════════════════════════════════════

# TODO: Find the linkage method with the HIGHEST silhouette. Use
# max(..., key=lambda x: x[1]["silhouette"]) on hier_results.items().
best_method = ____
print(
    f"  Best linkage by silhouette: {best_method[0]} "
    f"(sil={best_method[1]['silhouette']:.4f})"
)

ward_sil = hier_results["ward"]["silhouette"]
single_sil = hier_results["single"]["silhouette"]
print(f"  Ward's silhouette: {ward_sil:.4f}")
print(
    f"  Single silhouette: {single_sil:.4f}  "
    f"({'chains' if single_sil < ward_sil - 0.05 else 'competitive'})"
)


# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert best_method[0] in LINKAGE_METHODS, "Task 3: best method invalid"
assert hier_results["ward"]["n_clusters"] == CUT_K, "Task 3: Ward cut mismatch"
print("\n  [ok] Checkpoint 2 passed — Ward's linkage partition extracted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Ward dendrogram
# ════════════════════════════════════════════════════════════════════════

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        hier_results["ward"]["Z"],
        truncate_mode="lastp",
        p=30,
        ax=ax,
        leaf_font_size=9,
        color_threshold=0.7 * hier_results["ward"]["Z"][-CUT_K, 2],
    )
    ax.set_title(f"Ward Dendrogram — cut at K={CUT_K}")
    ax.set_xlabel("Cluster size (leaves)")
    ax.set_ylabel("Merge distance")
    fig.tight_layout()
    fig.savefig(str(out_path("02_hier_ward_dendrogram.png")), dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path('02_hier_ward_dendrogram.png')}")
except ImportError as e:  # pragma: no cover
    raise ImportError("02_hierarchical.py requires matplotlib") from e

print("\n  Read the dendrogram: bar height = merge distance; long vertical")
print("  gaps above a cut mean that cut is a robust, 'natural' K.")


# ── Checkpoint 3 ──────────────────────────────────────────────────────────
assert out_path("02_hier_ward_dendrogram.png").exists(), "Task 4: dendrogram missing"
print("\n  [ok] Checkpoint 3 passed — dendrogram rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: NTUC FairPrice Store-Cluster Taxonomy
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: FairPrice's ~230 stores get a tree-structured taxonomy.
# Merchandising thinks in trees; give them a dendrogram. Different K
# values serve different decisions (4 for national, 12 for regional).
#
# BUSINESS IMPACT: ~S$128M / year trade-promotion spend. Data-driven
# clustering recovers ~10% waste = S$12.8M / year.

print("  APPLY — NTUC FairPrice Store Taxonomy")
print("  ─────────────────────────────────────────────────────────────────")
ward_labels = hier_results["ward"]["labels"]

# TODO: Compute sizes = np.bincount(ward_labels). Print each cluster's
# size and percentage of n_hier.
sizes = ____
for i, n in enumerate(sizes):
    print(f"    Ward cluster {i}: {n:>5,} customers ({n/n_hier:6.1%})")
print("    Estimated annual promo waste recovery: S$12.8M.")


# ── Checkpoint 4 ──────────────────────────────────────────────────────────
assert int(sizes.sum()) == n_hier, "Task 5: Ward partition size mismatch"
assert len(sizes) == CUT_K, "Task 5: Ward cut should yield CUT_K clusters"
print("\n  [ok] Checkpoint 4 passed — Ward taxonomy valid\n")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# The four linkage methods all log into the SAME m4_clustering_zoo
# experiment so you can compare against the kmeans run from lesson 01.
# Best linkage = the one with the highest silhouette in `hier_results`.

best_method = max(hier_results.items(), key=lambda x: x[1]["silhouette"])

# TODO: call track_run with run_name f"hierarchical_{best_method[0]}" and
# scalar metrics constructed by dict-merging four per-method dicts:
#   {f"{m}_silhouette": float(r["silhouette"]) for m, r in hier_results.items()}
#   | {f"{m}_calinski_harabasz": float(r["ch"]) for m, r in hier_results.items()}
#   | {f"{m}_davies_bouldin": float(r["db"]) for m, r in hier_results.items()}
#   | {f"{m}_time_s": float(r["time"]) for m, r in hier_results.items()}
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "linkage_methods": ",".join(LINKAGE_METHODS),
        "cut_k": CUT_K,
        "best_linkage": best_method[0],
        "n_subsample": n_hier,
        "n_features": X_hier.shape[1],
    },
    scalar_metrics=____,
)
print(f"  [tracked] linkage comparison logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — engine surface honesty for hierarchical
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1 ClusteringEngine ships kmeans/dbscan/spectral/gmm but
# NOT hierarchical/agglomerative — it's the one mainstream clustering
# family the engine doesn't yet wrap. The engine-first surface for THIS
# lesson is therefore the ExperimentTracker we just used: every linkage
# method, every score, every timing — already in m4_clustering_zoo.db
# next to the kmeans run from 01. The destination is the comparable
# leaderboard, not a one-line replacement for scipy's `linkage`.

from kailash_ml.engines.clustering import ClusteringEngine

print("  ClusteringEngine 1.5.1 algorithms:", ClusteringEngine.__doc__ or "")
print("    Supported: kmeans, dbscan, spectral, gmm")
print(
    "    Hierarchical / agglomerative: use scipy.cluster.hierarchy until"
    " a future kailash-ml release adds the adapter.\n"
)
print(
    "  Engine-first take-away: the tracker leaderboard IS the destination —"
    " open mlfp04_ex1_clustering.db to compare every linkage method against"
    " the kmeans run from lesson 01 side-by-side.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Agglomerative merging builds a dendrogram bottom-up
  [x] Four linkage methods produce different cluster shapes
  [x] Read a dendrogram: height = merge distance; cut = partition
  [x] Ward's is the production default for compact clusters
  [x] Mapped the tree onto an NTUC FairPrice store taxonomy — S$12.8M/yr

  KEY INSIGHT: When the business thinks in a TREE, give them a tree.

  Next: 03_density_based.py — clusters of arbitrary SHAPE.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
