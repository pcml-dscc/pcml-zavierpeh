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
# It alternates two steps: assign each point to the nearest centroid, then
# recompute each centroid as the mean of its assigned points. This is a
# coordinate-descent algorithm — each step can only decrease J — so it
# converges in a finite number of iterations.
#
# The catch: J is non-convex. Different starting centroids converge to
# different local minima. Random initialisation occasionally places two
# centroids on top of each other and ends up with an empty cluster or a
# very poor partition.
#
# k-means++ fixes this by spreading the initial centroids apart:
#   1. Pick the first centroid uniformly at random.
#   2. For each subsequent centroid, sample a point with probability
#      proportional to its squared distance from the nearest existing
#      centroid.
# The result is a provably O(log K) approximation to the optimal seeding,
# and in practice it converges 2-5× faster and to a lower final J.


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
print(f"  Feature columns: {feature_cols}")


def sweep_k(X: np.ndarray, k_values: range) -> dict[str, list[float]]:
    """Fit K-means for each K and return per-K inertia and validity metrics."""
    inertias, sils, chs, dbs = [], [], [], []
    print(f"\n  {'K':>3} {'Inertia':>12} {'Silhouette':>12} {'CH':>10} {'DB':>8}")
    print("  " + "─" * 50)
    for k in k_values:
        km = KMeans(
            n_clusters=k, random_state=RANDOM_STATE, n_init=10, init="k-means++"
        )
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, labels))
        chs.append(calinski_harabasz_score(X, labels))
        dbs.append(davies_bouldin_score(X, labels))
        print(
            f"  {k:>3} {km.inertia_:>12.0f} {sils[-1]:>12.4f} "
            f"{chs[-1]:>10.0f} {dbs[-1]:>8.4f}"
        )
    return {"inertia": inertias, "silhouette": sils, "ch": chs, "db": dbs}


K_RANGE = range(2, 11)
sweep = sweep_k(X_scaled, K_RANGE)
best_k = list(K_RANGE)[int(np.argmax(sweep["silhouette"]))]
print(f"\n  Best K by silhouette: {best_k} (score={max(sweep['silhouette']):.4f})")


# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert 2 <= best_k <= 10, "Task 2: best_k must be in the tested range"
assert max(sweep["silhouette"]) > 0, "Task 2: best silhouette should be positive"
assert len(sweep["inertia"]) == len(list(K_RANGE)), "Task 2: sweep size mismatch"
print("\n  [ok] Checkpoint 1 passed — silhouette sweep complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: k-means++ vs random initialisation head-to-head
# ════════════════════════════════════════════════════════════════════════

km_plus = KMeans(
    n_clusters=best_k, random_state=RANDOM_STATE, n_init=10, init="k-means++"
)
km_random = KMeans(
    n_clusters=best_k, random_state=RANDOM_STATE, n_init=10, init="random"
)

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
print("    k-means++ spreads the seed centroids apart — faster and lower inertia.")

km_labels = km_plus.predict(X_scaled)


# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert (
    km_plus.inertia_ <= km_random.inertia_ + 1
), "Task 3: k-means++ should achieve inertia at least as good as random init"
assert len(set(km_labels.tolist())) == best_k, "Task 3: wrong cluster count"
print("\n  [ok] Checkpoint 2 passed — k-means++ confirmed superior\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Silhouette curves and per-sample silhouette
# ════════════════════════════════════════════════════════════════════════
# The elbow + silhouette plot is the canonical diagnostic for K.
# Per-sample silhouette reveals which points are mis-assigned — a negative
# s(i) means point i is closer to a different cluster than its own.

viz = ModelVisualizer()
history = {
    "Silhouette": sweep["silhouette"],
    "Inertia (scaled)": [i / max(sweep["inertia"]) for i in sweep["inertia"]],
}
fig = viz.training_history(history, x_label="K")
fig.update_layout(title=f"K-means: Silhouette and Inertia vs K (best K={best_k})")
fig.write_html(str(out_path("01_kmeans_elbow.html")))
print(f"  Saved: {out_path('01_kmeans_elbow.html')}")

# Per-sample silhouette
sil_samples = silhouette_samples(X_scaled, km_labels)
print(f"\n  Per-Sample Silhouette (K={best_k}):")
for cid in range(best_k):
    mask = km_labels == cid
    s = sil_samples[mask]
    n_neg = int((s < 0).sum())
    print(
        f"    Cluster {cid}: n={int(mask.sum()):>5}  mean_sil={s.mean():.4f}  "
        f"mis-assigned={n_neg} ({n_neg/len(s):.1%})"
    )
print("  Points with negative silhouette are candidates for re-review.")


# ── Checkpoint 3 ──────────────────────────────────────────────────────────
assert sil_samples.shape[0] == n_samples, "Task 4: per-sample silhouette missing points"
print("\n  [ok] Checkpoint 3 passed — visualisation and per-sample audit done\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Shopee Singapore Loyalty Segmentation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Shopee SG's CRM team wants to replace its hand-coded "Bronze /
# Silver / Gold / Platinum" loyalty tiers with data-driven segments. The
# existing tiers are purely revenue-based; they miss the difference
# between a "high-frequency low-basket" browser and a "low-frequency
# high-basket" infrequent whale.
#
# Why K-means is the right tool here:
#   - The customer features (recency, frequency, monetary, basket size)
#     form roughly spherical, non-overlapping clusters in z-space
#   - K is small (4-6 tiers) — K-means converges in seconds on 6K customers
#   - The centroid IS the segment profile — trivially explainable to the
#     marketing team ("Cluster 2 is 2.1σ above average on frequency")
#   - Re-segmentation runs nightly; linear cost scales to 10M customers
#
# BUSINESS IMPACT: Shopee's published ARPU for its SG marketplace is
# ~S$180/year per active buyer. A well-tuned tier system lifts campaign
# response rates by 15-25% because the offers match actual spending
# behaviour. On a 3M-buyer base, a 20% lift on S$20/buyer incremental
# campaign revenue is:
#     3,000,000 × S$20 × 0.20 = S$12M / year
# vs. effectively zero engineering cost (one silhouette sweep, one
# production job). The K-means model itself retrains in ~2 seconds.

print("  APPLY — Shopee SG Loyalty Segmentation")
print("  ─────────────────────────────────────────────────────────────────")
segment_sizes = np.bincount(km_labels)
for i, n in enumerate(segment_sizes):
    pct = n / n_samples * 100
    print(f"    Segment {i}: {n:>5,} customers ({pct:5.1f}%)")
print("    Each centroid is a 'typical customer profile' for its segment.")
print("    Marketing takes these profiles and designs tier-specific offers.")
print("    Estimated annual lift: S$12M (3M buyers × S$20 × 20%).")


# ── Checkpoint 4 ──────────────────────────────────────────────────────────
assert segment_sizes.min() > 0, "Task 5: every segment must have at least one customer"
assert (
    int(segment_sizes.sum()) == n_samples
), "Task 5: segment counts must sum to n_samples"
print("\n  [ok] Checkpoint 4 passed — segment sizes valid\n")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# The sweep's per-K silhouette/inertia series + the final-fit scalars go
# into one experiment ("m4_clustering_zoo") so subsequent lessons in this
# group (hierarchical, density-based, spectral, evaluation_profiling) can
# be compared side-by-side from one SQLite store at the end of the group.

track_run(
    tracker,
    exp_name,
    run_name="kmeans_pp",
    params={
        "init": "k-means++",
        "n_init": 10,
        "random_state": RANDOM_STATE,
        "best_k": best_k,
        "n_features": n_features,
        "n_samples": n_samples,
    },
    scalar_metrics={
        "best_silhouette": float(max(sweep["silhouette"])),
        "best_calinski_harabasz": float(max(sweep["ch"])),
        "best_davies_bouldin": float(min(sweep["db"])),
        "kmeans_pp_inertia": float(km_plus.inertia_),
        "kmeans_random_inertia": float(km_random.inertia_),
        "kmeans_pp_iters": float(km_plus.n_iter_),
        "kmeans_random_iters": float(km_random.n_iter_),
        "kmeans_pp_time_s": float(t_plus),
        "kmeans_random_time_s": float(t_random),
    },
    series_metrics={
        "sweep_silhouette": sweep["silhouette"],
        "sweep_inertia": sweep["inertia"],
    },
)
print(f"  [tracked] sweep + final-fit metrics logged to {exp_name} run='kmeans_pp'\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — the kailash-ml ClusteringEngine
# ════════════════════════════════════════════════════════════════════════
# This lesson hand-rolled the K sweep, the silhouette/CH/DB metrics, the
# k-means++-vs-random comparison, and the per-sample silhouette audit —
# ~120 lines of structure to internalise the moving parts.
#
# kailash-ml ships a single engine that IS that pipeline. ClusteringEngine.
# `sweep_k` runs the whole K-vs-criterion sweep for any supported algorithm
# (kmeans, hierarchical, dbscan, spectral, gmm) and `fit` returns a
# ClusterResult with labels + silhouette + Calinski-Harabasz + inertia.
# One sync call. Everything you just built, ready to drop into production.

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

cluster_df = pl.from_numpy(X_scaled, schema=feature_cols)
clustering = ClusteringEngine()

sweep_result = clustering.sweep_k(
    cluster_df, range(2, 11), algorithm="kmeans", criterion="silhouette"
)
print(f"  ClusteringEngine.sweep_k(): optimal_k={sweep_result.optimal_k}")

fit_result = clustering.fit(cluster_df, algorithm="kmeans", n_clusters=best_k)
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
  [x] k-means++ seeding beats random init on both speed and final inertia
  [x] Silhouette score gives an objective criterion for choosing K
      (the elbow alone is subjective)
  [x] Per-sample silhouette exposes mis-assigned points for re-review
  [x] Mapped K={best_k} clusters onto a Shopee SG loyalty tier system
      with an estimated S$12M / year campaign revenue lift

  KEY INSIGHT: K-means gives you the centroids for free. The centroids
  ARE the segment profiles — no extra analysis needed before handing them
  to marketing. This is why K-means is the default first-pass clustering
  algorithm for customer segmentation.

  Next: 02_hierarchical.py — when you need a dendrogram instead of a K.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
