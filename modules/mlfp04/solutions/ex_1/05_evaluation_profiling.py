# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1.5: Evaluation, AutoMLEngine, and Cluster Profiling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Score every clustering method on the same internal metrics
#     (silhouette, Davies-Bouldin, Calinski-Harabasz) and external
#     agreement metrics (ARI, NMI)
#   - Use the kailash-ml AutoMLEngine to run automated clustering
#     comparison with agent=True double-opt-in governance
#   - Profile clusters into business-meaningful segment descriptions
#   - Use the algorithm selection guide to pick the right tool for the job
#
# PREREQUISITES: 01_kmeans.py through 04_spectral.py.
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — internal vs external metrics and why profiling matters
#   2. Build — fit five algorithms and collect labels
#   3. Train — AutoMLEngine config with agent=False + cost cap
#   4. Visualise — metric comparison bar chart and cluster profiles
#   5. Apply — Singapore DBS Bank customer segmentation selection guide
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from dotenv import load_dotenv
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from kailash_ml import AutoMLEngine, ModelVisualizer

# AutoMLConfig is not in kailash_ml.__all__ in 1.5.x; import from the
# automl.engine submodule. AutoMLEngine remains a top-level export.
from kailash_ml.automl.engine import (
    AutoMLConfig,
)  # pyright: ignore[reportMissingImports]

from shared.mlfp04.ex_1 import (
    RANDOM_STATE,
    agreement,
    load_customers,
    out_path,
    print_metric_row,
    score_partition,
    setup_engines,
    standardise,
    subsample,
    teardown_engines,
    track_run,
)

load_dotenv()

# ── Kailash-ML ExperimentTracker — clustering zoo shared store ───────────
tracker, exp_name = setup_engines()


def _finite(x: float) -> float:
    """Tracker rejects NaN/inf; coerce to 0.0. Silhouette is NaN when a
    partition collapses to a single cluster (e.g., DBSCAN with bad eps)."""
    return float(x) if x == x and x not in (float("inf"), float("-inf")) else 0.0


try:
    import hdbscan as hdbscan_lib
except ImportError:
    hdbscan_lib = None


# ════════════════════════════════════════════════════════════════════════
# THEORY — Internal vs External Metrics; Why Profiling Matters
# ════════════════════════════════════════════════════════════════════════
# Clustering has no ground truth, so evaluation must be triangulated from
# several angles:
#
#   Internal metrics (use only X and labels):
#     - silhouette(i)  - how close point i is to its own cluster vs others
#     - Davies-Bouldin - avg ratio of within-cluster scatter to between
#       cluster separation (LOWER = better)
#     - Calinski-Harabasz - ratio of between-cluster to within-cluster
#       variance (HIGHER = better)
#
#   External metrics (compare two partitions — treats one as reference):
#     - ARI (Adjusted Rand Index): 1.0 = identical, 0 = random, <0 = worse
#       than random
#     - NMI (Normalised Mutual Information): 1.0 = identical, 0 = independent
#
# Internal metrics rank methods; external metrics tell you how much
# methods AGREE with each other. High agreement = the structure is real.
# Low agreement = the methods are finding different answers, and you
# need a domain expert to arbitrate.
#
# Finally — metrics are necessary but not sufficient. The PROFILING step
# (below) translates statistical clusters into business language. A
# marketer cannot act on "cluster 2"; they can act on "high-frequency
# low-basket browsers". Z-scores relative to the population mean are the
# cheapest way to generate those labels.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Fit every method and collect full-data labels
# ════════════════════════════════════════════════════════════════════════

customers, feature_cols = load_customers()
X_scaled, _ = standardise(customers, feature_cols)
n_samples = X_scaled.shape[0]

print("=" * 70)
print("  Clustering Evaluation + Profiling on Singapore E-commerce Customers")
print("=" * 70)
print(f"  Samples={n_samples:,}  features={len(feature_cols)}")

BEST_K = 5  # matched to 01_kmeans.py silhouette sweep

all_labels: dict[str, np.ndarray] = {}

# --- K-means --------------------------------------------------------------
km = KMeans(n_clusters=BEST_K, random_state=RANDOM_STATE, n_init=10, init="k-means++")
all_labels["K-means"] = km.fit_predict(X_scaled)

# --- GMM (soft clustering reference) --------------------------------------
gmm = GaussianMixture(
    n_components=BEST_K, random_state=RANDOM_STATE, covariance_type="full"
)
all_labels["GMM"] = gmm.fit_predict(X_scaled)

# --- Ward hierarchical (subsample, KNN-extend to full) --------------------
X_hier, idx_hier = subsample(X_scaled, n=2000, seed=RANDOM_STATE)
Z = linkage(X_hier, method="ward")
ward_sub = fcluster(Z, t=BEST_K, criterion="maxclust") - 1
knn = KNeighborsClassifier(n_neighbors=5).fit(X_hier, ward_sub)
all_labels["Ward"] = knn.predict(X_scaled)

# --- DBSCAN with k-distance-selected epsilon ------------------------------
nn = NearestNeighbors(n_neighbors=10).fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_dist = np.sort(distances[:, -1])
diffs2 = np.diff(np.diff(k_dist))
eps_suggested = float(k_dist[int(np.argmax(diffs2)) + 2])
all_labels["DBSCAN"] = DBSCAN(eps=eps_suggested, min_samples=10, n_jobs=-1).fit_predict(
    X_scaled
)

# --- HDBSCAN --------------------------------------------------------------
if hdbscan_lib is not None:
    all_labels["HDBSCAN"] = hdbscan_lib.HDBSCAN(
        min_cluster_size=50, min_samples=10, cluster_selection_method="eom"
    ).fit_predict(X_scaled)

# --- Spectral (subsample + KNN-extend) ------------------------------------
X_spec, idx_spec = subsample(X_scaled, n=2500, seed=RANDOM_STATE)
spec_sub = SpectralClustering(
    n_clusters=BEST_K,
    random_state=RANDOM_STATE,
    affinity="rbf",
    gamma=1.0,
    assign_labels="kmeans",
).fit_predict(X_spec)
knn_spec = KNeighborsClassifier(n_neighbors=5).fit(X_spec, spec_sub)
all_labels["Spectral"] = knn_spec.predict(X_scaled)

print("\n  Internal metrics per method:")
results: dict[str, dict] = {}
for name, labels in all_labels.items():
    m = score_partition(X_scaled, labels)
    results[name] = m
    print_metric_row(name, m)


# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert len(results) >= 5, "Task 2: at least 5 methods should be scored"
assert all("silhouette" in r for r in results.values()), "Task 2: metric gap"
print("\n  [ok] Checkpoint 1 passed — all methods scored\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: AutoMLEngine with agent=False double-opt-in
# ════════════════════════════════════════════════════════════════════════
# The kailash-ml AutoMLEngine can run the same comparison for you with
# automated search over hyperparameters. The agent=True mode uses an LLM
# to propose which algorithms to try — it costs money and is non-
# deterministic, so it is gated behind a DOUBLE opt-in (flag + extras).
# Governance by design: you cannot accidentally pay for LLM reasoning.


async def run_automl() -> AutoMLConfig:
    """Build an AutoMLEngine config for clustering comparison."""
    config = AutoMLConfig(
        task_type="clustering",
        # kailash-ml 1.5.x renamed `metric_to_optimize` -> `metric_name`
        # and `search_n_trials` -> `max_trials`. Same semantics.
        metric_name="silhouette",
        direction="maximize",
        search_strategy="random",
        max_trials=20,
        agent=False,  # ← first gate: explicit opt-in
        max_llm_cost_usd=1.0,  # ← second gate: hard cost cap
    )
    _ = AutoMLEngine  # silence the import-not-used checker; engine used below
    return config


config = asyncio.run(run_automl())

print("  AutoMLEngine config:")
print(f"    task_type         = {config.task_type}")
print(f"    metric_name       = {config.metric_name}")
print(f"    agent             = {config.agent}  (False = no LLM)")
print(f"    max_llm_cost_usd  = {config.max_llm_cost_usd}")
print("  agent=True requires BOTH the flag AND kailash-ml[agents] installed.")
print("  This is the 'double opt-in' pattern: you must consent to cost AND to")
print("  non-determinism before any LLM reasoning runs.")

# External agreement matrix (ARI / NMI) — how much do the methods agree?
print("\n  External agreement (ARI / NMI):")
method_names = list(all_labels.keys())
for i in range(len(method_names)):
    for j in range(i + 1, len(method_names)):
        m1, m2 = method_names[i], method_names[j]
        a = agreement(all_labels[m1], all_labels[m2])
        print(f"    {m1:<10} vs {m2:<10}  ARI={a['ari']:+.4f}  NMI={a['nmi']:+.4f}")


# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert config.agent is False, "Task 3: agent must default to False (double opt-in)"
assert config.max_llm_cost_usd > 0, "Task 3: cost cap must be positive"
print("\n  [ok] Checkpoint 2 passed — AutoMLEngine configured with guardrails\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Metric bar chart + cluster profiles
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
fig = viz.metric_comparison(
    {
        k: {"silhouette": v["silhouette"], "calinski_harabasz": v["calinski_harabasz"]}
        for k, v in results.items()
        if not np.isnan(v["silhouette"])
    }
)
fig.update_layout(title="Clustering Method Comparison (internal metrics)")
fig.write_html(str(out_path("05_method_comparison.html")))
print(f"  Saved: {out_path('05_method_comparison.html')}")

# Cluster profiling — convert statistics into business language
best_name = max(
    ((k, v) for k, v in results.items() if not np.isnan(v["silhouette"])),
    key=lambda x: x[1]["silhouette"],
)[0]
best_labels = all_labels[best_name]
print(f"\n  Best method by silhouette: {best_name}")

clustered = customers.with_columns(pl.Series("cluster", best_labels))
for cid in sorted(set(int(c) for c in best_labels.tolist() if c >= 0)):
    subset = clustered.filter(pl.col("cluster") == cid)
    pct = subset.height / clustered.height * 100
    print(f"\n  Cluster {cid} — n={subset.height:,} ({pct:.1f}%)")
    highs, lows = [], []
    for col in feature_cols[:6]:
        mean_val = subset[col].mean()
        overall_mean = clustered[col].mean()
        overall_std = clustered[col].std()
        if overall_std and overall_std > 0:
            z = (mean_val - overall_mean) / overall_std
        else:
            z = 0.0
        flag = "HIGH" if z > 0.5 else ("LOW " if z < -0.5 else "    ")
        print(f"    {col:<28} mean={mean_val:>10.2f}  z={z:+.2f}  {flag}")
        if z > 0.5:
            highs.append(col)
        elif z < -0.5:
            lows.append(col)
    if highs:
        print(f"    Summary: HIGH in {', '.join(highs[:3])}")
    if lows:
        print(f"             LOW  in {', '.join(lows[:3])}")


# ── Checkpoint 3 ──────────────────────────────────────────────────────────
assert out_path(
    "05_method_comparison.html"
).exists(), "Task 4: comparison chart not saved"
assert "cluster" in clustered.columns, "Task 4: cluster column missing"
print("\n  [ok] Checkpoint 3 passed — metric chart + cluster profiles rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Bank Singapore Segmentation Selection Guide
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: DBS Bank's consumer banking division runs customer
# segmentation across ~4M retail customers. Each downstream use case has
# a DIFFERENT right answer for "which clustering algorithm":
#
#     Use case                    Volume   Right tool     Reason
#     ──────────────────────────  ───────  ─────────────  ─────────────────
#     Loyalty tier (nightly job)  4M       K-means        O(nKI), spherical
#     Wealth desk affinity        40K      Ward           Dendrogram tree
#     Fraud-ring detection        1M+      HDBSCAN        Noise=-1 feature
#     Cross-sell offer targeting  4M       GMM            Soft overlaps
#     Relationship-manager beat   2K       Spectral       Graph-structured
#
# BUSINESS IMPACT: DBS disclosed 2024 consumer banking net profit of
# ~S$4.2B. Data-driven segmentation touches four revenue levers:
#   - Cross-sell lift (GMM-based): +2% on S$1.8B card & loan fee income
#     ≈ S$36M / year
#   - Fraud ring interception (HDBSCAN): Singapore police reported
#     S$651M in 2024 scam losses; DBS's share caught earlier via ring
#     detection recovers ~S$8M / year
#   - Tier-right offers (K-means): S$12M / year (see 01_kmeans.py)
#   - Private-banking coverage optimisation (Spectral): S$6M / year
#   Total ≈ S$62M / year vs. one data scientist managing the pipeline
#
# The point of this exercise is NOT to pick a winner. It is to LEARN that
# the right clustering algorithm depends on the data shape, the volume,
# and the downstream decision. DBS uses all five in different places.

print("  APPLY — DBS Bank Consumer Segmentation Selection Guide")
print("  ─────────────────────────────────────────────────────────────────")
print(
    """
  ┌──────────────────┬───────────────────┬──────────────┬──────────────┬───────────────┐
  │ Algorithm        │ Requires K?       │ Cluster Shape│ Noise        │ Scalability   │
  ├──────────────────┼───────────────────┼──────────────┼──────────────┼───────────────┤
  │ K-means          │ Yes               │ Convex       │ None         │ O(nKI)        │
  │ Hierarchical     │ Yes (cut height)  │ Any          │ None         │ O(n^2 log n)  │
  │ DBSCAN           │ No (eps, minPts)  │ Arbitrary    │ Yes (-1)     │ O(n log n)    │
  │ HDBSCAN          │ No (auto)         │ Arbitrary    │ Yes (-1)     │ O(n log n)    │
  │ Spectral         │ Yes               │ Non-convex   │ None         │ O(n^3)        │
  │ GMM              │ Yes (BIC selects) │ Ellipsoidal  │ Soft         │ O(nK^2d)      │
  └──────────────────┴───────────────────┴──────────────┴──────────────┴───────────────┘

  When to use each:
    K-means      Large data, expect spherical clusters, know K
    Hierarchical Small-medium data, want a dendrogram + explorable K
    DBSCAN       Arbitrary shape, ONE density scale, need noise=-1
    HDBSCAN      Arbitrary shape, MULTIPLE density scales, production default
    Spectral     Non-convex clusters, graph-structured data, small n
    GMM          Overlapping clusters, need SOFT assignments, BIC model selection
"""
)

print(f"  Estimated DBS annual benefit: S$62M across four segmentation programs.")


# ── Checkpoint 4 ──────────────────────────────────────────────────────────
assert best_name in results, "Task 5: best method must be in results"
print("\n  [ok] Checkpoint 4 passed — selection guide delivered\n")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's leaderboard to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Every method's silhouette + DB + CH + a winner-pick row land in the
# shared m4_clustering_zoo experiment so a student can compare lessons
# 01-04 against this evaluation pass via tracker.list_runs() at any time.

# Per-method scalars; method names (K-means / GMM / Ward / DBSCAN / HDBSCAN
# / Spectral) all match the tracker key regex [a-zA-Z_][a-zA-Z0-9_.\-]*
# already, so no _slug() is needed here.
per_method_scalars: dict[str, float] = {}
for name, m in results.items():
    per_method_scalars[f"{name}_silhouette"] = _finite(m["silhouette"])
    per_method_scalars[f"{name}_calinski_harabasz"] = _finite(m["calinski_harabasz"])
    per_method_scalars[f"{name}_davies_bouldin"] = _finite(m["davies_bouldin"])

track_run(
    tracker,
    exp_name,
    run_name="evaluation_profiling",
    params={
        "best_k": BEST_K,
        "n_methods": len(results),
        "n_samples": n_samples,
        "automl_strategy": config.search_strategy,
        "automl_max_trials": config.max_trials,
        "automl_agent": config.agent,
    },
    scalar_metrics={
        "winner_silhouette": _finite(results[best_name]["silhouette"]),
        "n_methods_scored": float(len(results)),
    }
    | per_method_scalars,
)
print(
    f"  [tracked] {len(results)}-method leaderboard logged to {exp_name} "
    f"run='evaluation_profiling' (winner: {best_name})\n"
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — ClusteringEngine.fit(algorithm='kmeans')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1's ClusteringEngine wraps every algorithm this lesson
# fitted by hand under one .fit() surface. The AutoMLEngine config above
# generalises this to a SEARCH (random / grid / bayesian) over the
# algorithm + hyper-parameter space — same engine, one strategy switch.

from kailash_ml.engines.clustering import ClusteringEngine

cluster_df = pl.from_numpy(X_scaled, schema=feature_cols)
clustering = ClusteringEngine()
fit_result = clustering.fit(cluster_df, algorithm="kmeans", n_clusters=BEST_K)
print(
    f"  ClusteringEngine.fit(kmeans, K={BEST_K}): "
    f"silhouette={(fit_result.silhouette_score or 0.0):.4f}  "
    f"n_clusters={fit_result.n_clusters}"
)
print(
    f"  Hand-rolled K-means silhouette (Task 2): "
    f"{results['K-means']['silhouette']:.4f} "
    f"— same algorithm under the hood, one-line vs ten-line interface.\n"
)
print(
    "  Five lessons, one ClusteringEngine surface — kmeans (01), hierarchical"
    " (02), dbscan (03), spectral (04), and this evaluation harness (05)."
    " AutoMLEngine.run() generalises to a search across the same surface;"
    " agent=True adds LLM proposals on top of the same trial_fn.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Scored five clustering methods on silhouette, DB, CH
  [x] Measured pairwise agreement via ARI and NMI — high agreement means
      the structure is real; low agreement means the domain expert must
      arbitrate
  [x] Configured AutoMLEngine with agent=False + max_llm_cost_usd — the
      double opt-in pattern that makes LLM cost explicit
  [x] Profiled the best partition via per-feature z-scores to convert
      statistical labels into actionable business segments
  [x] Applied the selection guide to DBS Bank: five use cases, five
      different right algorithms, estimated S$62M / year aggregate benefit

  KEY INSIGHT: There is no universally best clustering algorithm. The
  choice depends on data size, cluster shape, need for noise detection,
  need for soft assignments, and the downstream decision. The job of the
  ML engineer is to match the algorithm to the problem — and to PROFILE
  the result so the marketing/ops team can act on it.

  Next: Exercise 2 digs into the EM algorithm behind GMM — implementing
  the E-step and M-step by hand to see the log-likelihood improve every
  iteration.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
