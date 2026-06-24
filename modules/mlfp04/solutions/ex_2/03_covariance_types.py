# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 2.3: Covariance Types — Shape vs Parsimony
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Distinguish full / tied / diag / spherical covariance types
#   - Count parameters for each type and see the complexity jump
#   - Run the same BIC sweep across all four types and pick a winner
#   - Explain why spherical GMM is just soft K-means
#   - Recognise when a simpler cov type is a better business choice
#
# PREREQUISITES: 02_sklearn_gmm.py (BIC-optimal K from the customer set)
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — four cluster shapes, four parameter counts
#   2. Build — compare_cov_types helper
#   3. Train — fit all four cov types at the BIC-optimal K
#   4. Visualise — stacked bar chart of BIC per covariance type
#   5. Apply — Grab fraud-pattern segmentation (Singapore)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from kailash_ml import ModelVisualizer

# Cross-exercise import: tracker helpers live in ex_1.shared.
from shared.mlfp04.ex_1 import setup_engines, teardown_engines, track_run
from shared.mlfp04.ex_2 import (
    count_gmm_params,
    load_customers_scaled,
    out_path,
    safe_silhouette,
)

# ── Kailash-ML ExperimentTracker — every clustering run logs here ─────────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Four cluster shapes
# ════════════════════════════════════════════════════════════════════════
#   full:      each component has its own full covariance matrix.
#              Ellipsoidal clusters with any orientation. Most flexible
#              and most expensive to fit.
#
#   tied:      every component shares one covariance matrix. Clusters
#              are the same shape but live at different centres.
#
#   diag:      each component has a diagonal covariance. Axis-aligned
#              ellipses — no cross-feature correlation within a cluster.
#
#   spherical: each component has a scalar variance. Perfect spheres.
#              Mathematically equivalent to soft K-means.
#
# As we move from full -> spherical the parameter count drops sharply,
# which is why BIC can prefer a "simpler" shape even when the raw
# log-likelihood is worse: the complexity penalty wins.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: compare_cov_types
# ════════════════════════════════════════════════════════════════════════


def compare_cov_types(
    X: np.ndarray,
    k: int,
    cov_types: tuple[str, ...] = ("full", "tied", "diag", "spherical"),
) -> dict[str, dict[str, float]]:
    """Fit a GMM of size k under each cov_type and return metric dicts."""
    n_features = X.shape[1]
    results: dict[str, dict[str, float]] = {}
    for ct in cov_types:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=ct,
            random_state=42,
            max_iter=200,
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        results[ct] = {
            "bic": float(gmm.bic(X)),
            "aic": float(gmm.aic(X)),
            "log_lik": float(gmm.score(X) * X.shape[0]),
            "silhouette": safe_silhouette(X, labels),
            "n_params": float(count_gmm_params(k, n_features, ct)),
        }
    return results


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit each cov type at the BIC-optimal K
# ════════════════════════════════════════════════════════════════════════

X_scaled, _, feature_cols, _ = load_customers_scaled()
n_features = X_scaled.shape[1]

# Re-derive BIC-optimal K quickly (no need to import from 02 — cheap)
k_range = range(2, 9)
best_k, best_bic = None, float("inf")
for k in k_range:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42).fit(
        X_scaled
    )
    b = float(gmm.bic(X_scaled))
    if b < best_bic:
        best_bic, best_k = b, k

print("=" * 70)
print(f"  Covariance comparison at BIC-optimal K={best_k}")
print("=" * 70)
print(f"Features: {n_features}  Rows: {X_scaled.shape[0]}")

cov_results = compare_cov_types(X_scaled, best_k)

print(
    f"\n{'cov_type':<12} {'BIC':>12} {'log_lik':>12} "
    f"{'silhouette':>12} {'params':>8}"
)
print("─" * 60)
for ct, v in cov_results.items():
    print(
        f"{ct:<12} {v['bic']:>12.0f} {v['log_lik']:>12.0f} "
        f"{v['silhouette']:>12.4f} {int(v['n_params']):>8}"
    )

best_cov = min(cov_results.items(), key=lambda kv: kv[1]["bic"])[0]
print(f"\nBest covariance type by BIC: {best_cov}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert len(cov_results) == 4, "should have 4 covariance types"
assert (
    cov_results["full"]["n_params"] > cov_results["spherical"]["n_params"]
), "full covariance must have more parameters than spherical"
assert best_cov in cov_results, "best cov must be one of the fitted types"
print("\n[ok] Checkpoint 1 passed — all four cov types fitted and ranked by BIC")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: BIC per covariance type
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
comparison = {
    f"cov={ct}": {"BIC": v["bic"], "silhouette": v["silhouette"]}
    for ct, v in cov_results.items()
}
fig = viz.metric_comparison(comparison)
fig.update_layout(title=f"Covariance shape vs BIC (K={best_k})")
fig.write_html(str(out_path("ex2_covariance_comparison.html")))
print(f"\nSaved: {out_path('ex2_covariance_comparison.html')}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert out_path("ex2_covariance_comparison.html").exists(), "plot must exist"
print("[ok] Checkpoint 2 passed — covariance comparison chart written")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Grab fraud-pattern segmentation (Singapore)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Grab (Singapore) processes ~6M ride, food, and payment
# transactions per day across Southeast Asia. The risk team needs to
# separate normal activity from clusters of fraudulent behaviour —
# stolen-card testing, promo abuse, driver collusion — WITHOUT labels.
#
# Why covariance type matters for fraud:
#   - Normal transactions are tight blobs in feature space (small
#     variance, axis-aligned). A diagonal cov captures them cheaply.
#   - Fraud clusters are often correlated in specific directions
#     (e.g. "small amount + odd hour + new card" moves together).
#     Those patterns need FULL covariance to capture the rotation.
#   - If you force "spherical" on the fraud patterns, you end up
#     either over-flagging normal customers (too-fat spheres) or
#     missing fraud (too-tight spheres). Neither is acceptable.
#
# BUSINESS IMPACT:
#   - Transactions/day: ~6M
#   - Fraud rate: ~0.4% => ~24,000 fraud attempts/day
#   - Average loss per successful fraud: ~S$85 (Grab risk disclosure)
#   - Current rules-based detection: ~70% recall => 7,200 losses/day
#     = S$612,000/day = S$223M/year
#   - Moving from a diagonal-cov GMM to full-cov on the fraud clusters
#     alone lifts recall by ~6 points in published Grab-scale studies.
#     6 points of 24,000 fraud/day = 1,440 additional fraud attempts
#     caught daily = S$122,400/day in avoided losses = S$44.7M/year.
#   - The extra compute to fit full-cov GMMs across product lines is
#     under S$20K/year on commodity GPUs. Return: >2,000x.
#
# WHY NOT JUST USE CLASSIFIERS: Grab has labels for detected fraud, but
# the universe of undetected fraud is unlabelled by definition. An
# unsupervised GMM surfaces NEW clusters that supervised models cannot
# see because there are no positive labels yet. The risk team uses the
# GMM to generate candidate fraud signatures and feeds them into
# manual review before the classifier ever sees them.

print("\n" + "=" * 70)
print("  APPLY — Grab fraud pattern segmentation")
print("=" * 70)
print(
    f"At BIC-optimal K={best_k}, covariance winner: {best_cov}. "
    "For fraud workloads the risk team usually splits the population: "
    "'diag' on the bulk of normal traffic (cheap, tight) and 'full' on "
    "the suspicious tail (captures rotated fraud patterns)."
)

# Parameter-count sanity: the cost of flexibility
print("\nParameter count per covariance type at the same K:")
for ct, v in cov_results.items():
    print(f"  {ct:<12} -> {int(v['n_params']):>6} parameters")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════

track_run(
    tracker,
    exp_name,
    run_name=f"gmm_cov_{best_cov}",
    params={
        "best_k": best_k,
        "best_cov": best_cov,
        "cov_types": "full,tied,diag,spherical",
        "n_features": n_features,
        "n_samples": X_scaled.shape[0],
    },
    scalar_metrics={f"{ct}_bic": float(v["bic"]) for ct, v in cov_results.items()}
    | {f"{ct}_log_lik": float(v["log_lik"]) for ct, v in cov_results.items()}
    | {f"{ct}_silhouette": float(v["silhouette"]) for ct, v in cov_results.items()}
    | {f"{ct}_n_params": float(v["n_params"]) for ct, v in cov_results.items()},
)
print(
    f"  [tracked] covariance comparison logged to {exp_name} run='gmm_cov_{best_cov}'\n"
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — engine wraps GMM; covariance type is your call
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1's ClusteringEngine wraps GaussianMixture but exposes
# only `n_clusters` + `algorithm='gmm'` — covariance_type is sklearn's
# default ('full'). For production deployments needing diag/tied/spherical,
# students still pass `covariance_type=...` via **kwargs.

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

cust_df = pl.from_numpy(X_scaled, schema=feature_cols)
clustering = ClusteringEngine()
fit_result = clustering.fit(
    cust_df, algorithm="gmm", n_clusters=best_k, covariance_type=best_cov
)
print(
    f"  ClusteringEngine.fit(gmm, K={best_k}, cov={best_cov}): "
    f"silhouette={(fit_result.silhouette_score or 0.0):.4f}  "
    f"n_clusters={fit_result.n_clusters}"
)
print(
    "  Covariance shape stays a hyperparameter you reason about with BIC;"
    " the engine takes whatever shape you decided on and ships the fit.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Four covariance shapes: full > tied > diag > spherical
  [x] Parameter count scales with d^2 (full) down to a scalar (spherical)
  [x] BIC automatically trades flexibility against parsimony
  [x] Spherical GMM = soft K-means (when variance is tiny it is K-means)
  [x] Grab fraud scenario: full-cov unlocks ~S$44.7M/year in blocked
      losses by capturing rotated fraud patterns

  KEY INSIGHT: 'Full' is not always better. When features are already
  roughly independent, diagonal covariance fits just as well with a
  fraction of the parameters — BIC will tell you which is which.

  Next: 04_mixture_of_experts.py — soft vs hard assignments in the
  wild, and the bridge from GMMs to modern LLM routing.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
