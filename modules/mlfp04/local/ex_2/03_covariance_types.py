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
# PREREQUISITES: 02_sklearn_gmm.py
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — four cluster shapes, four parameter counts
#   2. Build — compare_cov_types helper
#   3. Train — fit all four cov types at the BIC-optimal K
#   4. Visualise — BIC per covariance type
#   5. Apply — Grab fraud-pattern segmentation (Singapore)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from kailash_ml import ModelVisualizer

# Cross-exercise import: tracker helpers live in ex_1.shared so every M4
# unsupervised technique logs to the same `m4_clustering_zoo` experiment.
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
#   full:      each component has its own full covariance (most flexible)
#   tied:      all components share one covariance (same shape, diff centre)
#   diag:      each component has a diagonal covariance (axis-aligned)
#   spherical: each component has a scalar variance (equivalent to K-means)


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
        # TODO: construct GaussianMixture with covariance_type=ct, n_components=k,
        # random_state=42, max_iter=200 — then .fit(X)
        gmm = ____
        ____
        labels = gmm.predict(X)
        # TODO: fill metrics — gmm.bic(X), gmm.aic(X), gmm.score(X)*n,
        # safe_silhouette(X, labels), count_gmm_params(k, n_features, ct)
        results[ct] = {
            "bic": ____,
            "aic": ____,
            "log_lik": ____,
            "silhouette": ____,
            "n_params": ____,
        }
    return results


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit each cov type at the BIC-optimal K
# ════════════════════════════════════════════════════════════════════════

X_scaled, _, feature_cols, _ = load_customers_scaled()
n_features = X_scaled.shape[1]

# Re-derive BIC-optimal K quickly
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

# TODO: call compare_cov_types(X_scaled, best_k)
cov_results = ____

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

# TODO: pick the cov type with the lowest BIC
best_cov = ____
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
# TODO: viz.metric_comparison(comparison)
fig = ____
fig.update_layout(title=f"Covariance shape vs BIC (K={best_k})")
fig.write_html(str(out_path("ex2_covariance_comparison.html")))
print(f"\nSaved: {out_path('ex2_covariance_comparison.html')}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert out_path("ex2_covariance_comparison.html").exists(), "plot must exist"
print("[ok] Checkpoint 2 passed — covariance comparison chart written")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Grab fraud-pattern segmentation (Singapore)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Grab processes ~6M daily transactions across SEA. The risk
# team must separate normal activity from fraud patterns — stolen-card
# testing, promo abuse, driver collusion — WITHOUT labels.
#
# Why covariance type matters:
#   Normal transactions cluster as tight, axis-aligned blobs (diagonal
#   cov fits cheaply). Fraud patterns are CORRELATED: e.g. "small
#   amount + odd hour + new card" moves together. Capturing the
#   rotation needs full covariance.
#
# BUSINESS IMPACT (from Grab risk disclosures):
#   - ~24,000 daily fraud attempts
#   - Current rules-based recall: ~70% => ~S$612K/day = S$223M/year losses
#   - Full-cov GMM on fraud clusters lifts recall ~6 points
#   - => ~S$44.7M/year in avoided losses vs <S$20K/year extra compute
#   - Return: >2,000x.

print("\n" + "=" * 70)
print("  APPLY — Grab fraud pattern segmentation")
print("=" * 70)
print(
    f"At BIC-optimal K={best_k}, covariance winner: {best_cov}. "
    "For fraud workloads the risk team usually splits the population: "
    "'diag' on the bulk of normal traffic and 'full' on the suspicious tail."
)

print("\nParameter count per covariance type at the same K:")
for ct, v in cov_results.items():
    print(f"  {ct:<12} -> {int(v['n_params']):>6} parameters")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# scalar_metrics merges four parallel per-cov-type dicts (bic, log_lik,
# silhouette, n_params) — the same |-merge pattern used in 02_hierarchical
# and 04_spectral.

# TODO: call track_run with run_name f"gmm_cov_{best_cov}". scalar_metrics
# is the |-merge of four dict comprehensions, all over cov_results.items():
#   {f"{ct}_bic": float(v["bic"]) ...}
#   | {f"{ct}_log_lik": float(v["log_lik"]) ...}
#   | {f"{ct}_silhouette": float(v["silhouette"]) ...}
#   | {f"{ct}_n_params": float(v["n_params"]) ...}
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "best_k": best_k,
        "best_cov": best_cov,
        "cov_types": "full,tied,diag,spherical",
        "n_features": n_features,
        "n_samples": X_scaled.shape[0],
    },
    scalar_metrics=____,
)
print(f"  [tracked] covariance comparison logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — engine wraps GMM; covariance type is your call
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1's ClusteringEngine wraps GaussianMixture but exposes
# only `n_clusters` + `algorithm='gmm'` — covariance_type is sklearn's
# default ('full'). For production deployments needing diag/tied/spherical,
# pass `covariance_type=...` via **kwargs.

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

cust_df = pl.from_numpy(X_scaled, schema=feature_cols)

# TODO: instantiate ClusteringEngine and call .fit on cust_df with
# algorithm='gmm', n_clusters=best_k, and covariance_type=best_cov.
clustering = ____
fit_result = ____
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
  [x] Four cov shapes: full > tied > diag > spherical
  [x] Parameter count scales with d^2 (full) down to a scalar (spherical)
  [x] BIC automatically trades flexibility against parsimony
  [x] Grab fraud scenario: full-cov unlocks ~S$44.7M/year in blocked losses

  Next: 04_mixture_of_experts.py — soft vs hard assignments, and the
  bridge from GMMs to modern LLM routing.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
