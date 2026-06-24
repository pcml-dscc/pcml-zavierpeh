# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 3.5: Method comparison + intrinsic dimensionality
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compare PCA, Kernel PCA, t-SNE, UMAP, Isomap on one ruler
#   - Estimate intrinsic dimensionality (variance, Kaiser, NN MLE)
#   - Pick the right reducer given production vs visualisation goals
#
# PREREQUISITES: 01-04_*.py (all four previous files).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why intrinsic dim is the right target
#   2. Build — five reducers on the same data
#   3. Train — compute silhouette for every configuration
#   4. Visualise — leaderboard + intrinsic-dimensionality summary
#   5. Apply — GovTech Singapore FormSG respondent segmentation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.neighbors import NearestNeighbors

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_3 import (
    OUTPUT_DIR,
    evaluate_embedding_silhouette,
    load_customer_matrix,
    setup_engines,
    subsample_indices,
    teardown_engines,
    track_run,
)

# ── Kailash-ML ExperimentTracker — every dim-reduction run logs here ─────
tracker, exp_name = setup_engines()

try:
    import umap as umap_lib  # type: ignore

    UMAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    umap_lib = None
    UMAP_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════
# THEORY — intrinsic dimensionality
# ════════════════════════════════════════════════════════════════════════
# Your data has p ambient dimensions but varies along d << p independent
# axes. If d is small, dim-reduction is nearly free. If d ≈ p, it hurts.
# Estimate d BEFORE picking a method.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: shared preprocessing + pre-reduction
# ════════════════════════════════════════════════════════════════════════

X, feature_cols, _ = load_customer_matrix()
n_samples, n_features = X.shape
print(f"=== E-commerce customers ===  n={n_samples:,}, p={n_features}")

# TODO: fit a full-rank PCA on X. We need it for intrinsic-dim estimates
# AND for the pre-reduced inputs to t-SNE/UMAP.
pca_full = ____
pca_full.fit(X)
evr = pca_full.explained_variance_ratio_
cum_evr = np.cumsum(evr)
explained_variance = pca_full.explained_variance_

n_80 = int(np.searchsorted(cum_evr, 0.80) + 1)
n_90 = int(np.searchsorted(cum_evr, 0.90) + 1)
n_95 = int(np.searchsorted(cum_evr, 0.95) + 1)

X_pca10 = pca_full.transform(X)[:, : min(10, n_features)]
idx = subsample_indices(n_samples, n_target=3000)
# UMAP/Isomap transform target — 50K × N configs is too slow for a comparison.
# A 10K OOS slice still produces a meaningful silhouette ranking.
TRANSFORM_TARGET = 10_000
transform_idx = subsample_indices(n_samples, n_target=TRANSFORM_TARGET)
X_pca10_oos = X_pca10[transform_idx]


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: silhouette across every reducer
# ════════════════════════════════════════════════════════════════════════

method_silhouettes: dict[str, float] = {}

# (a) PCA at 2D, 80%, 90%, 95% variance
for n_comp in [2, n_80, n_90, n_95]:
    pca_test = PCA(n_components=n_comp, random_state=42)
    X_test = pca_test.fit_transform(X)
    method_silhouettes[f"PCA {n_comp}d"] = evaluate_embedding_silhouette(X_test)

# (b) Kernel PCA — three configs on the subsample
for kernel, params, label in [
    ("rbf", {"gamma": 0.1}, "KernelPCA rbf g=0.1"),
    ("rbf", {"gamma": 1.0}, "KernelPCA rbf g=1.0"),
    ("poly", {"degree": 3, "gamma": 0.1}, "KernelPCA poly d=3"),
]:
    # TODO: build and fit a KernelPCA(n_components=8, kernel=..., **params)
    # on X[idx], then record its silhouette.
    kpca = ____
    X_kpca = ____
    method_silhouettes[label] = evaluate_embedding_silhouette(X_kpca)

# (c) t-SNE at a few perplexities
for perp in [15, 30, 50]:
    # TODO: fit TSNE(n_components=2, perplexity=perp, max_iter=1000,
    # random_state=42, init='pca', learning_rate='auto') on X_pca10[idx].
    tsne = ____
    emb = ____
    method_silhouettes[f"t-SNE p={perp}"] = evaluate_embedding_silhouette(emb)

# (d) UMAP — three configs
if UMAP_AVAILABLE:
    for n_nbr, min_d, label in [
        (15, 0.1, "UMAP default"),
        (50, 0.5, "UMAP global"),
        (15, 0.0, "UMAP tight"),
    ]:
        reducer = umap_lib.UMAP(
            n_components=2,
            n_neighbors=n_nbr,
            min_dist=min_d,
            random_state=42,
            metric="euclidean",
        )
        reducer.fit(X_pca10[idx])
        emb_full = reducer.transform(X_pca10_oos)
        method_silhouettes[label] = evaluate_embedding_silhouette(emb_full)

# (e) Isomap — manifold learning reference
iso = Isomap(n_components=2, n_neighbors=10)
X_iso = iso.fit_transform(X_pca10[idx])
method_silhouettes["Isomap k=10"] = evaluate_embedding_silhouette(X_iso)


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert (
    len(method_silhouettes) >= 10
), f"Expected >=10 method configurations, got {len(method_silhouettes)}"
print(
    f"\n[ok] Checkpoint 1 — {len(method_silhouettes)} reducer configurations scored\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: leaderboard + intrinsic dimensionality
# ════════════════════════════════════════════════════════════════════════

print("=== Leaderboard (descending silhouette) ===")
for name, sil in sorted(method_silhouettes.items(), key=lambda x: -x[1]):
    print(f"  {name:<30}: {sil:+.4f}")

viz = ModelVisualizer()
fig = viz.metric_comparison(
    {name: {"Silhouette": sil} for name, sil in method_silhouettes.items()}
)
fig.update_layout(title="Dimensionality reduction: cluster-quality leaderboard")
leaderboard_path = OUTPUT_DIR / "05_leaderboard.html"
fig.write_html(str(leaderboard_path))
print(f"\nSaved: {leaderboard_path}")


# Intrinsic dimensionality estimators
# TODO: Kaiser — count eigenvalues greater than 1.
n_kaiser = ____

broken_stick = np.array(
    [
        sum(1.0 / j for j in range(i, n_features + 1)) / n_features
        for i in range(1, n_features + 1)
    ]
)
n_broken = int((evr > broken_stick).sum())


def estimate_intrinsic_dim_nn(
    X: np.ndarray, k_values: list[int], n_sub: int = 1000
) -> float:
    """Levina-Bickel MLE estimator of intrinsic dimension."""
    rng = np.random.default_rng(42)
    sub = rng.choice(len(X), min(n_sub, len(X)), replace=False)
    X_s = X[sub]
    nn = NearestNeighbors(n_neighbors=max(k_values)).fit(X_s)
    dists, _ = nn.kneighbors(X_s)
    log_ratios = []
    for k in k_values:
        d_k = dists[:, k - 1]
        d_1 = dists[:, 0]
        valid = (d_k > 0) & (d_1 > 0)
        if valid.sum() > 10:
            log_ratios.append(float(np.mean(np.log(d_k[valid] / d_1[valid]))))
    if not log_ratios:
        return float("nan")
    m = float(np.mean(log_ratios))
    return 1.0 / m if m > 0 else float("nan")


intrinsic_mle = estimate_intrinsic_dim_nn(X, k_values=[5, 10, 20, 30])

print("\n=== Intrinsic dimensionality estimates ===")
print(f"  Ambient (p)          : {n_features}")
print(f"  PCA 80% variance     : {n_80}")
print(f"  PCA 90% variance     : {n_90}")
print(f"  PCA 95% variance     : {n_95}")
print(f"  Kaiser (eig > 1)     : {n_kaiser}")
print(f"  Broken-stick         : {n_broken}")
print(f"  NN MLE (Levina-Bickel): {intrinsic_mle:.1f}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert n_90 <= n_features, "intrinsic dim cannot exceed ambient"
assert 1 <= n_kaiser <= n_features
print("\n[ok] Checkpoint 2 — intrinsic dimensionality estimated via 4 methods\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: GovTech Singapore FormSG respondent segmentation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: GovTech's FormSG serves ~5M form submissions/yr. Three
# audiences need different reducers from the same data:
#   - PM deck         -> t-SNE picture (micro-segments)
#   - DS team         -> PCA features (invertible, stable)
#   - Production ML   -> UMAP embedder (out-of-sample)
# Segmentation-driven UX fixes cut form abandonment by 14% -> ~700K more
# completed forms/yr -> ~S$8.4M/yr in avoided manual handling.
# WORKFLOW: run this notebook on the monthly snapshot, use intrinsic dim
# to bound expectations, pick per audience from the leaderboard.

top_method, top_sil = max(method_silhouettes.items(), key=lambda kv: kv[1])
print(f"\n=== GovTech FormSG projection ===")
print(f"  Top reducer on this dataset : {top_method}  (silhouette {top_sil:+.4f})")
print(f"  Intrinsic dim (NN MLE)      : {intrinsic_mle:.1f}")
print(f"  Ambient dim                 : {n_features}")


# ════════════════════════════════════════════════════════════════════════
# DECISION GUIDE
# ════════════════════════════════════════════════════════════════════════
print(
    """

  PRODUCTION:   PCA first, UMAP if PCA is insufficient.
  VISUALISE:    t-SNE for dense micro-clusters, UMAP for mixed scales.
  EXPLAIN:      PCA — it's the only one with true inverse_transform.
"""
)


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# This is the FINAL lesson in the M4 ex_3 dim-reduction block. After this,
# m4_dimreduction_zoo holds five runs across PCA / Kernel-PCA / t-SNE /
# UMAP / cross-method comparison.


def _slug(s: str) -> str:
    """Tracker metric keys allow only [A-Za-z0-9_.-]; slugify the rest."""
    out = "".join(c if c.isalnum() or c in "_.-" else "_" for c in s)
    return out.lstrip("_") or "k"


# TODO: call track_run with run_name "method_comparison". scalar_metrics
# headline = top_silhouette + intrinsic_dim_mle + four PCA component picks
# + n_kaiser + n_broken_stick, then |-merge per-method silhouettes from
# method_silhouettes — use _slug(name) as the suffix because method names
# like 'PCA 2d' / 't-SNE p=15' contain spaces and equals signs.
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "algorithms_compared": "pca,kernel_pca,tsne,umap,isomap",
        "n_configs": len(method_silhouettes),
        "n_features_ambient": n_features,
        "n_samples": n_samples,
        "intrinsic_dim_method": "levina_bickel_nn_mle",
    },
    scalar_metrics={
        "top_silhouette": float(top_sil),
        # NaN-guard: Levina-Bickel returns NaN when k-NN distances degenerate;
        # the tracker rejects non-finite values. Same pattern as ex_2/01's
        # `recovered_silhouette` guard.
        "intrinsic_dim_mle": (
            float(intrinsic_mle) if intrinsic_mle == intrinsic_mle else 0.0
        ),
        "n_components_80": float(n_80),
        "n_components_90": float(n_90),
        "n_components_95": float(n_95),
        "n_kaiser": float(n_kaiser),
        "n_broken_stick": float(n_broken),
    }
    | ____,
)
print(f"  [tracked] cross-method leaderboard logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — DimReductionEngine across the four supported
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1's DimReductionEngine wraps pca / tsne / umap / nmf
# under one `reduce` surface. The leaderboard you just built is the
# engine's natural output: same input, four algorithms, one comparable
# silhouette ruler. Run them through the engine to confirm the cross-
# method story holds end-to-end.

import polars as pl

from kailash_ml.engines.dim_reduction import DimReductionEngine

cust_df = pl.from_numpy(X[idx], schema=feature_cols)
dimreduce = DimReductionEngine()

engine_leaderboard: dict[str, float] = {}
for alg in ("pca", "tsne", "umap", "nmf"):
    try:
        if alg == "nmf":
            # NMF requires non-negative data; shift to make it valid.
            cust_for_nmf = pl.from_numpy(
                X[idx] - X[idx].min(axis=0) + 1e-6, schema=feature_cols
            )
            r = dimreduce.reduce(cust_for_nmf, algorithm=alg, n_components=2)
        else:
            r = dimreduce.reduce(cust_df, algorithm=alg, n_components=2)
        emb = np.asarray(r.transformed)
        engine_leaderboard[f"engine.{alg}"] = evaluate_embedding_silhouette(emb)
    except Exception as exc:  # pragma: no cover — engine drift safety
        engine_leaderboard[f"engine.{alg}"] = float("nan")
        print(f"  engine.{alg}: skipped ({type(exc).__name__})")

print("\n  DimReductionEngine.reduce leaderboard:")
for name, sil in sorted(
    engine_leaderboard.items(), key=lambda x: -(x[1] if x[1] == x[1] else -1)
):
    print(f"    {name:<22}: {sil:+.4f}")
print(
    "\n  Same silhouette ruler, four algorithms, one engine surface —"
    " open mlfp04_ex3_dimreduction.db for the full m4_dimreduction_zoo"
    " leaderboard.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Ran five reducer families on the same dataset with one metric
  [x] Built a silhouette leaderboard across all configurations
  [x] Estimated intrinsic dimensionality with four methods
  [x] Picked per-audience reducers for a real GovTech use case

  KEY INSIGHT: There is no "best" dimensionality reducer — only a best
  reducer FOR A SPECIFIC AUDIENCE AND DOWNSTREAM TASK.

  Exercise 3 complete. Next: Exercise 4 — anomaly detection ensembles.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
