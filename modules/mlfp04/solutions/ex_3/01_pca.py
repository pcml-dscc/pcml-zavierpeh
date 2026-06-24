# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 3.1: Principal Component Analysis via SVD
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Derive PCA from the Singular Value Decomposition (X = U S V^T)
#   - Read a scree plot and pick n_components by variance threshold
#   - Interpret loadings to assign business meaning to each component
#   - Quantify compression quality via reconstruction error
#   - Recognise when PCA is the right tool (linear, fast, invertible)
#
# PREREQUISITES: MLFP04 Exercise 1 (clustering) + linear algebra basics.
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — why PCA = SVD of the centred data matrix
#   2. Build — compute SVD, verify against sklearn.PCA
#   3. Train — scree plot + three component-selection criteria
#   4. Visualise — loadings heatmap + reconstruction error curve
#   5. Apply — Shopee Singapore customer analytics compression
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_3 import (
    OUTPUT_DIR,
    load_customer_matrix,
    setup_engines,
    teardown_engines,
    track_run,
)

# ── Kailash-ML ExperimentTracker — every dim-reduction run logs here ─────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — PCA is SVD
# ════════════════════════════════════════════════════════════════════════
# Given a centred (and usually standardised) data matrix X of shape
# (n_samples, n_features), the Singular Value Decomposition gives:
#
#     X = U  S  V^T
#
#   - Columns of V are the principal directions (unit vectors in
#     feature space). These are the "axes of maximum variance".
#   - Singular values s_k encode the strength of each direction.
#     The variance explained by PC_k is  s_k^2 / (n - 1).
#   - Scores (the projection of each sample onto the PCs) = X V = U S.
#
# Everything else about PCA is bookkeeping on top of SVD. sklearn.PCA
# computes exactly this under the hood — we will verify it below.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: compute SVD, derive explained variance
# ════════════════════════════════════════════════════════════════════════

X, feature_cols, _ = load_customer_matrix()
n_samples, n_features = X.shape
print(f"=== E-commerce customers ===")
print(f"Samples: {n_samples:,}  Features: {n_features}")

U, S, Vt = np.linalg.svd(X, full_matrices=False)

explained_variance = S**2 / (n_samples - 1)
total_variance = explained_variance.sum()
evr = explained_variance / total_variance
cum_evr = np.cumsum(evr)

print(f"\nTotal variance (~n_features={n_features}): {total_variance:.2f}")
print(f"\nTop 10 principal components:")
print(f"{'PC':>4} {'Sing. val':>12} {'Expl. var %':>14} {'Cumulative %':>14}")
print("-" * 48)
for i in range(min(10, n_features)):
    print(f"{i + 1:>4} {S[i]:>12.4f} {evr[i]:>13.2%} {cum_evr[i]:>13.2%}")

# Cross-check against sklearn's implementation. These MUST agree to ~1e-6;
# if they don't, something is wrong with how X was centred/scaled.
pca_full = PCA(n_components=n_features)
pca_full.fit(X)
max_diff = float(np.abs(pca_full.explained_variance_ratio_ - evr).max())
print(f"\nSVD vs sklearn PCA max diff: {max_diff:.2e}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert max_diff < 1e-6, f"SVD must match sklearn.PCA, got {max_diff:.2e}"
assert abs(cum_evr[-1] - 1.0) < 1e-6, "cumulative variance must sum to 1"
print("[ok] Checkpoint 1 — SVD-derived PCA matches sklearn\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: scree plot + three selection criteria
# ════════════════════════════════════════════════════════════════════════
# There is no training loop for PCA — the "training" is the SVD itself.
# The decision we make here is how many components to RETAIN.

n_80 = int(np.searchsorted(cum_evr, 0.80) + 1)
n_90 = int(np.searchsorted(cum_evr, 0.90) + 1)
n_95 = int(np.searchsorted(cum_evr, 0.95) + 1)

# Kaiser: retain components whose eigenvalue > 1 (a component that
# captures more than one original feature's worth of variance).
n_kaiser = int((explained_variance > 1.0).sum())

# Broken-stick: a random partition of total variance would give each PC
# a share of sum_{j=i}^p 1/j / p. Retain PCs that beat that share.
broken_stick = np.array(
    [
        sum(1.0 / j for j in range(i, n_features + 1)) / n_features
        for i in range(1, n_features + 1)
    ]
)
n_broken = int((evr > broken_stick).sum())

print(f"=== Component-selection criteria ===")
print(f"  80% variance threshold : {n_80} components")
print(f"  90% variance threshold : {n_90} components")
print(f"  95% variance threshold : {n_95} components")
print(f"  Kaiser (eigenvalue>1)  : {n_kaiser} components")
print(f"  Broken-stick           : {n_broken} components")
print(
    f"  Compression at 95%     : "
    f"{n_features} -> {n_95} ({n_features / max(n_95, 1):.1f}x)"
)

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert n_80 <= n_90 <= n_95 <= n_features, "monotonic thresholds"
assert 1 <= n_kaiser <= n_features
print("[ok] Checkpoint 2 — variance thresholds + Kaiser + broken-stick\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: scree + loadings + reconstruction error
# ════════════════════════════════════════════════════════════════════════
# R9A: visual proof, not just a number. Three plots:
#   (a) scree + cumulative variance
#   (b) loadings for the first 5 PCs
#   (c) reconstruction error vs k

viz = ModelVisualizer()

# (a) Scree plot
fig_scree = viz.training_history(
    {
        "Explained variance %": (evr[:20] * 100).tolist(),
        "Cumulative %": (cum_evr[:20] * 100).tolist(),
    },
    x_label="Principal component",
)
fig_scree.update_layout(title="PCA scree: explained variance by component")
scree_path = OUTPUT_DIR / "01_pca_scree.html"
fig_scree.write_html(str(scree_path))
print(f"Saved: {scree_path}")

# (b) Loadings — which features drive each PC
n_pcs_inspect = min(5, n_features)
loadings = Vt[:n_pcs_inspect, :].T  # (n_features, n_pcs)

print(f"\n=== Loadings on first {n_pcs_inspect} PCs ===")
for i in range(n_pcs_inspect):
    col_norm = float(np.linalg.norm(loadings[:, i]))
    assert abs(col_norm - 1.0) < 1e-6, f"PC{i + 1} must be unit norm"
    abs_l = np.abs(loadings[:, i])
    top = np.argsort(abs_l)[::-1][:3]
    names = [f"{feature_cols[j]} ({loadings[j, i]:+.2f})" for j in top]
    print(f"  PC{i + 1}: {', '.join(names)}")

# (c) Reconstruction error as a function of k
# For standardised data, MSE(k) = sum_{j>k} s_j^2 / (n_samples * n_features)
n_range = list(range(1, min(n_features + 1, 21)))
recon_errors = []
for k in n_range:
    pca_k = PCA(n_components=k, random_state=42)
    X_proj = pca_k.fit_transform(X)
    X_hat = pca_k.inverse_transform(X_proj)
    recon_errors.append(float(np.mean((X - X_hat) ** 2)))

fig_recon = viz.training_history(
    {"Reconstruction MSE": recon_errors}, x_label="Components retained"
)
fig_recon.update_layout(title="PCA: reconstruction error vs components")
recon_path = OUTPUT_DIR / "01_pca_reconstruction.html"
fig_recon.write_html(str(recon_path))
print(f"Saved: {recon_path}")

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert recon_errors[0] > recon_errors[-1], "MSE must fall as k grows"
assert recon_errors[-1] < 0.1, "full-rank reconstruction should be ~0"
print("\n[ok] Checkpoint 3 — visualisations + loadings + reconstruction\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Shopee Singapore customer analytics compression
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Shopee SG (Sea Group) stores ~14 million active shoppers, each
# described by 120+ behavioural features (category views, basket size,
# delivery windows, coupon response, returns, support tickets...). The
# analytics team runs a nightly K-means segmentation to power the homepage
# recommendation carousel — but the full 120-D distance computation over
# 14M shoppers misses the 6-hour window before morning traffic spikes.
#
# WHY PCA IS THE RIGHT TOOL HERE:
#   - Linear and fast: O(np min(n, p)) on the nightly batch, ~8 minutes.
#   - Invertible: we can reconstruct any customer from their compressed
#     vector, which matters for downstream churn-model explainability.
#   - Interpretable: loadings let marketing name the components
#     ("Component 1 = price sensitivity, Component 2 = bulk buyers...").
#
# BUSINESS IMPACT: Retaining 95% of variance typically compresses 120-D
# shopper vectors to ~18-D. K-means on 18-D completes in ~42 minutes vs
# ~5 hours at full rank — comfortably inside the pre-dawn window. The
# recommendation carousel refreshes on time, and each on-time refresh is
# worth an estimated S$180K/day in incremental GMV (Sea Q3 2025 investor
# deck attributes ~0.4% basket lift to freshness of personalisation).
#
# At 95% variance we drop 5% of ambient variance — which is mostly noise,
# and removing noise is a feature, not a bug: the downstream segmentation
# is MORE stable under PCA than over raw features.

# Demonstrate the compression numerically.
compression_ratio = n_features / max(n_95, 1)
print(f"=== Shopee-style compression estimate ===")
print(f"  Raw dimensions       : {n_features}")
print(f"  At 95% variance      : {n_95}")
print(f"  Compression ratio    : {compression_ratio:.1f}x")
print(f"  Variance discarded   : {(1 - cum_evr[n_95 - 1]) * 100:.2f}%")
print(
    f"  Reconstruction MSE   : {recon_errors[min(n_95 - 1, len(recon_errors) - 1)]:.4f}"
)


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Every M4 ex_3 lesson logs into the SAME experiment ('m4_dimreduction_zoo')
# so PCA / Kernel-PCA / t-SNE / UMAP can be compared side-by-side from one
# SQLite store after the lesson group ends. Sweep series = per-k explained-
# variance + reconstruction-MSE; scalar metrics = the chosen-k headlines.

track_run(
    tracker,
    exp_name,
    run_name="pca_svd",
    params={
        "algorithm": "pca",
        "method": "svd",
        "n_features": n_features,
        "n_samples": n_samples,
        "n_components_chosen": n_95,
        "variance_threshold": 0.95,
    },
    scalar_metrics={
        "n_components_80": float(n_80),
        "n_components_90": float(n_90),
        "n_components_95": float(n_95),
        "n_components_kaiser": float(n_kaiser),
        "n_components_broken_stick": float(n_broken),
        "compression_ratio_at_95": float(compression_ratio),
        "variance_kept_at_95": float(cum_evr[n_95 - 1]),
        "recon_mse_at_95": float(recon_errors[min(n_95 - 1, len(recon_errors) - 1)]),
        "svd_vs_sklearn_max_diff": float(max_diff),
    },
    series_metrics={
        "explained_variance_ratio": evr.tolist(),
        "cumulative_explained_variance": cum_evr.tolist(),
        "reconstruction_mse": recon_errors,
    },
)
print(f"  [tracked] PCA scree + reconstruction series logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — the kailash-ml DimReductionEngine
# ════════════════════════════════════════════════════════════════════════
# This lesson hand-rolled the SVD, the explained-variance bookkeeping,
# the three component-selection criteria, and the reconstruction-error
# curve — ~140 lines of structure to internalise PCA from first principles.
#
# kailash-ml ships a single engine that IS that pipeline. DimReductionEngine.
# `reduce` runs the whole fit-and-transform on a polars DataFrame and
# returns a DimReductionResult with the embedding, explained variance, and
# reconstruction error. One sync call. The engine owns pca, tsne, umap,
# nmf — the next four lessons all pivot on this same surface.

import polars as pl

from kailash_ml.engines.dim_reduction import DimReductionEngine

cust_df = pl.from_numpy(X, schema=feature_cols)
dimreduce = DimReductionEngine()

reduce_result = dimreduce.reduce(cust_df, algorithm="pca", n_components=n_95)
print(
    f"  DimReductionEngine.reduce(pca, n_components={n_95}): "
    f"embedding shape=({len(reduce_result.transformed)}, "
    f"{reduce_result.n_components})  "
    f"recon_mse={reduce_result.reconstruction_error:.4f}"
)
if reduce_result.explained_variance_ratio is not None:
    cum = sum(reduce_result.explained_variance_ratio)
    print(f"  Explained variance retained: {cum:.2%}")
print()
print("  Every metric you computed by hand — explained variance ratio,")
print("  cumulative variance, reconstruction MSE — is a field on")
print("  DimReductionResult. The engine IS the destination.\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Derived PCA directly from X = U S V^T and verified against sklearn
  [x] Built a scree plot and applied three selection criteria
  [x] Read loadings to assign business meaning to principal components
  [x] Measured reconstruction error as a compression quality metric
  [x] Costed PCA as the right tool for nightly Shopee segmentation

  KEY INSIGHT: PCA is the only linear dim-reduction method with a true
  inverse transform. If your downstream job needs to EXPLAIN a customer
  in the original feature space, PCA is the answer — every other method
  in this exercise gives you points in a new space with no way back.

  Next: 02_kernel_pca.py lifts this into nonlinear territory via the
  kernel trick — same math, richer feature space.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
