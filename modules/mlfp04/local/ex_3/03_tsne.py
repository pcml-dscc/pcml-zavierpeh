# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 3.3: t-SNE for local structure
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Understand what t-SNE optimises (KL divergence of neighbourhoods)
#   - Tune the perplexity hyperparameter
#   - Recognise the three classic t-SNE pitfalls
#   - Know when t-SNE is a visualisation tool, not a feature extractor
#
# PREREQUISITES: 01_pca.py — we pre-reduce with PCA before t-SNE.
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — t-SNE as a neighbourhood-preserving map
#   2. Build — PCA pre-reduction + t-SNE at 4 perplexity values
#   3. Train — KL divergence + silhouette per perplexity
#   4. Visualise — 2D embedding scatter + perplexity comparison
#   5. Apply — Changi Airport passenger journey clustering
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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


# ════════════════════════════════════════════════════════════════════════
# THEORY — t-SNE in one paragraph
# ════════════════════════════════════════════════════════════════════════
# Build a high-dim Gaussian P over pairwise neighbours, a low-dim
# Student-t Q, then minimise KL(P || Q). Nearby points stay close in 2D.
# PERPLEXITY = effective # neighbours (5..50). PITFALLS:
#   A. Cluster SIZES are meaningless.
#   B. Inter-cluster distances are meaningless.
#   C. No out-of-sample transform — t-SNE is a picture, not a feature.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: PCA pre-reduction + subsample
# ════════════════════════════════════════════════════════════════════════

X, feature_cols, _ = load_customer_matrix()
n_samples, n_features = X.shape

# TODO: pre-reduce X with PCA to min(10, n_features) components. This is
# standard practice before t-SNE — it denoises without losing distances.
# Hint: PCA(n_components=..., random_state=42)
pca_pre = ____
X_pca = ____

# TODO: subsample 3,000 rows for t-SNE (Barnes-Hut scales but has a big
# constant factor). Use the shared helper.
idx = ____
X_tsne_input = X_pca[idx]
print(f"=== t-SNE input ===  n={X_tsne_input.shape[0]:,}, d={X_tsne_input.shape[1]}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: sweep perplexity
# ════════════════════════════════════════════════════════════════════════

tsne_results: dict[int, dict] = {}
perplexities = [5, 15, 30, 50]

print(f"\n=== t-SNE perplexity sweep ===")
print(f"{'perplexity':>12}{'KL div':>14}{'silhouette':>14}{'time (s)':>12}")
print("-" * 52)

for perplexity in perplexities:
    t0 = time.time()
    # TODO: build a TSNE with n_components=2, the current perplexity,
    # max_iter=1000, random_state=42, init='pca', learning_rate='auto'.
    tsne = ____
    # TODO: fit_transform the pre-reduced input.
    embedding = ____
    elapsed = time.time() - t0

    # TODO: score the 2D embedding with the shared silhouette helper.
    sil = ____
    tsne_results[perplexity] = {
        "embedding": embedding,
        "kl": float(tsne.kl_divergence_),
        "silhouette": sil,
        "time_s": elapsed,
    }
    print(f"{perplexity:>12}{tsne.kl_divergence_:>14.4f}{sil:>14.4f}{elapsed:>11.1f}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert len(tsne_results) == 4, "Must test 4 perplexity values"
for perp, res in tsne_results.items():
    assert res["embedding"].shape[1] == 2, "t-SNE must produce 2D output"
    assert res["kl"] > 0, "KL divergence must be positive"
print("\n[ok] Checkpoint 1 — 2D embeddings across 4 perplexity settings")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: perplexity comparison
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
fig_perp = viz.metric_comparison(
    {
        f"perplexity={p}": {"Silhouette": r["silhouette"], "KL": r["kl"]}
        for p, r in tsne_results.items()
    }
)
fig_perp.update_layout(title="t-SNE: perplexity vs KL divergence and silhouette")
perp_path = OUTPUT_DIR / "03_tsne_perplexity.html"
fig_perp.write_html(str(perp_path))
print(f"\nSaved: {perp_path}")

print("\nPerplexity guidance:")
print("  5  — micro-clusters (fragile)")
print("  15 — fine local structure")
print("  30 — balanced default")
print("  50 — smoother, fewer isolated clusters")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Changi Airport passenger journey clustering
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: CAG instruments T3 with 80+ touchpoints per passenger. The
# retail team wants MICRO-SEGMENTS inside the transit-passenger group:
# families, business travellers, premium cabins, budget lingerers. t-SNE
# captures LOCAL structure — perfect for a dashboard the retail planners
# can act on. Perplexity is tuned until the cluster granularity matches
# the 9 retail managers on duty. 7% F&B uplift on ~S$280M GMV = ~S$19.6M/yr.
# NEVER feed t-SNE coordinates into a downstream model — stochastic.

best_p, best_r = max(tsne_results.items(), key=lambda kv: kv[1]["silhouette"])
print(f"\n=== Changi-style micro-segment projection ===")
print(f"  Best perplexity : {best_p}")
print(f"  Silhouette      : {best_r['silhouette']:.4f}")
print(f"  KL divergence   : {best_r['kl']:.4f}")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Per-perplexity KL/silhouette/time scalars + parallel series go into the
# m4_dimreduction_zoo experiment.

perplexities_sorted = sorted(tsne_results.keys())

# TODO: call track_run with run_name f"tsne_perp_{best_p}". scalar_metrics
# headline is best_silhouette + best_kl, then |-merge three per-perplexity
# dicts (silhouette, kl, time_s) over tsne_results.items(). series_metrics:
# parallel silhouette + KL arrays in perplexities_sorted order.
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "algorithm": "tsne",
        "n_components": 2,
        "n_subsample": int(X_tsne_input.shape[0]),
        "pca_pre_components": int(X_tsne_input.shape[1]),
        "perplexities": ",".join(str(p) for p in perplexities_sorted),
        "best_perplexity": best_p,
    },
    scalar_metrics={
        "best_silhouette": float(best_r["silhouette"]),
        "best_kl": float(best_r["kl"]),
    }
    | ____
    | ____
    | ____,
    series_metrics={
        "sweep_silhouette": ____,
        "sweep_kl": ____,
    },
)
print(f"  [tracked] perplexity sweep logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — DimReductionEngine.reduce(algorithm='tsne')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1's DimReductionEngine wraps sklearn t-SNE under the same
# `reduce` surface that backed PCA in lesson 01. The engine handles the
# polars→numpy conversion, runs t-SNE, returns a DimReductionResult — one
# sync call.

import polars as pl

from kailash_ml.engines.dim_reduction import DimReductionEngine

# Engine takes raw features (not the PCA-pre-reduced matrix) and runs the
# whole pipeline; we slice down to the same subsample for fairness.
sub_idx = idx
cust_df = pl.from_numpy(X[sub_idx], schema=feature_cols)

# TODO: instantiate DimReductionEngine and call .reduce on cust_df with
# algorithm='tsne', n_components=2, perplexity=best_p.
dimreduce = ____
reduce_result = ____
print(
    f"  DimReductionEngine.reduce(tsne, perplexity={best_p}): "
    f"embedding shape=({len(reduce_result.transformed)}, "
    f"{reduce_result.n_components})  "
    f"kl={reduce_result.metrics.get('kl_divergence', float('nan')):.4f}"
)
print()
print("  Same t-SNE you swept by hand — wrapped under the engine surface")
print("  that backs pca / tsne / umap / nmf. The leaderboard now compares")
print("  this perplexity with the PCA baseline from lesson 01.\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Ran t-SNE at 4 perplexity values and measured KL + silhouette
  [x] Pre-reduced with PCA before t-SNE (standard practice)
  [x] Recognised the three pitfalls: cluster size, inter-cluster
      distance, no out-of-sample transform
  [x] Sized t-SNE for a Changi retail dashboard where the output is a
      visual, not a feature

  KEY INSIGHT: t-SNE is a PICTURE generator. When your deliverable is
  an insight for a human, t-SNE is brilliant. When it's a feature for
  another model, use PCA or UMAP.

  Next: 04_umap.py adds out-of-sample transform.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
