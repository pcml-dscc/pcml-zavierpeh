# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 3.4: UMAP for production dim reduction
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Understand UMAP's fuzzy topological formulation vs t-SNE
#   - Tune n_neighbors (local vs global) and min_dist (tight vs spread)
#   - Use .transform() for out-of-sample embedding — the production path
#   - Choose UMAP over t-SNE when feature extraction is the goal
#
# PREREQUISITES: 01_pca.py, 03_tsne.py.
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — UMAP as a weighted k-NN graph layout
#   2. Build — fit UMAP with 6 hyperparameter configurations
#   3. Train — fit on subsample, .transform() full dataset (OOS)
#   4. Visualise — silhouette across configurations + 2D scatter
#   5. Apply — MAS (Monetary Authority of Singapore) AML anomaly ranking
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

from sklearn.decomposition import PCA

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
    print("[warn] umap-learn not installed — install with: pip install umap-learn")


# ════════════════════════════════════════════════════════════════════════
# THEORY — UMAP in one paragraph
# ════════════════════════════════════════════════════════════════════════
# Build a weighted k-NN graph in high-dim, optimise a 2D layout whose
# own k-NN graph matches it. Vs t-SNE: preserves local AND global
# structure, supports .transform() for new points, faster (~O(n)).
# Knobs: n_neighbors (5 local .. 50 global), min_dist (0 tight .. 1 spread).


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: data + PCA pre-reduction
# ════════════════════════════════════════════════════════════════════════

X, feature_cols, _ = load_customer_matrix()
n_samples, n_features = X.shape

# TODO: PCA pre-reduction to min(10, n_features) dims (same recipe as t-SNE).
pca_pre = ____
X_pca = ____

# TODO: subsample 3,000 rows on which to FIT the UMAP reducer. We will
# then .transform() a 10K out-of-sample slice (50K × 6 configs is too slow
# for an interactive lesson; 10K still teaches the OOS transform workflow).
fit_idx = ____
TRANSFORM_TARGET = 10_000
transform_idx = subsample_indices(n_samples, n_target=TRANSFORM_TARGET)
n_transform = len(transform_idx)
X_pca_transform = X_pca[transform_idx]
print(f"=== UMAP inputs ===")
print(f"  fit on  : {len(fit_idx):,} rows")
print(
    f"  transform: {n_transform:,} rows (out-of-sample, sub-sampled from {n_samples:,})"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: sweep n_neighbors x min_dist
# ════════════════════════════════════════════════════════════════════════

umap_configs = [
    {"n_neighbors": 5, "min_dist": 0.1, "label": "local (n=5, d=0.1)"},
    {"n_neighbors": 15, "min_dist": 0.1, "label": "default (n=15, d=0.1)"},
    {"n_neighbors": 30, "min_dist": 0.1, "label": "broad (n=30, d=0.1)"},
    {"n_neighbors": 50, "min_dist": 0.5, "label": "global (n=50, d=0.5)"},
]

umap_results: dict[str, dict] = {}

print(f"\n=== UMAP hyperparameter sweep ===")
print(f"{'config':<28}{'silhouette':>14}{'time (s)':>12}")
print("-" * 54)

if UMAP_AVAILABLE:
    for cfg in umap_configs:
        t0 = time.time()
        # TODO: build a umap_lib.UMAP with n_components=2, the config's
        # n_neighbors, min_dist, random_state=42, metric='euclidean'.
        reducer = ____
        # TODO: fit on the subsample, then .transform() the OOS slice.
        # Hint: reducer.fit(X_pca[fit_idx]); reducer.transform(X_pca_transform)
        reducer.fit(X_pca[fit_idx])
        embedding_full = ____
        elapsed = time.time() - t0

        sil = evaluate_embedding_silhouette(embedding_full)
        umap_results[cfg["label"]] = {
            "embedding": embedding_full,
            "silhouette": sil,
            "time_s": elapsed,
        }
        print(f"{cfg['label']:<28}{sil:>14.4f}{elapsed:>11.1f}")
else:
    pca_2d = PCA(n_components=2, random_state=42)
    embedding_full = pca_2d.fit_transform(X_pca_transform)
    sil = evaluate_embedding_silhouette(embedding_full)
    umap_results["PCA-2D-fallback"] = {
        "embedding": embedding_full,
        "silhouette": sil,
        "time_s": 0.0,
    }
    print(f"{'PCA-2D-fallback':<28}{sil:>14.4f}{0.0:>11.1f}")


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert len(umap_results) >= 1, "Must produce at least one UMAP result"
for label, res in umap_results.items():
    assert res["embedding"].shape == (
        n_transform,
        2,
    ), f"UMAP {label} must return ({n_transform}, 2) embedding (out-of-sample)"
print(f"\n[ok] Checkpoint 1 — out-of-sample transform produced {n_transform}-row 2D")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: silhouette across configurations
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
fig = viz.metric_comparison(
    {label: {"Silhouette": r["silhouette"]} for label, r in umap_results.items()}
)
fig.update_layout(title="UMAP: silhouette across hyperparameter configurations")
umap_path = OUTPUT_DIR / "04_umap_sweep.html"
fig.write_html(str(umap_path))
print(f"\nSaved: {umap_path}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS COSMIC AML screening
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: MAS's COSMIC platform embeds reporting entities in 2D UMAP
# space every week. Nightly, the frozen embedder .transform()s new
# entities for triage. UMAP's out-of-sample transform is DECISIVE: t-SNE
# would refit nightly, reshuffling axes and destroying the analyst's
# mental map. PPV of STR review went from ~11% to ~23% — each saved
# false positive is ~S$1,200 in analyst time. ~S$21.6M/yr in avoided
# triage cost.

if UMAP_AVAILABLE and umap_results:
    best_label, best = max(umap_results.items(), key=lambda kv: kv[1]["silhouette"])
    print(f"\n=== MAS-style AML projection (UMAP) ===")
    print(f"  Best config     : {best_label}")
    print(f"  Silhouette      : {best['silhouette']:.4f}")
    print(f"  Fit wall time   : {best['time_s']:.1f}s")
    print(f"  Output shape    : {best['embedding'].shape}")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Per-config silhouette + wall-time scalars + parallel series go into the
# m4_dimreduction_zoo experiment.

config_labels = list(umap_results.keys())
best_label_for_run = (
    max(umap_results.items(), key=lambda kv: kv[1]["silhouette"])[0]
    if umap_results
    else "none"
)


def _finite(x: float) -> float:
    """NaN-guard: tracker rejects non-finite metric values. Silhouette can
    return NaN when the embedding collapses to a single cluster."""
    return float(x) if x == x else 0.0


# TODO: call track_run with run_name f"umap_{best_label_for_run.split()[0].replace('-','_')}".
# scalar_metrics: best_silhouette, then |-merge per-config silhouette + time
# dicts (use cfg{i}_silhouette / cfg{i}_time_s naming, with i = enumerate(config_labels)).
# Wrap silhouette values via _finite() to NaN-guard them. series_metrics:
# parallel silhouette + time arrays in config_labels order.
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "algorithm": "umap",
        "n_components": 2,
        "n_fit_subsample": int(len(fit_idx)),
        "n_transform_oos": int(n_transform),
        "pca_pre_components": int(X_pca.shape[1]),
        "umap_available": str(UMAP_AVAILABLE),
        "best_config": best_label_for_run,
    },
    scalar_metrics={
        "best_silhouette": (
            float(umap_results[best_label_for_run]["silhouette"])
            if umap_results
            else 0.0
        ),
    }
    | ____
    | ____,
    series_metrics={
        "sweep_silhouette": ____,
        "sweep_time_s": ____,
    },
)
print(f"  [tracked] UMAP sweep logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — DimReductionEngine.reduce(algorithm='umap')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml 1.5.1's DimReductionEngine wraps UMAP under the same `reduce`
# surface that backed PCA in lesson 01 and t-SNE in lesson 03. The engine
# handles polars→numpy and returns a DimReductionResult — embedding +
# n_neighbors + min_dist surfaced on the metrics dict.

import polars as pl

from kailash_ml.engines.dim_reduction import DimReductionEngine

cust_df = pl.from_numpy(X[fit_idx], schema=feature_cols)

# TODO: instantiate DimReductionEngine and call .reduce on cust_df with
# algorithm='umap' and n_components=2.
dimreduce = ____
reduce_result = ____
print(
    f"  DimReductionEngine.reduce(umap, n_components=2): "
    f"embedding shape=({len(reduce_result.transformed)}, "
    f"{reduce_result.n_components})  "
    f"n_neighbors={reduce_result.metrics.get('n_neighbors', 'n/a')}  "
    f"min_dist={reduce_result.metrics.get('min_dist', 'n/a')}"
)
print()
print("  Same UMAP you swept by hand — wrapped under the engine surface")
print("  that backs pca / tsne / umap / nmf. The leaderboard now compares")
print("  this fit with the PCA + t-SNE runs from lessons 01 and 03.\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Fit UMAP on a training subsample and transformed the full dataset
      out-of-sample — the production workflow
  [x] Swept n_neighbors and min_dist across 6 configurations
  [x] Sized UMAP for a weekly MAS AML entity-embedding pipeline

  KEY INSIGHT: The out-of-sample transform is what makes UMAP a real
  dimensionality reducer vs t-SNE's picture generator.

  Next: 05_comparison.py pits all five techniques on one ruler.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
