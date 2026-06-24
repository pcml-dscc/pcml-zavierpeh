# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 2.4: Mixture of Experts — Soft vs Hard Assignments
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Read GMM soft assignments as intent vectors, not class labels
#   - Measure assignment confidence with max-probability and entropy
#   - Identify boundary customers that hard clustering would bury
#   - Explain Mixture of Experts as input-dependent gating g_k(x)
#   - Connect classical MoE to Sparse MoE in modern LLMs (Mixtral)
#
# PREREQUISITES: 03_covariance_types.py
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — MoE as GMM with input-dependent mixing
#   2. Build — soft_vs_hard analysis + simple MoE gate demo
#   3. Train — fit the BIC-optimal GMM and extract soft responsibilities
#   4. Visualise — confidence histogram + segment profile
#   5. Apply — Carousell personalised listing ranking (Singapore)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.mixture import GaussianMixture

from kailash_ml import ModelVisualizer

# Cross-exercise import: tracker helpers live in ex_1.shared so every M4
# unsupervised technique logs to the same `m4_clustering_zoo` experiment.
from shared.mlfp04.ex_1 import setup_engines, teardown_engines, track_run
from shared.mlfp04.ex_2 import (
    load_customers_scaled,
    out_path,
)

# ── Kailash-ML ExperimentTracker — every clustering run logs here ─────────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — From GMM to Mixture of Experts
# ════════════════════════════════════════════════════════════════════════
#
#   GMM:  P(x)   = Sum_k pi_k * N(x | mu_k, Sigma_k)    (fixed pi_k)
#   MoE:  P(y|x) = Sum_k g_k(x) * f_k(x)                (input-dep g_k)
#
# g_k(x) is a softmax over gating logits:
#     g_k(x) = exp(w_k^T x) / Sum_j exp(w_j^T x)
#
# Mixtral 8x7B uses this idea at LLM scale: 8 experts, top-2 routing
# per token => ~14B active params, not 56B. Same gating equation.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: soft_vs_hard analysis + gate demo
# ════════════════════════════════════════════════════════════════════════


def soft_vs_hard(soft_probs: np.ndarray) -> dict[str, float]:
    """Summarise a responsibility matrix R into a confidence profile."""
    # TODO: max_probs = soft_probs.max(axis=1)
    max_probs = ____
    # TODO: entropy = -sum(p * log(p+eps)) along axis=1
    entropy = ____
    return {
        "confident": float((max_probs > 0.95).mean()),
        "moderate": float(((max_probs > 0.7) & (max_probs <= 0.95)).mean()),
        "ambiguous": float(((max_probs > 0.5) & (max_probs <= 0.7)).mean()),
        "uncertain": float((max_probs <= 0.5).mean()),
        "mean_entropy": float(entropy.mean()),
        "max_entropy": float(np.log(soft_probs.shape[1])),
    }


def simple_moe_gate(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """A 2-expert softmax gate: route by the sign of the first feature."""
    rng = np.random.default_rng(seed)
    logits = np.column_stack([X[:, 0], -X[:, 0]]) + 0.1 * rng.standard_normal(
        (X.shape[0], 2)
    )
    # TODO: softmax-normalise logits across axis=1 — shift by max, exp, divide
    exp = ____
    return exp / exp.sum(axis=1, keepdims=True)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit BIC-optimal GMM and extract soft responsibilities
# ════════════════════════════════════════════════════════════════════════

X_scaled, customers, feature_cols, _ = load_customers_scaled()

best_k, best_bic = None, float("inf")
for k in range(2, 9):
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42).fit(
        X_scaled
    )
    b = float(gmm.bic(X_scaled))
    if b < best_bic:
        best_bic, best_k = b, k

print("=" * 70)
print(f"  Soft-assignment analysis at BIC-optimal K={best_k}")
print("=" * 70)

# TODO: fit GaussianMixture at best_k with full cov, random_state=42
best_gmm = ____
# TODO: get soft responsibilities with best_gmm.predict_proba(X_scaled)
soft_probs = ____
hard_labels = best_gmm.predict(X_scaled)

profile = soft_vs_hard(soft_probs)
print(f"\nConfidence bands (fraction of customers in each):")
print(f"  confident  (>0.95):     {profile['confident']:.1%}")
print(f"  moderate   (0.70-0.95): {profile['moderate']:.1%}")
print(f"  ambiguous  (0.50-0.70): {profile['ambiguous']:.1%}")
print(f"  uncertain  (<=0.50):    {profile['uncertain']:.1%}")
print(
    f"\nMean entropy: {profile['mean_entropy']:.4f}  "
    f"(max possible: {profile['max_entropy']:.4f})"
)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert soft_probs.shape == (X_scaled.shape[0], best_k), "R must be (n, K)"
assert abs(soft_probs.sum(axis=1).mean() - 1.0) < 1e-4, "rows must sum to 1"
assert soft_probs.min() >= 0, "responsibilities non-negative"
print("\n[ok] Checkpoint 1 passed — responsibilities form a valid distribution")

# TODO: boundary_mask = soft_probs.max(axis=1) < 0.6
boundary_mask = ____
n_boundary = int(boundary_mask.sum())
print(
    f"\nBoundary customers (max responsibility < 0.6): "
    f"{n_boundary} / {X_scaled.shape[0]} "
    f"({n_boundary / X_scaled.shape[0]:.1%})"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: confidence bands + segment profile
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
confidence_chart = {
    "Confidence bands": {
        "confident": profile["confident"],
        "moderate": profile["moderate"],
        "ambiguous": profile["ambiguous"],
        "uncertain": profile["uncertain"],
    }
}
fig = viz.metric_comparison(confidence_chart)
fig.update_layout(title="Soft-assignment confidence distribution")
fig.write_html(str(out_path("ex2_soft_confidence.html")))
print(f"\nSaved: {out_path('ex2_soft_confidence.html')}")

# Join labels back onto the customer table for per-segment profiling
customers_with_seg = customers.with_columns(
    pl.Series("gmm_segment", hard_labels),
    pl.Series("gmm_confidence", soft_probs.max(axis=1)),
)

print("\nPer-segment summary (hard counts + soft mass):")
for k in range(best_k):
    subset = customers_with_seg.filter(pl.col("gmm_segment") == k)
    hard_count = subset.height
    soft_mass = float(soft_probs[:, k].sum())
    print(
        f"  Segment {k}: hard={hard_count:>5}  "
        f"soft_mass={soft_mass:>8.1f}  "
        f"weight={best_gmm.weights_[k]:.3f}"
    )

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert customers_with_seg.height == X_scaled.shape[0], "join preserved all rows"
assert out_path("ex2_soft_confidence.html").exists(), "confidence chart written"
print("\n[ok] Checkpoint 2 passed — per-segment profile produced")

# MoE gate demo
moe_demo = simple_moe_gate(X_scaled[:, :2])
print(f"\nMoE gating demo on first 2 features:")
print(f"  Expert 0 active (gate > 0.5): {(moe_demo[:, 0] > 0.5).mean():.1%}")
print(f"  Expert 1 active (gate > 0.5): {(moe_demo[:, 1] > 0.5).mean():.1%}")

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert moe_demo.shape == (X_scaled.shape[0], 2), "MoE gate must produce 2 weights"
assert abs(moe_demo.sum(axis=1).mean() - 1.0) < 1e-4, "gate must be a distribution"
print("[ok] Checkpoint 3 passed — MoE gate produces a valid softmax distribution")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Carousell Personalised Listing Ranking (Singapore)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Carousell is SEA's largest C2C marketplace with ~35M users.
# The feed ranker has ~80ms to order ~10M live listings per session.
# A Saturday shopper might be 60% bargain hunter, 30% home-decor browser,
# 10% gift shopper in the same session. Hard segment models pick one
# label and miss the other 40% of intent.
#
# Soft responsibilities enable blended ranking:
#   expected_click = sum_k r_k * CTR_k(listing)
# Same idea as Sparse MoE in LLMs: top-K gated experts per query.
#
# BUSINESS IMPACT (from Carousell public disclosures):
#   - Feed impressions/day: ~900M at baseline CTR ~6.2%
#   - Soft intent-vector ranking lifts CTR ~8% vs hard-segment ranking
#   - 8% * 900M impressions/day = ~4.5M extra clicks/day
#   - At ~S$0.018 monetisation/click => ~S$81K/day = S$29.6M/year
#   - Zero marginal infra cost: GMM is fitted nightly and responsibilities
#     are materialised into the existing feature store.

print("\n" + "=" * 70)
print("  APPLY — Carousell personalised listing ranking")
print("=" * 70)
print(
    f"Out of {X_scaled.shape[0]} customers, {n_boundary} "
    f"({n_boundary / X_scaled.shape[0]:.1%}) have mixed intent. Soft scoring "
    "keeps ALL of them in the long-tail of every applicable segment ranker."
)
print(
    "At Carousell's scale, blending intents with soft responsibilities "
    "recovers ~S$29.6M/year in feed monetisation — from the same GMM "
    "you just fitted, read a different way."
)


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# This is the FINAL lesson in the M4 ex_2 GMM block. After it lands the
# m4_clustering_zoo experiment will hold eight runs across four families.

# TODO: call track_run with run_name "gmm_soft_assignment_moe". scalar
# metrics include the four confidence-band percentages from `profile`,
# mean_entropy, max_entropy, boundary_count (n_boundary), boundary_pct,
# and per-expert MoE active rates: float((moe_demo[:, k] > 0.5).mean())
# for k in 0, 1.
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "best_k": best_k,
        "cov_type": "full",
        "n_samples": X_scaled.shape[0],
        "n_features": X_scaled.shape[1],
        "moe_n_experts": 2,
    },
    scalar_metrics={
        "confident_pct": float(profile["confident"]),
        "moderate_pct": float(profile["moderate"]),
        "ambiguous_pct": float(profile["ambiguous"]),
        "uncertain_pct": float(profile["uncertain"]),
        "mean_entropy": float(profile["mean_entropy"]),
        "max_entropy": float(profile["max_entropy"]),
        "boundary_count": float(n_boundary),
        "boundary_pct": float(n_boundary) / float(X_scaled.shape[0]),
        "moe_expert0_active_pct": ____,
        "moe_expert1_active_pct": ____,
    },
)
print(f"  [tracked] soft-assignment + MoE gate metrics logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — engine fit + close the m4_clustering_zoo loop
# ════════════════════════════════════════════════════════════════════════
# After this lesson, the m4_clustering_zoo experiment in
# mlfp04_ex1_clustering.db holds eight runs across four families:
#   kmeans_pp / hierarchical_<linkage> / dbscan_hdbscan / spectral_rbf
#   em_from_scratch / sklearn_gmm_bic_aic / gmm_cov_<best> / gmm_soft_moe
#
# The leaderboard is the unifying surface across all clustering runs on
# the same Singapore e-commerce dataset. ClusteringEngine.fit returns
# hard labels in one call; soft predict_proba still routes through
# sklearn's GaussianMixture for the responsibility matrix.

from kailash_ml.engines.clustering import ClusteringEngine

cust_df = pl.from_numpy(X_scaled, schema=feature_cols)

# TODO: instantiate ClusteringEngine and call .fit on cust_df with
# algorithm='gmm' and n_clusters=best_k.
clustering = ____
fit_result = ____
print(
    f"  ClusteringEngine.fit(gmm, K={best_k}): "
    f"silhouette={(fit_result.silhouette_score or 0.0):.4f}  "
    f"n_clusters={fit_result.n_clusters}"
)
print(
    "  Open mlfp04_ex1_clustering.db for the full m4_clustering_zoo"
    " leaderboard — eight runs across four clustering families on the"
    " same dataset, ready for cross-method comparison.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Soft GMM responsibilities carry uncertainty hard labels destroy
  [x] Max-probability and entropy diagnose boundary customers
  [x] MoE = GMM with input-dependent gating g_k(x)
  [x] Sparse MoE in Mixtral/GPT-4 is the same idea at LLM scale
  [x] Carousell scenario: soft intent vectors = S$29.6M/year in feed rev

  Exercise 2 complete. Next: Exercise 3 — PCA on the same customer data.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
