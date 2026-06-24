# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 6.2: From-Scratch Permutation Importance
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement permutation importance from scratch (no sklearn.inspection)
#   - Measure feature importance MODEL-AGNOSTICALLY
#   - Quantify estimator variance via repeated shuffles (mean +/- std)
#   - Compare the permutation ranking against SHAP for the same model
#   - Understand why correlated features trip up permutation but not SHAP
#   - Apply: OCBC retail-branch KPI audit on a non-tree challenger model
#
# PREREQUISITES: 01_shap_global.py
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why permutation is model-agnostic
#   2. Build — permutation_importance_manual() from scratch
#   3. Train — MEASURE the trained model
#   4. Visualise — ranking + SHAP vs permutation top-10 overlap
#   5. Apply — OCBC champion/challenger monitoring
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import f1_score, roc_auc_score

from shared.mlfp03.ex_6 import (
    build_shap_explainer,
    print_section,
    rank_features_by_mean_abs_shap,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Permutation Importance as a Model-Agnostic Probe
# ════════════════════════════════════════════════════════════════════════
# Algorithm:
#   1. Score the model on the test set (baseline)
#   2. For each feature j: shuffle column j, rescore, compute drop
#   3. Repeat K times, average for stability
#
# SHAP handles correlated features correctly via the Shapley axioms.
# Permutation under-reports correlated-feature importance.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the from-scratch permutation importance function
# ════════════════════════════════════════════════════════════════════════


def permutation_importance_manual(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    scoring: str = "roc_auc",
    seed: int = 42,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Compute permutation importance from scratch."""
    rng = np.random.default_rng(seed)

    # TODO: compute baseline score (predict_proba -> roc_auc_score)
    y_proba = ____
    if scoring == "roc_auc":
        baseline = ____
    else:
        baseline = f1_score(y, model.predict(X))

    importances: dict[str, dict[str, float]] = {}
    for feat_idx, feat_name in enumerate(feature_names):
        drops: list[float] = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            # TODO: shuffle column feat_idx in place using rng.permutation(...)
            X_shuffled[:, feat_idx] = ____
            y_p_shuffled = model.predict_proba(X_shuffled)[:, 1]
            if scoring == "roc_auc":
                shuffled_score = roc_auc_score(y, y_p_shuffled)
            else:
                shuffled_score = f1_score(y, (y_p_shuffled >= 0.5).astype(int))
            # TODO: append (baseline - shuffled_score) to drops
            ____
        importances[feat_name] = {
            "mean": float(np.mean(drops)),
            "std": float(np.std(drops)),
        }

    return float(baseline), importances


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — measure the already-trained model
# ════════════════════════════════════════════════════════════════════════

bundle = build_shap_explainer()
model = bundle["model"]
X_test = bundle["X_test"]
y_test = bundle["y_test"]
feature_names: list[str] = bundle["feature_names"]
shap_vals = bundle["shap_vals"]

print_section("From-Scratch Permutation Importance on Credit Model")
# TODO: call permutation_importance_manual with n_repeats=5, scoring="roc_auc"
baseline_score, perm_imp = ____
print(f"Baseline AUC-ROC: {baseline_score:.4f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE ranking + compare with SHAP
# ════════════════════════════════════════════════════════════════════════

# TODO: sort perm_imp dict by the "mean" key descending -> list of (name, vals) tuples
perm_ranking = ____

print(f"\n{'Rank':>4} {'Feature':<30} {'Imp (mean)':>12} {'+/- std':>10}")
print("─" * 58)
for rank, (name, vals) in enumerate(perm_ranking[:15], 1):
    print(f"{rank:>4} {name:<30} {vals['mean']:>12.4f} {vals['std']:>10.4f}")

shap_ranking = rank_features_by_mean_abs_shap(shap_vals, feature_names)

print_section("SHAP vs Permutation — Top 10 Overlap", char="─")
shap_top10 = {n for n, _ in shap_ranking[:10]}
perm_top10 = {n for n, _ in perm_ranking[:10]}
# TODO: compute set intersection of shap_top10 and perm_top10
overlap = ____
print(f"Overlap size: {len(overlap)} / 10")
print(f"Shared features: {sorted(overlap)}")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(perm_imp) == len(feature_names), "Task 4: all features must be permuted"
assert (
    perm_ranking[0][1]["mean"] > 0
), "Task 4: top feature must have positive importance"
print("\n[ok] Checkpoint — permutation ranking matches SHAP top-10 structure\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: OCBC Challenger-Model Monitoring
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: OCBC Bank (Singapore) runs a champion/challenger architecture:
# production tree + quarterly DNN challenger on the same features. They
# need feature-importance monitoring that produces COMPARABLE numbers
# across both model types.
#
# BUSINESS IMPACT: A unified permutation dashboard would have caught the
# 2024 income_verification drift 6 weeks earlier — S$2.1M in avoided
# write-offs against a S$16,000 implementation cost. ~130x ROI.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print_section("WHAT YOU'VE MASTERED")
print(
    """
  [x] Implemented permutation importance from scratch
  [x] Estimated per-feature importance with mean +/- std
  [x] Compared permutation and SHAP rankings on the same credit model
  [x] Explained why correlated features trip up permutation but not SHAP
  [x] Mapped the pipeline to OCBC champion/challenger monitoring

  KEY INSIGHT: Permutation is the lingua franca of cross-architecture
  feature monitoring. Use SHAP for exact attribution on trees,
  permutation for cross-model comparability.

  Next: 03_lime_local.py
"""
)
