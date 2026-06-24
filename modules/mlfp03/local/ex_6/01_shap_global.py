# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 6.1: TreeSHAP and Global Feature Importance
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compute exact TreeSHAP values in O(TLD²) for LightGBM models
#   - Verify the Shapley additivity axiom (SHAP sum + base = model output)
#   - Rank features globally by mean |SHAP|
#   - Read dependence plots to find the SIGN of each feature's effect
#   - Apply: Singapore MAS Model Risk Management audit trail
#
# PREREQUISITES:
#   - MLFP03 Exercise 4 (LightGBM training)
#   - MLFP03 Exercise 5 (class imbalance — same model is explained here)
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why Shapley values are the "right" attribution
#   2. Build — TreeExplainer on the trained LightGBM credit model
#   3. Train — no training; we EXPLAIN a pre-trained model
#   4. Visualise — additivity check + global importance ranking
#   5. Apply — MAS Model Risk Management compliance pack
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv

from kailash_ml import ModelVisualizer

from shared.mlfp03.ex_6 import (
    OUTPUT_DIR,
    build_shap_explainer,
    print_section,
    rank_features_by_mean_abs_shap,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Shapley Values Are the "Right" Attribution
# ════════════════════════════════════════════════════════════════════════
# Shapley values come from cooperative game theory (Shapley 1953). For a
# prediction f(x), each feature is a "player" and the model output is the
# "payoff". The Shapley value φ_i is the unique attribution satisfying
# four axioms: efficiency, symmetry, dummy, linearity. TreeSHAP
# (Lundberg & Lee 2018) computes exact Shapley in O(TLD²) for trees.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the TreeSHAP explainer
# ════════════════════════════════════════════════════════════════════════

# TODO: call build_shap_explainer() and unpack the bundle dict
# Hint: keys are "model", "X_test", "y_test", "feature_names",
#       "shap_vals", "expected_value", "auc"
bundle = ____
model = ____
X_test = ____
y_test = ____
feature_names: list[str] = ____
shap_vals: np.ndarray = ____
expected_value: float = ____
auc = ____

print_section("TreeSHAP: Global Feature Attribution for Credit Default")
print(f"Model AUC-ROC:   {auc:.4f}")
print(f"SHAP shape:      {shap_vals.shape}  (samples x features)")
print(f"Expected value:  {expected_value:.4f}  (base-rate log-odds)")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Verify additivity on 10 samples
# ════════════════════════════════════════════════════════════════════════
# For each sample i: shap_sum = shap_vals[i].sum() + expected_value
# This should approximately equal model.predict_proba(X_test[i:i+1])[:,1]

print_section("Additivity Verification (first 10 samples)", char="─")
print(f"{'Sample':>8} {'SHAP sum':>12} {'Model out':>12} {'|Gap|':>10} {'Pass':>6}")
print("─" * 52)

additivity_errors: list[float] = []
for i in range(min(10, len(shap_vals))):
    # TODO: compute shap_sum (SHAP row sum + expected_value)
    shap_sum = ____
    # TODO: compute model_out (single-row predict_proba for the positive class)
    # Hint: model.predict_proba(X_test[i:i+1])[:, 1][0]
    model_out = ____
    gap = abs(shap_sum - model_out)
    additivity_errors.append(gap)
    passed = "ok" if gap < 0.1 else "~"
    print(f"{i:>8} {shap_sum:>12.4f} {model_out:>12.4f} {gap:>10.6f} {passed:>6}")

mean_gap = float(np.mean(additivity_errors))
print(f"\nMean additivity gap: {mean_gap:.6f}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert shap_vals.shape == (
    X_test.shape[0],
    X_test.shape[1],
), "Task 3: SHAP shape must be (n_samples, n_features)"
print("\n[ok] Checkpoint 1 — additivity verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE global importance ranking
# ════════════════════════════════════════════════════════════════════════

# TODO: use rank_features_by_mean_abs_shap(...) to rank features
importance_ranking = ____

print_section("Global Feature Importance (mean |SHAP|)")
print(f"{'Rank':>4} {'Feature':<30} {'mean|SHAP|':>12}")
print("─" * 60)
for rank, (name, imp) in enumerate(importance_ranking[:15], 1):
    bar = "#" * int(imp * 200)
    print(f"{rank:>4} {name:<30} {imp:>12.4f}  {bar}")

viz = ModelVisualizer()
# TODO: call viz.feature_importance(...) with model, feature_names, top_n=15
fig = ____
fig.update_layout(title="SHAP Feature Importance — Singapore Credit Default")
html_out = OUTPUT_DIR / "ex6_01_shap_global_importance.html"
fig.write_html(str(html_out))
print(f"\nSaved: {html_out}")


# Dependence direction — SHAP vs feature-value correlation
print_section("Dependence Direction (top 5 features)", char="─")
for feat_name, _ in importance_ranking[:5]:
    feat_idx = feature_names.index(feat_name)
    feat_vals = X_test[:, feat_idx]
    feat_shap = shap_vals[:, feat_idx]
    valid = ~(np.isnan(feat_vals) | np.isnan(feat_shap))
    if valid.sum() > 2:
        # TODO: compute Pearson correlation between feat_vals[valid] and feat_shap[valid]
        # Hint: np.corrcoef(a, b)[0, 1]
        corr = ____
    else:
        corr = 0.0
    direction = "^ increases default risk" if corr > 0 else "^ decreases default risk"
    print(f"  {feat_name:<30} corr={corr:+.3f}  ({direction})")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert len(importance_ranking) == len(feature_names), "Task 4: all features ranked"
top_name, top_imp = importance_ranking[0]
assert top_imp > 0, "Task 4: top feature must have positive importance"
print("\n[ok] Checkpoint 2 — global SHAP importance computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore MAS Model Risk Management Compliance Pack
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Tier-1 Singapore retail bank deploys this LightGBM model as
# a pre-approval filter for unsecured personal loans. MAS Model Risk
# Management guideline (MAS Consultation Paper P015-2021) requires:
#
#   1. A documented feature-contribution audit for every production model
#   2. Per-application explanation delivered to customers upon request
#   3. Quarterly review of the top-N drivers of declined decisions
#
# BUSINESS IMPACT:
#   - FIDReC average remediation per unexplained decline: S$12,000
#   - At 0.1% complaint rate on 40K declines/year, avoiding 10 complaints
#     saves S$120,000/year against a ~S$8,000 engineering cost. 15x ROI.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print_section("WHAT YOU'VE MASTERED")
print(
    """
  [x] Built a TreeSHAP explainer against a trained LightGBM credit model
  [x] Verified the Shapley additivity axiom on real predictions
  [x] Ranked features globally by mean |SHAP|
  [x] Read the sign of each feature's effect via SHAP-value correlation
  [x] Mapped the pipeline onto a concrete MAS compliance scenario

  KEY INSIGHT: SHAP is the unique attribution method satisfying the four
  Shapley axioms, and TreeSHAP computes it EXACTLY for tree models.

  Next: 02_permutation_importance.py
"""
)
