# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 6.3: LIME Local Explanations + SHAP Waterfalls
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Use LIME to fit a local linear surrogate around a single prediction
#   - Compare LIME's local weights against SHAP for the same sample
#   - Explain the HIGHEST- and LOWEST-risk applications in the test set
#   - Build the "right to explanation" artifact required by PDPA
#   - Apply: adverse-action notices for declined Singapore loan applicants
#
# PREREQUISITES: 01_shap_global.py
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — how LIME's local linear surrogate works
#   2. Build — LimeTabularExplainer with graceful ImportError handling
#   3. Train — EXPLAIN extreme-risk individual cases
#   4. Visualise — LIME weights and SHAP waterfall side-by-side
#   5. Apply — PDPA adverse-action notices for declined applicants
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dotenv import load_dotenv

from shared.mlfp03.ex_6 import (
    build_shap_explainer,
    print_section,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — LIME fits a sparse linear surrogate on perturbations weighted
# by proximity to the target sample. Fast, model-agnostic, less stable
# than SHAP.
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the LIME explainer
# ════════════════════════════════════════════════════════════════════════

bundle = build_shap_explainer()
model = bundle["model"]
X_train = bundle["X_train"]
X_test = bundle["X_test"]
y_proba = bundle["y_proba"]
feature_names: list[str] = bundle["feature_names"]
shap_vals: np.ndarray = bundle["shap_vals"]

try:
    from lime.lime_tabular import LimeTabularExplainer

    # TODO: instantiate LimeTabularExplainer with training_data=X_train,
    # feature_names=feature_names, class_names=["no_default","default"],
    # mode="classification", discretize_continuous=True, random_state=42
    lime_explainer = ____
    HAS_LIME = True
except ImportError:
    lime_explainer = None
    HAS_LIME = False
    print("\n[warn] LIME not installed. Install with: uv pip install lime\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — pick the most interesting individuals to explain
# ════════════════════════════════════════════════════════════════════════

# TODO: sort indices by y_proba ascending (use np.argsort)
risk_order = ____
high_risk_idx = int(risk_order[-1])
low_risk_idx = int(risk_order[0])
borderline_idx = int(risk_order[len(risk_order) // 2])

print_section("Local Explanation Targets")
print(f"  Highest risk idx: {high_risk_idx}  P(default)={y_proba[high_risk_idx]:.4f}")
print(f"  Lowest  risk idx: {low_risk_idx}  P(default)={y_proba[low_risk_idx]:.4f}")
print(f"  Borderline  idx: {borderline_idx}  P(default)={y_proba[borderline_idx]:.4f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE local explanations
# ════════════════════════════════════════════════════════════════════════

if HAS_LIME:
    print_section("LIME Local Weights — HIGHEST-risk applicant", char="─")
    # TODO: call lime_explainer.explain_instance with X_test[high_risk_idx],
    # model.predict_proba, num_features=10, top_labels=1
    lime_exp = ____
    for feat_desc, weight in lime_exp.as_list():
        direction = "^risk" if weight > 0 else "v risk"
        print(f"  {feat_desc:<45} {weight:+.4f} ({direction})")

print_section("SHAP Waterfall — HIGHEST-risk applicant", char="─")
# TODO: sort (name, shap_value, feature_value) triples by abs(shap_value) desc
#       using the high_risk_idx row. Show top 10.
shap_sorted = ____
for name, sv, fv in shap_sorted[:10]:
    direction = "^risk" if sv > 0 else "v risk"
    print(f"  {name:<30} value={fv:>8.2f}  SHAP={sv:+.4f} ({direction})")


print_section("SHAP Waterfall — LOWEST-risk applicant", char="─")
for name, sv, fv in sorted(
    zip(feature_names, shap_vals[low_risk_idx], X_test[low_risk_idx]),
    key=lambda t: abs(t[1]),
    reverse=True,
)[:8]:
    direction = "^risk" if sv > 0 else "v risk"
    print(f"  {name:<30} value={fv:>8.2f}  SHAP={sv:+.4f} ({direction})")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    y_proba[high_risk_idx] > y_proba[low_risk_idx]
), "Task 4: highest-risk probability must exceed lowest-risk"
print("\n[ok] Checkpoint — local explanations rendered for three risk tiers\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: PDPA Adverse-Action Notices
# ════════════════════════════════════════════════════════════════════════
# PDPA + MAS require a SPECIFIC reason for every declined loan.
# BUSINESS IMPACT: automating adverse-action notices with SHAP+LIME
# reduces analyst review from S$675,000/month to S$20,250/month (~S$7.85M
# annual saving) against a S$32,000 implementation. Payback < 1 week.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print_section("WHAT YOU'VE MASTERED")
print(
    """
  [x] Built a LIME explainer with graceful fallback
  [x] Explained highest-, lowest-, borderline-risk applicants
  [x] Compared LIME vs SHAP waterfall decomposition
  [x] Designed an adverse-action notice template
  [x] Quantified PDPA compliance savings

  KEY INSIGHT: PDPA requires per-decision explanations, not just global
  feature rankings. SHAP is the ground truth, LIME is the plain-English
  layer.

  Next: 04_shap_interactions.py
"""
)
