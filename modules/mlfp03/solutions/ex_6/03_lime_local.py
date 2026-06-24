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
#   - Explain the HIGHEST-risk and LOWEST-risk applications in the test set
#   - Build the "right to explanation" artifact required by PDPA
#   - Apply: adverse-action notices for declined Singapore loan applicants
#
# PREREQUISITES: 01_shap_global.py (same model, same SHAP bundle).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — how LIME's local linear surrogate works
#   2. Build — LimeTabularExplainer with graceful ImportError handling
#   3. Train — no training; EXPLAIN extreme-risk individual cases
#   4. Visualise — LIME weights and SHAP waterfall side-by-side
#   5. Apply — PDPA adverse-action notices for declined applicants
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

from shared.mlfp03.ex_6 import (
    OUTPUT_DIR,
    build_shap_explainer,
    print_section,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — LIME's Local Linear Surrogate
# ════════════════════════════════════════════════════════════════════════
# LIME (Ribeiro, Singh, Guestrin, 2016) — "Local Interpretable
# Model-agnostic Explanations":
#
# For a single prediction f(x):
#   1. Sample perturbed rows around x (Gaussian for continuous,
#      discrete bin flips for categorical)
#   2. Score each perturbation with the original model f
#   3. Weight perturbations by proximity to x (exponential kernel)
#   4. Fit a SPARSE LINEAR model on the weighted samples
#   5. The linear coefficients ARE the local feature importances
#
# SHAP vs LIME at a glance:
#   SHAP: axioms + exact for trees + consistent across samples
#   LIME: fast + model-agnostic + easier to explain to non-statisticians
#
# Production rule:
#   Tree model   -> TreeSHAP (exact)
#   Black-box    -> LIME or KernelSHAP (approximate)
#   Mixed stack  -> Both; SHAP as ground truth, LIME as sanity check


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

    lime_explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["no_default", "default"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    HAS_LIME = True
except ImportError:
    lime_explainer = None
    HAS_LIME = False
    print(
        "\n[warn] LIME is not installed in this environment.\n"
        "       Install with: uv pip install lime\n"
        "       The SHAP comparison below still runs.\n"
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — "TRAIN" = pick the most interesting individuals to explain
# ════════════════════════════════════════════════════════════════════════

risk_order = np.argsort(y_proba)
high_risk_idx = int(risk_order[-1])
low_risk_idx = int(risk_order[0])
borderline_idx = int(risk_order[len(risk_order) // 2])

print_section("Local Explanation Targets")
print(
    f"  Highest risk sample idx: {high_risk_idx}  P(default)={y_proba[high_risk_idx]:.4f}"
)
print(
    f"  Lowest  risk sample idx: {low_risk_idx}  P(default)={y_proba[low_risk_idx]:.4f}"
)
print(
    f"  Borderline      idx:     {borderline_idx}  P(default)={y_proba[borderline_idx]:.4f}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE local explanations (LIME + SHAP side-by-side)
# ════════════════════════════════════════════════════════════════════════

if HAS_LIME:
    print_section("LIME Local Weights — HIGHEST-risk applicant", char="─")
    lime_exp = lime_explainer.explain_instance(
        X_test[high_risk_idx],
        model.predict_proba,
        num_features=10,
        top_labels=1,
    )
    for feat_desc, weight in lime_exp.as_list():
        direction = "^risk" if weight > 0 else "v risk"
        print(f"  {feat_desc:<45} {weight:+.4f} ({direction})")

    print_section("SHAP Waterfall — same HIGHEST-risk applicant", char="─")
    sample_shap = shap_vals[high_risk_idx]
    shap_sorted = sorted(
        zip(feature_names, sample_shap, X_test[high_risk_idx]),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:10]
    for name, sv, fv in shap_sorted:
        direction = "^risk" if sv > 0 else "v risk"
        print(f"  {name:<30} value={fv:>8.2f}  SHAP={sv:+.4f} ({direction})")
else:
    print_section("SHAP Waterfall — HIGHEST-risk applicant", char="─")
    sample_shap = shap_vals[high_risk_idx]
    shap_sorted = sorted(
        zip(feature_names, sample_shap, X_test[high_risk_idx]),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:10]
    for name, sv, fv in shap_sorted:
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


print_section("SHAP Waterfall — BORDERLINE applicant", char="─")
for name, sv, fv in sorted(
    zip(feature_names, shap_vals[borderline_idx], X_test[borderline_idx]),
    key=lambda t: abs(t[1]),
    reverse=True,
)[:8]:
    direction = "^risk" if sv > 0 else "v risk"
    print(f"  {name:<30} value={fv:>8.2f}  SHAP={sv:+.4f} ({direction})")

# ── Visual: LIME / SHAP bar chart for highest-risk applicant ─────────────
sample_shap_hr = shap_vals[high_risk_idx]
shap_sorted_hr = sorted(
    zip(feature_names, sample_shap_hr),
    key=lambda t: abs(t[1]),
    reverse=True,
)[:10]
fig = go.Figure()
fig.add_trace(
    go.Bar(
        y=[n for n, _ in reversed(shap_sorted_hr)],
        x=[v for _, v in reversed(shap_sorted_hr)],
        orientation="h",
        marker_color=[
            "#ef4444" if v > 0 else "#3b82f6" for _, v in reversed(shap_sorted_hr)
        ],
        name="SHAP value",
    )
)
fig.update_layout(
    title=f"SHAP Waterfall — Highest-Risk Applicant (P(default)={y_proba[high_risk_idx]:.4f})",
    xaxis_title="SHAP value (red = increases risk, blue = decreases risk)",
    yaxis_title="Feature",
    height=max(400, len(shap_sorted_hr) * 35),
)
viz_path = OUTPUT_DIR / "ex6_03_lime_shap_high_risk.html"
fig.write_html(str(viz_path))
print(f"\n  Saved: {viz_path}")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert (
    y_proba[high_risk_idx] > y_proba[low_risk_idx]
), "Task 4: highest-risk probability must exceed lowest-risk"
# INTERPRETATION: The three waterfalls (high, low, borderline) are the
# concrete artifact a regulator wants to see: each individual decision
# decomposed into per-feature contributions.
print("\n[ok] Checkpoint — local explanations rendered for three risk tiers\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: PDPA Adverse-Action Notices for Declined Applicants
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Singapore's Personal Data Protection Act (PDPA) and the MAS
# Notice on Credit Decisions require banks to provide a SPECIFIC reason
# — not a generic "did not meet our criteria" — for every declined loan
# application. The explanation must name the top drivers of the decline
# AND must be defensible under audit.
#
# The adverse-action notice template:
#
#     "Dear [applicant], your application for loan #[id] was declined.
#      The top three factors contributing to this decision were:
#
#        1. [feature_1] — your value of [v1] compared to our acceptance
#           range of [range_1]
#        2. [feature_2] — ...
#        3. [feature_3] — ...
#
#      You may contest this decision by contacting our FIDReC liaison."
#
# Why LIME+SHAP together are the right tool:
#   - SHAP provides the LEGALLY DEFENSIBLE decomposition (exact, axiomatic)
#   - LIME provides the CUSTOMER-READABLE linear summary (easier to
#     translate into "your income was too low" narrative)
#   - The two must AGREE on the top-3 drivers before a notice is sent;
#     disagreement flags the application for human review
#
# BUSINESS IMPACT:
#   - ~15,000 declines/month across the bank; PDPA compliance is mandatory
#   - Manual analyst review of declined applications: S$45/application
#     (15,000 * S$45 = S$675,000/month)
#   - SHAP+LIME automated notices reduce analyst review to the 3% of cases
#     where the two methods disagree: 15,000 * 0.03 = 450 manual reviews,
#     costing 450 * S$45 = S$20,250/month
#   - Monthly saving: S$654,750. Annual saving: S$7.85M.
#   - Implementation cost: 4 engineer-weeks (S$32,000). Payback: <1 week.
#
# LIMITATION: LIME's random perturbation sampling makes it unstable —
# running it twice on the same sample can give different top-3 features.
# That's why SHAP is the ground truth and LIME is the plain-English layer.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print_section("WHAT YOU'VE MASTERED")
print(
    """
  [x] Built a LIME explainer (with graceful fallback if not installed)
  [x] Explained the highest-, lowest-, and borderline-risk applicants
  [x] Compared LIME local weights against SHAP waterfall decomposition
  [x] Designed an adverse-action notice template backed by both methods
  [x] Quantified the PDPA compliance cost savings in S$

  KEY INSIGHT: Global explanations tell regulators how the MODEL works;
  local explanations tell individual customers why THEIR application
  was declined. PDPA requires the latter — one explanation per decision.

  Next: 04_shap_interactions.py — move beyond single-feature effects and
  uncover which FEATURE PAIRS the model uses together.
"""
)
