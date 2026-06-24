# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 6.4: SHAP Interaction Effects
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compute pairwise SHAP interaction values via TreeExplainer
#   - Distinguish main effects (diagonal) from interactions (off-diagonal)
#   - Rank feature pairs by mean |interaction|
#   - Understand why interactions are what makes trees beat linear models
#   - Apply: UOB SME loan cross-feature risk factors
#
# PREREQUISITES: 01_shap_global.py
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — main effects vs interaction effects
#   2. Build — shap_interaction_values on a 500-row sample
#   3. Train — MEASURE the trained model's interactions
#   4. Visualise — top-10 interaction table
#   5. Apply — UOB SME cross-feature audit
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
# THEORY — TreeExplainer.shap_interaction_values returns a tensor of
# shape (n_samples, n_features, n_features). Diagonal = main effects,
# off-diagonal = pairwise interactions. A linear model has zero
# off-diagonal mass.
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the interaction explainer (sample for speed)
# ════════════════════════════════════════════════════════════════════════

bundle = build_shap_explainer()
explainer = bundle["explainer"]
X_test = bundle["X_test"]
feature_names: list[str] = bundle["feature_names"]

sample_size = min(500, X_test.shape[0])
X_sample = X_test[:sample_size]

print_section("SHAP Interaction Values")
print(
    f"Computing interaction tensor for {sample_size} samples x "
    f"{len(feature_names)} features ..."
)

# TODO: call explainer.shap_interaction_values(X_sample)
# Hint: if the return is a list (binary classifier), take index [1]
shap_interaction = ____
if isinstance(shap_interaction, list):
    shap_interaction = shap_interaction[1]

print(f"Interaction tensor shape: {shap_interaction.shape}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — MEASURE interactions
# ════════════════════════════════════════════════════════════════════════

n_features = len(feature_names)
interaction_strengths: list[tuple[str, str, float]] = []
for i in range(n_features):
    for j in range(i + 1, n_features):
        # TODO: compute mean |interaction| for feature pair (i, j)
        # Hint: np.abs(shap_interaction[:, i, j]).mean()
        strength = ____
        interaction_strengths.append((feature_names[i], feature_names[j], strength))

# TODO: sort interaction_strengths by strength descending (in-place .sort)
____

# TODO: compute per-feature main effects as mean |diagonal|
# Hint: np.abs(np.diagonal(shap_interaction, axis1=1, axis2=2)).mean(axis=0)
main_effects = ____
main_effects_ranked = sorted(
    zip(feature_names, main_effects), key=lambda t: t[1], reverse=True
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE top interactions + main effects
# ════════════════════════════════════════════════════════════════════════

print_section("Top 10 Main Effects (diagonal)")
print(f"{'Rank':>4} {'Feature':<30} {'Main |SHAP|':>14}")
print("─" * 52)
for rank, (name, val) in enumerate(main_effects_ranked[:10], 1):
    print(f"{rank:>4} {name:<30} {val:>14.4f}")


print_section("Top 10 Pairwise Interactions (off-diagonal)")
print(f"{'Rank':>4} {'Feature 1':<25} {'Feature 2':<25} {'Strength':>12}")
print("─" * 70)
for rank, (f1, f2, strength) in enumerate(interaction_strengths[:10], 1):
    print(f"{rank:>4} {f1:<25} {f2:<25} {strength:>12.4f}")

total_main = float(main_effects.sum())
total_interaction = float(sum(s for _, _, s in interaction_strengths))
nonlinear_share = total_interaction / (total_main + total_interaction)
print(f"\nMain-effect mass:     {total_main:.4f}")
print(f"Interaction mass:     {total_interaction:.4f}")
print(f"Non-linear share:     {nonlinear_share:.1%}")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(interaction_strengths) > 0, "Task 4: interaction list must be non-empty"
assert (
    interaction_strengths[0][2] >= 0
), "Task 4: interaction strength must be non-negative"
print("\n[ok] Checkpoint — interaction tensor computed and ranked\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: UOB SME Cross-Feature Risk Factor Audit
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: UOB underwrites ~8,000 SME working-capital loans/year.
# Hypothesis: INCOME x TENURE is a stronger signal than either alone.
#
# BUSINESS IMPACT: 1pp underwriting improvement = ~S$4.8M/year on an
# S$480M book. The 2023 SHAP interaction audit surfaced an (age x DSR)
# interaction that closed a DIR gap from 0.71 to 0.89.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print_section("WHAT YOU'VE MASTERED")
print(
    """
  [x] Computed the SHAP interaction tensor on a 500-sample slice
  [x] Separated main effects (diagonal) from interactions (off-diagonal)
  [x] Ranked feature PAIRS by mean |interaction|
  [x] Quantified non-linear share of model's predictive power
  [x] Mapped the audit to UOB SME underwriting

  KEY INSIGHT: If non-linear share is tiny, the linear challenger is
  the better production choice. Interactions are what justify trees.

  Next: 05_fairness_audit.py
"""
)
