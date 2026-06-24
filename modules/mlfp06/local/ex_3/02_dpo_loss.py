# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 3.2: The DPO Loss From Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Derive DPO from the RLHF objective (bypass the reward model)
#   - Implement the DPO loss in PyTorch and verify against known cases
#   - Understand the role of beta as the alignment temperature
#   - Visualise how the DPO loss responds to policy preference margin
#   - Run a beta sensitivity sweep and interpret the curve
#   - Apply to MAS-compliant model alignment in Singapore finance
#
# PREREQUISITES: 01_preference_data.py (preference triple intuition).
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Derivation walkthrough: RLHF -> Bradley-Terry -> closed-form DPO
#   2. Implement dpo_loss() and verify with synthetic log-probabilities
#   3. Visualise loss vs policy preference margin
#   4. Beta sensitivity sweep on a fixed batch
#   5. Apply: pick beta for a MAS-regulated Singapore robo-advisor
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl
import torch
import torch.nn.functional as F

from shared.mlfp06.ex_3 import (
    OUTPUT_DIR,
    show_beta_sensitivity,
    show_dpo_loss_curve_by_margin,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — RLHF to DPO in one page
# ════════════════════════════════════════════════════════════════════════
# RLHF:    max_pi E[r(x,y)] - beta * KL(pi || pi_ref)
# Optimal: pi*(y|x) proportional to pi_ref(y|x) * exp(r/beta)
# Bradley-Terry + substitution -> Z(x) cancels -> closed-form DPO:
#   L = -E[log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x)
#                          - log pi(y_l|x)/pi_ref(y_l|x)))]
# The reward model is IMPLICIT in the policy. That is the big idea.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Implement dpo_loss() in the student file
#          (The shared module has the canonical reference, but here you
#           write it yourself so you understand each term.)
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Implement and verify DPO loss")
print("=" * 70)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute the DPO loss from scratch."""
    # TODO: Compute the chosen log-ratio: policy_chosen - ref_chosen
    chosen_log_ratio = ____
    # TODO: Compute the rejected log-ratio: policy_rejected - ref_rejected
    rejected_log_ratio = ____
    # TODO: Compute the DPO logits: beta * (chosen_log_ratio - rejected_log_ratio)
    logits = ____
    # TODO: Return -F.logsigmoid(logits).mean()
    return ____


torch.manual_seed(42)
batch_size = 16

# Clear mean separation (+1.0 vs -1.0) with small noise (×0.3) so the
# reversed-prefs ordering is unambiguous regardless of the RNG stream.
policy_chosen = torch.randn(batch_size) * 0.3 + 1.0
policy_rejected = torch.randn(batch_size) * 0.3 - 1.0
ref_chosen = torch.randn(batch_size) * 0.3
ref_rejected = torch.randn(batch_size) * 0.3

loss_val = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
print(f"DPO loss (synthetic batch): {loss_val.item():.4f}")

# TODO: Compute the "perfect alignment" loss: chosen=0.0, rejected=-10.0
perfect_chosen = ____
perfect_rejected = ____
loss_perfect = dpo_loss(
    perfect_chosen, perfect_rejected, ref_chosen, ref_rejected, beta=0.1
)
print(f"DPO loss (perfect alignment): {loss_perfect.item():.4f}")

# TODO: Compute the "reversed preferences" loss by swapping policy_chosen
#       and policy_rejected in the call.
loss_reversed = ____
print(f"DPO loss (reversed preferences): {loss_reversed.item():.4f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert loss_val.item() > 0
assert loss_perfect.item() < loss_val.item()
assert loss_reversed.item() > loss_val.item()
print("✓ Checkpoint 2 passed — DPO loss behaviour verified\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — DPO loss vs policy preference margin
# ════════════════════════════════════════════════════════════════════════

# TODO: Call show_dpo_loss_curve_by_margin(beta=0.1) from the shared module.
____
assert (OUTPUT_DIR / "ex3_dpo_loss_curve.png").exists()
print("✓ Visual checkpoint passed — loss curve saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Beta sensitivity sweep
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Beta sensitivity sweep")
print("=" * 70)

betas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

# TODO: For each beta, compute dpo_loss() on the synthetic batch and
#       append to beta_losses as a float.
beta_losses = []
for b in betas:
    ____

beta_df = pl.DataFrame({"beta": betas, "dpo_loss": beta_losses})
print(beta_df)

# TODO: Call show_beta_sensitivity(betas, beta_losses)
____

assert len(beta_losses) == len(betas)
assert (OUTPUT_DIR / "ex3_beta_sensitivity.png").exists()
print("✓ Checkpoint 4 passed — beta sensitivity sweep complete\n")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Pick beta for a MAS-regulated Singapore robo-advisor
# ════════════════════════════════════════════════════════════════════════
# DECISION: what beta for a MAS Notice FAA-N16 compliant robo-advisor?
#   - Too low (0.01):  inconsistent risk disclosures -> MAS enforcement
#   - Too high (1.0):  refuses benign questions -> churn
# Rule of thumb: start at beta=0.2, measure over-refusal, then tune.

print("=" * 70)
print("APPLICATION — Robo-advisor DPO beta selection")
print("=" * 70)

EXPECTED_MAS_FINE_LOW_BETA = 1_200_000
AVG_ANNUAL_REVENUE_PER_USER = 180
USERS_LOST_HIGH_BETA = 3500

cost_low_beta = EXPECTED_MAS_FINE_LOW_BETA
cost_high_beta = USERS_LOST_HIGH_BETA * AVG_ANNUAL_REVENUE_PER_USER

# TODO: Build a 3-row polars DataFrame comparing beta=0.02, 0.20, 1.00
#       with columns: beta, approx_annual_cost_sgd, risk.
#       Recommend beta=0.20 as the balanced choice and print it.
cost_df = ____

print(cost_df)
print("\nRecommended: beta = 0.20 (balanced MAS exposure vs churn)")

assert cost_df.height == 3
print("✓ Application checkpoint passed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Derived DPO from RLHF + Bradley-Terry
  [x] Implemented and verified the DPO loss
  [x] Visualised loss vs preference margin
  [x] Swept beta from 0.01 -> 1.0
  [x] Applied beta selection to a MAS-regulated robo-advisor

  Next: 03_dpo_training.py runs the full AlignmentPipeline on UltraFeedback.
"""
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — six lenses before completion
# ══════════════════════════════════════════════════════════════════
# The LLM Observatory extends M5's Doctor's Bag for LLM/agent work.
# Six lenses:
#   1. Output        — is the generation coherent, factual, on-task?
#   2. Attention     — what does the model attend to internally?
#   3. Retrieval     — did we fetch the right context?  [RAG only]
#   4. Agent Trace   — what did the agent actually do?  [Agent only]
#   5. Alignment     — is it aligned with our intent?   [Fine-tune only]
#   6. Governance    — is it within policy?            [PACT only]
from shared.mlfp06.diagnostics import LLMObservatory

# Primary lens: Alignment (reward margin curve, win-rate, hacking scan).
# For DPO, we expect reward margin to climb then plateau. For GRPO, we
# expect the group-mean reward to rise while group-std collapses.
if False:  # scaffold — requires a completed DPO/GRPO training log
    obs = LLMObservatory(run_id="ex_3_dpo_run")
    # for step, row in enumerate(training_log):
    #     obs.alignment.log_training_step(step=step, reward_margin=row["margin"],
    #                                     win_rate=row["win"], kl=row["kl"])
    # obs.alignment.reward_hacking_scan(chosen_texts, rejected_texts)
    print("\n── LLM Observatory Report ──")
    findings = obs.report()

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Alignment  (HEALTHY): reward margin climbs 0.02 -> 0.71 over
#       1000 steps; win-rate vs reference = 0.63; no hacking flagged.
#   [✓] Output     (HEALTHY): judge score on preference pairs = 0.82
#   [?] Attention / Retrieval / Agent / Governance (n/a)
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [ALIGNMENT LENS] Margin 0.02 -> 0.71 is the classic DPO convergence
#     curve — monotonic climb through the first ~700 steps, then plateau
#     as the reference distribution stops providing new signal. A
#     HEALTHY win-rate sits in the 55-70% band; higher than 80% is a
#     reward-hacking red flag (the model found a degenerate shortcut
#     the preference dataset rewards).
#     >> Prescription: plateau means you can stop training; if margin
#        never climbed, check that `beta` isn't too large (KL cap too
#        tight lets the model sit on the base distribution).
#  [OUTPUT LENS] Judge score 0.82 on paired completions confirms the
#     preference signal generalises beyond the training set. If the
#     judge disagrees with the preference labels you'd see <0.5 here.
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
