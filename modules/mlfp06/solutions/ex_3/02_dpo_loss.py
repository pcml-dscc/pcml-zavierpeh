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
    dpo_loss,
    show_beta_sensitivity,
    show_dpo_loss_curve_by_margin,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — RLHF to DPO in one page
# ════════════════════════════════════════════════════════════════════════
# RLHF objective (maximise reward, minimise KL from reference policy):
#   max_pi  E_x,y~pi [r(x,y)]  -  beta * KL(pi || pi_ref)
#
# Closed-form optimal policy:
#   pi*(y|x) = pi_ref(y|x) * exp(r(x,y) / beta) / Z(x)
#
# Bradley-Terry preference model:
#   P(y_w > y_l | x) = sigma(r(x,y_w) - r(x,y_l))
#
# Substitute pi* back into Bradley-Terry. The partition Z(x) cancels
# because it appears in both sides of the preference. What's left:
#   P(y_w > y_l | x) = sigma(
#       beta * ( log pi(y_w|x)/pi_ref(y_w|x)
#              - log pi(y_l|x)/pi_ref(y_l|x) )
#   )
#
# Negative log-likelihood of preferences = the DPO loss:
#   L_DPO = -E[ log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x)
#                                - log pi(y_l|x)/pi_ref(y_l|x))) ]
#
# KEY INSIGHT: the reward model is IMPLICIT in the policy. DPO bypasses
# reward-model training entirely — it directly optimises the policy to
# satisfy preferences. Cheaper, simpler, and empirically very strong.

print("=" * 70)
print("TASK 1: DPO derivation walkthrough")
print("=" * 70)
print(
    """
  RLHF objective:
    max_pi E[r(x,y)] - beta * KL(pi || pi_ref)

  -> Optimal policy:
    pi*(y|x) proportional to  pi_ref(y|x) * exp(r(x,y) / beta)

  -> Plug into Bradley-Terry. Z(x) cancels.

  -> DPO loss:
    L = -E[log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x)
                            - log pi(y_l|x)/pi_ref(y_l|x)))]
"""
)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Implement dpo_loss() (already in shared/mlfp06/ex_3.py)
#          Verify its behaviour on synthetic log-probabilities
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Verify DPO loss on synthetic log-probs")
print("=" * 70)

torch.manual_seed(42)
batch_size = 16

# Policy clearly prefers chosen (high log-prob) over rejected (low log-prob).
# We use a clear mean separation (+1.0 vs -1.0) with small noise (×0.3) so the
# demonstrated ordering (reversed prefs -> higher loss) is unambiguous and does
# NOT depend on the exact RNG stream — importing shared.mlfp06 advances the
# global torch RNG, so a subtle ±0.5 separation could flip the comparison.
policy_chosen = torch.randn(batch_size) * 0.3 + 1.0
policy_rejected = torch.randn(batch_size) * 0.3 - 1.0
ref_chosen = torch.randn(batch_size) * 0.3
ref_rejected = torch.randn(batch_size) * 0.3

loss_val = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
print(f"DPO loss (synthetic batch, beta=0.1): {loss_val.item():.4f}")

# Perfect alignment — very high chosen log-prob, very low rejected log-prob
perfect_chosen = torch.zeros(batch_size)
perfect_rejected = torch.full((batch_size,), -10.0)
loss_perfect = dpo_loss(
    perfect_chosen, perfect_rejected, ref_chosen, ref_rejected, beta=0.1
)
print(f"DPO loss (perfect alignment):         {loss_perfect.item():.4f}")

# Reversed preferences — loss should spike
loss_reversed = dpo_loss(
    policy_rejected, policy_chosen, ref_chosen, ref_rejected, beta=0.1
)
print(f"DPO loss (reversed preferences):      {loss_reversed.item():.4f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert loss_val.item() > 0, "DPO loss should be positive"
assert loss_perfect.item() < loss_val.item(), "Perfect alignment -> lower loss"
assert loss_reversed.item() > loss_val.item(), "Reversed prefs -> higher loss"
print("✓ Checkpoint 2 passed — DPO loss behaviour verified\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — DPO loss as a function of the policy preference margin
# ════════════════════════════════════════════════════════════════════════
# The "margin" is: (log pi(y_w)/pi_ref(y_w)) - (log pi(y_l)/pi_ref(y_l))
# DPO loss is monotonic in this margin — as the model learns to prefer
# chosen over rejected relative to reference, loss drops smoothly.

show_dpo_loss_curve_by_margin(beta=0.1)

# ── Checkpoint Visual ───────────────────────────────────────────────────
assert (OUTPUT_DIR / "ex3_dpo_loss_curve.png").exists()
print("✓ Visual checkpoint passed — loss-vs-margin curve saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Beta sensitivity sweep
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Beta sensitivity sweep")
print("=" * 70)

betas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
beta_losses = []
for b in betas:
    loss_b = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=b)
    beta_losses.append(loss_b.item())

beta_df = pl.DataFrame({"beta": betas, "dpo_loss": beta_losses})
print(beta_df)

show_beta_sensitivity(betas, beta_losses)

print(
    """
  Interpretation of beta:
    0.01 - 0.05 : Weak preference pressure. Stays close to SFT base.
                  Use when SFT is already good and you want minimal shift.
    0.1         : Standard default. Balanced for most tasks.
    0.2 - 0.5   : Strong alignment. Use for safety-critical (finance, medical).
                  Risk: over-refusal on benign variations.
    >= 1.0      : Very strong. Rarely the right call — over-refusal is near-certain.
"""
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert len(beta_losses) == len(betas)
assert (OUTPUT_DIR / "ex3_beta_sensitivity.png").exists()
print("✓ Checkpoint 4 passed — beta sensitivity sweep complete\n")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Pick beta for a MAS-regulated Singapore robo-advisor
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: A Singapore robo-advisor must comply with MAS
# Notice FAA-N16 on investment advice. The model must refuse unlicensed
# advice, refuse to promise returns, and always disclose risk.
#
# DECISION: what beta should you pick for DPO?
#   - Too low (0.01): model still hedges on risk warnings inconsistently
#     -> compliance risk, MAS action, S$1M+ fines possible
#   - Too high (1.0): model refuses benign questions like "what is a
#     stock?" -> user frustration, churn, loss of AUM
#
# RULE OF THUMB for regulated finance: start at beta=0.2, measure
# over-refusal rate on a held-out "benign" eval set, then tune.

print("=" * 70)
print("APPLICATION — Robo-advisor DPO beta selection")
print("=" * 70)

AUM_SGD = 450_000_000  # Assets under management
EXPECTED_MAS_FINE_LOW_BETA = 1_200_000  # Expected value of a single enforcement
AVG_ANNUAL_REVENUE_PER_USER = 180  # SGD per user per year
USERS_LOST_HIGH_BETA = 3500  # Churn from over-refusing benign queries

cost_low_beta = EXPECTED_MAS_FINE_LOW_BETA
cost_high_beta = USERS_LOST_HIGH_BETA * AVG_ANNUAL_REVENUE_PER_USER
cost_balanced = max(0, 0.25 * EXPECTED_MAS_FINE_LOW_BETA) + 0.2 * cost_high_beta

rows = [
    {
        "beta": 0.02,
        "approx_annual_cost_sgd": cost_low_beta,
        "risk": "MAS enforcement — inconsistent risk disclosures",
    },
    {
        "beta": 0.20,
        "approx_annual_cost_sgd": cost_balanced,
        "risk": "Balanced — target for MAS-regulated deployments",
    },
    {
        "beta": 1.00,
        "approx_annual_cost_sgd": cost_high_beta,
        "risk": "Over-refusal — churn from frustrated retail users",
    },
]
cost_df = pl.DataFrame(rows)
print(cost_df)

print(
    f"\nRecommended: beta = 0.20  "
    f"(estimated annual cost ~ S${cost_balanced:,.0f} vs worst-case "
    f"S${max(cost_low_beta, cost_high_beta):,.0f})"
)

# ── Checkpoint Application ──────────────────────────────────────────────
assert cost_df.height == 3
assert cost_balanced < max(cost_low_beta, cost_high_beta)
print("✓ Application checkpoint passed — beta selection justified\n")


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
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Derived DPO from RLHF + Bradley-Terry in one page
  [x] Implemented and verified the DPO loss with three sanity cases
      (typical, perfect alignment, reversed preferences)
  [x] Visualised the loss as a smooth function of preference margin
  [x] Swept beta from 0.01 -> 1.0 and interpreted the curve
  [x] Applied beta selection to a MAS-regulated robo-advisor — framed
      the trade-off as expected-cost in S$

  KEY INSIGHT: Beta is the alignment temperature. Low beta stays close
  to the SFT base; high beta bolts preference onto the policy at the
  cost of over-refusal. For regulated deployments, start at 0.2 and
  measure over-refusal before tightening further.

  Next: 03_dpo_training.py runs the full AlignmentPipeline on UltraFeedback
  and registers the resulting adapter.
"""
)
