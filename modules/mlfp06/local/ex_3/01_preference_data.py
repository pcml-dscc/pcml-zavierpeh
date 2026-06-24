# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 3.1: Preference Data — UltraFeedback Binarized
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - What a preference dataset is: (prompt, chosen, rejected) triples
#   - How to load UltraFeedback Binarized — the dataset behind Zephyr/Tulu/OpenChat
#   - Split preferences into train/eval for DPO training
#   - Visualise chosen-vs-rejected length distributions (verbosity bias preview)
#   - Apply to Singapore fintech customer-support alignment
#
# PREREQUISITES: Exercise 2 (LoRA + SFT). DPO consumes SFT-trained checkpoints.
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load UltraFeedback Binarized via the shared loader
#   2. Inspect a preference triple — understand chosen vs rejected
#   3. Split into train (90%) / eval (10%)
#   4. Visualise chosen vs rejected response lengths
#   5. Apply: use this dataset to align a customer-support agent at a
#      Singapore fintech
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl

from shared.mlfp06.ex_3 import (
    OUTPUT_DIR,
    load_ultrafeedback,
    show_preference_length_distribution,
    split_preferences,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why preference pairs, not labels?
# ════════════════════════════════════════════════════════════════════════
# SFT teaches a model to imitate "correct" responses. But "correct" is
# rarely binary — two answers can both be truthful yet differ in tone,
# helpfulness, safety, style. Preference data captures the difference.
# Analogy: you cannot rate every wine on a 0-100 scale, but you CAN
# reliably pick a winner between two glasses.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load UltraFeedback Binarized
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Load UltraFeedback Binarized")
print("=" * 70)

# TODO: Call the shared loader with n_samples=2000 and print the shape.
pref_data = ____
print(f"Shape: {pref_data.shape}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert "prompt" in pref_data.columns, "Task 1: need 'prompt' column"
assert "chosen" in pref_data.columns, "Task 1: need 'chosen' column"
assert "rejected" in pref_data.columns, "Task 1: need 'rejected' column"
assert pref_data.height > 0, "Task 1: dataset should not be empty"
print(f"✓ Checkpoint 1 passed — {pref_data.height} preference pairs loaded\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Inspect a preference triple
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Inspect a preference triple")
print("=" * 70)

# TODO: Pick the first row. Print prompt, chosen, and rejected (truncate to
#       ~280 chars each for readability).
sample_prompt = ____
sample_chosen = ____
sample_rejected = ____

print(f"PROMPT:\n  {sample_prompt[:280]}\n")
print(f"CHOSEN:\n  {sample_chosen[:280]}...\n")
print(f"REJECTED:\n  {sample_rejected[:280]}...\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train / eval split
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Train / eval split")
print("=" * 70)

# TODO: Use split_preferences() from the shared module with train_frac=0.9.
train_pref, eval_pref = ____
print(f"Train: {train_pref.height}")
print(f"Eval:  {eval_pref.height}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert train_pref.height + eval_pref.height == pref_data.height
assert train_pref.height > eval_pref.height
print("✓ Checkpoint 3 passed — train/eval split ready\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Chosen vs rejected response lengths
# ════════════════════════════════════════════════════════════════════════

# TODO: Call show_preference_length_distribution() with pref_data and a title.
____

chosen_lens = np.array([len(s) for s in pref_data["chosen"].to_list()])
rejected_lens = np.array([len(s) for s in pref_data["rejected"].to_list()])
longer_chosen_frac = float((chosen_lens > rejected_lens).mean())
print(f"\nChosen longer than rejected in {longer_chosen_frac:.0%} of pairs.")

assert (OUTPUT_DIR / "ex3_preference_lengths.png").exists()
print("✓ Visual checkpoint passed\n")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Customer-support alignment at a Singapore fintech (KaiPay)
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: A Singapore fintech runs a support chatbot. After
# SFT the bot is competent but occasionally rude, occasionally refuses
# reasonable refunds, and over-promises timelines. Collect preference
# pairs, apply DPO, measure escalation-rate drop.

print("=" * 70)
print("APPLICATION — KaiPay Singapore support-bot alignment")
print("=" * 70)

# TODO: Build a polars DataFrame with 4 rows of (prompt, chosen, rejected)
#       that represents KaiPay support scenarios. Aim for chosen responses
#       that cite PDPA / MAS rules where relevant. Keep rejected terse/rude.
kaipay_pairs = ____

print(f"KaiPay preference pairs: {kaipay_pairs.height}")

# TODO: Compute daily_saved_sgd and annual_saved_sgd given:
#   DAILY_SUPPORT_CONVERSATIONS = 18000
#   ESCALATION_RATE_BEFORE = 0.12
#   ESCALATION_RATE_AFTER = 0.07
#   HUMAN_AGENT_COST_SGD = 4.20
DAILY_SUPPORT_CONVERSATIONS = 18000
ESCALATION_RATE_BEFORE = 0.12
ESCALATION_RATE_AFTER = 0.07
HUMAN_AGENT_COST_SGD = 4.20

daily_saved_sgd = ____
annual_saved_sgd = ____

print(f"Daily saving: S${daily_saved_sgd:,.0f}")
print(f"Annual saving: S${annual_saved_sgd:,.0f}")

assert kaipay_pairs.height >= 4
assert {"prompt", "chosen", "rejected"}.issubset(set(kaipay_pairs.columns))
print("✓ Application checkpoint passed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Loaded UltraFeedback Binarized — real human preference pairs
  [x] Understood the (prompt, chosen, rejected) triple structure
  [x] Split into train/eval for DPO
  [x] Visualised chosen-vs-rejected length distribution
  [x] Applied preference data to KaiPay Singapore support alignment

  Next: 02_dpo_loss.py derives the DPO loss and implements it from scratch.
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
