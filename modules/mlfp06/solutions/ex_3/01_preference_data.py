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
# DATASET: UltraFeedback Binarized (trl-lib/ultrafeedback_binarized)
#   Real human-curated preference pairs. 2K subsample.
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
# SFT teaches a model to imitate "correct" responses via cross-entropy on
# instruction-response pairs. But "correct" is rarely binary — two answers
# can both be truthful yet differ in tone, helpfulness, safety, and style.
#
# Preference data captures this: given a prompt, we present TWO responses
# and mark which one humans prefer. The model learns to shift probability
# mass toward preferred responses WITHOUT needing an absolute reward.
#
# Analogy: you cannot rate every wine on a 0-100 scale consistently, but
# you CAN reliably say "I prefer this glass to that one". Pairwise
# preference is the cheapest reliable human signal.

print("=" * 70)
print("TASK 1: Load UltraFeedback Binarized")
print("=" * 70)

pref_data = load_ultrafeedback(n_samples=2000)
print(f"Shape: {pref_data.shape}")
print(f"Columns: {pref_data.columns}")

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

sample_prompt = pref_data["prompt"][0]
sample_chosen = pref_data["chosen"][0]
sample_rejected = pref_data["rejected"][0]

print(f"PROMPT:\n  {sample_prompt[:280]}\n")
print(f"CHOSEN (human-preferred):\n  {sample_chosen[:280]}...\n")
print(f"REJECTED (less preferred):\n  {sample_rejected[:280]}...\n")

# INTERPRETATION: chosen is NOT necessarily longer, more formal, or more
# detailed. It is preferred for some combination of helpfulness, accuracy,
# tone, and safety. DPO will learn this mix implicitly.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train / eval split
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Train / eval split")
print("=" * 70)

train_pref, eval_pref = split_preferences(pref_data, train_frac=0.9)
print(f"Train: {train_pref.height}")
print(f"Eval:  {eval_pref.height}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert train_pref.height + eval_pref.height == pref_data.height
assert train_pref.height > eval_pref.height, "Train should be larger than eval"
print("✓ Checkpoint 3 passed — train/eval split ready\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Chosen vs rejected response lengths
# ════════════════════════════════════════════════════════════════════════

show_preference_length_distribution(
    pref_data, title="UltraFeedback — Chosen vs Rejected response length"
)

chosen_lens = np.array([len(s) for s in pref_data["chosen"].to_list()])
rejected_lens = np.array([len(s) for s in pref_data["rejected"].to_list()])
longer_chosen_frac = float((chosen_lens > rejected_lens).mean())
print(f"\nChosen is longer than rejected in {longer_chosen_frac:.0%} of pairs.")
print(
    "If this number is >> 50%, the dataset has verbosity bias — the model\n"
    "may learn 'longer = better' instead of 'clearer = better'."
)

# ── Checkpoint Visual ────────────────────────────────────────────────────
assert (OUTPUT_DIR / "ex3_preference_lengths.png").exists(), "Visual must exist"
print("✓ Visual checkpoint passed — length distribution saved\n")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Customer-support alignment at a Singapore fintech
# ════════════════════════════════════════════════════════════════════════
# BUSINESS SCENARIO: A Singapore fintech (e.g., hypothetical "KaiPay")
# runs a support chatbot. After SFT on historical tickets the bot is
# competent but occasionally rude, occasionally refuses reasonable
# refund requests, and sometimes over-promises timelines.
#
# The support team reviewed 2000 bot responses and for each, picked the
# better of two candidates. That is exactly the UltraFeedback format
# (prompt, chosen, rejected). DPO will shift the bot toward the chosen
# style WITHOUT needing numeric scores.

print("=" * 70)
print("APPLICATION — KaiPay Singapore support-bot alignment")
print("=" * 70)

# --- Synthetic KaiPay preference pairs (mirrors UltraFeedback schema) ---
kaipay_pairs = pl.DataFrame(
    {
        "prompt": [
            "My PayNow transfer to my landlord failed but S$2400 was deducted. Help.",
            "I forgot my 6-digit PIN. How do I reset it?",
            "Can you cancel my auto-debit for my gym membership from yesterday?",
            "The app crashes when I try to view my August statement.",
        ],
        "chosen": [
            "I'm sorry for the scare. Failed PayNow transfers normally auto-reverse within 2 hours on the same business day. I've logged a priority trace on your reference and escalated to our settlements team. You'll get an SMS the moment the S$2400 lands back. If you haven't seen it by 5pm today, reply here and I'll open a formal reversal claim with MAS reference.",
            "Tap 'Forgot PIN' on the login screen, verify your NRIC last 4 digits and OTP, then set a new 6-digit PIN. The new PIN is active immediately — no overnight delay. Let me know if the OTP doesn't arrive and I'll walk through mobile-number verification.",
            "I can't reverse a cleared debit directly, but I can raise a dispute with your gym on your behalf under PDPA recurring-authorisation rules. I'll need the merchant name and today's transaction ID. Once lodged, your gym must respond within 10 business days.",
            "Thanks for reporting. The August statement crash is a known issue on app build 4.2.1. Fix is rolling out today — please update from the App Store/Play Store after 4pm. If you need the statement now, I can email a PDF to your registered address within 15 minutes.",
        ],
        "rejected": [
            "Failed transfers get reversed. Wait.",
            "Use forgot PIN flow.",
            "No, we can't cancel transactions after the fact.",
            "Try reinstalling the app. That usually works.",
        ],
    }
)

print(f"KaiPay preference pairs: {kaipay_pairs.height}")
print(f"\nSample chosen vs rejected:")
print(f"  Prompt:   {kaipay_pairs['prompt'][0][:80]}...")
print(f"  Chosen:   {kaipay_pairs['chosen'][0][:100]}...")
print(f"  Rejected: {kaipay_pairs['rejected'][0][:100]}...")

# --- Business impact back-of-envelope ---
DAILY_SUPPORT_CONVERSATIONS = 18000
ESCALATION_RATE_BEFORE = 0.12  # 12% escalate to human agent
ESCALATION_RATE_AFTER = 0.07  # 7% after DPO alignment (target)
HUMAN_AGENT_COST_SGD = 4.20  # cost per escalated conversation

daily_escalation_saved = DAILY_SUPPORT_CONVERSATIONS * (
    ESCALATION_RATE_BEFORE - ESCALATION_RATE_AFTER
)
daily_saved_sgd = daily_escalation_saved * HUMAN_AGENT_COST_SGD
annual_saved_sgd = daily_saved_sgd * 365

print("\n" + "-" * 64)
print("BUSINESS IMPACT — KaiPay support alignment via DPO")
print("-" * 64)
print(f"Daily support conversations:        {DAILY_SUPPORT_CONVERSATIONS:>10,}")
print(f"Human escalation rate (before):     {ESCALATION_RATE_BEFORE:>10.0%}")
print(f"Human escalation rate (after DPO):  {ESCALATION_RATE_AFTER:>10.0%}")
print(f"Escalations avoided per day:        {int(daily_escalation_saved):>10,}")
print(f"Cost saved per day:                 {'S$' + f'{daily_saved_sgd:,.0f}':>10}")
print(f"Cost saved per year:                {'S$' + f'{annual_saved_sgd:,.0f}':>10}")
print("-" * 64)

# ── Checkpoint Application ──────────────────────────────────────────────
assert kaipay_pairs.height >= 4, "Need at least 4 preference pairs for KaiPay"
assert {"prompt", "chosen", "rejected"}.issubset(set(kaipay_pairs.columns))
print("\n✓ Application checkpoint passed — KaiPay alignment dataset ready\n")


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
  [x] Loaded UltraFeedback Binarized — real human preference pairs
  [x] Understood the (prompt, chosen, rejected) triple structure
  [x] Split into train/eval for DPO
  [x] Visualised chosen-vs-rejected length distribution
      — early warning for verbosity bias in your dataset
  [x] Applied preference data to KaiPay Singapore support alignment
      — quantified the escalation-rate business case

  KEY INSIGHT: Pairwise preferences are the cheapest reliable human
  signal. Humans struggle to rate responses on a 1-10 scale consistently,
  but they CAN reliably pick a winner between two candidates. DPO
  exploits exactly that.

  Next: 02_dpo_loss.py derives the DPO loss and implements it from scratch.
"""
)
