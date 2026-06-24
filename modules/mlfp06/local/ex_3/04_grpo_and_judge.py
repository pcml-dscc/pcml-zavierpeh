# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 3.4: GRPO and LLM-as-Judge Evaluation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Understand GRPO (Group Relative Policy Optimization) and when it beats DPO
#   - Compute group-relative advantages and verify the zero-sum invariant
#   - Visualise GRPO advantages as a reward heatmap
#   - Run an LLM-as-judge evaluation with Kaizen Delegate
#   - Measure two known biases: position bias and verbosity bias
#   - Survey standard benchmarks (MMLU, HellaSwag, HumanEval, MT-Bench, etc.)
#   - Apply to an NUH clinical RAG assistant evaluation plan
#
# PREREQUISITES: 03_dpo_training.py (you trained a DPO adapter).
# ESTIMATED TIME: ~50 min
#
# TASKS:
#   1. GRPO theory and zero-sum advantage computation
#   2. Visualise GRPO advantages
#   3. LLM-as-judge: compare two responses with Kaizen Delegate
#   4. Position bias test (swap A/B)
#   5. Verbosity bias test (concise vs padded response)
#   6. Benchmarks survey
#   7. Apply: NUH clinical RAG assistant evaluation plan
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json

import polars as pl
import torch
from kaizen_agents import Delegate

from shared.mlfp06.ex_3 import (
    MODEL_NAME,
    OUTPUT_DIR,
    show_grpo_advantages,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — GRPO vs DPO in one page
# ════════════════════════════════════════════════════════════════════════
# GRPO (DeepSeek-R1 2025):
#   For each prompt, sample K completions, score each with r(x, y_i),
#   compute advantage A_i = r_i - mean(r_1,...,r_K), update with
#   L = -E[sum_i A_i * log pi(y_i|x)].
# DPO = pairwise preferences; GRPO = group-relative scoring with any
# reward function. GRPO is the right call when a verifiable reward
# exists (math correctness, code execution).


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — GRPO advantage computation
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: GRPO advantages")
print("=" * 70)

torch.manual_seed(42)
K = 5
n_prompts = 8
rewards = torch.randn(n_prompts, K)

# TODO: Compute advantages by subtracting the per-prompt (row-wise) mean.
#       Hint: rewards.mean(dim=1, keepdim=True)
advantages = ____

print(f"  Advantage sum per group ~ 0: {advantages.sum(dim=1).mean().item():.6f}")
assert advantages.shape == rewards.shape
assert abs(advantages.sum(dim=1).mean().item()) < 1e-5
print("✓ Checkpoint 1 passed — zero-sum invariant holds\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Reward and advantage heatmaps
# ════════════════════════════════════════════════════════════════════════

# TODO: Call show_grpo_advantages(rewards, advantages).
____
assert (OUTPUT_DIR / "ex3_grpo_advantages.png").exists()
print("✓ Visual checkpoint passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — LLM-as-judge
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: LLM-as-judge with Kaizen Delegate")
print("=" * 70)


async def llm_judge(prompt: str, response_a: str, response_b: str) -> dict:
    """Ask an LLM to pick between two responses. Returns parsed JSON verdict."""
    # TODO: Construct a Kaizen Delegate with model=MODEL_NAME, max_llm_cost_usd=0.5.
    delegate = ____

    judge_prompt = f"""You are an impartial judge evaluating two responses to a user query.

Query: {prompt[:500]}

Response A:
{response_a[:500]}

Response B:
{response_b[:500]}

Evaluate on: helpfulness, accuracy, clarity, safety.
Output ONLY a JSON object:
{{"winner": "A" or "B" or "tie", "score_a": 1-10, "score_b": 1-10, "reasoning": "..."}}"""

    response = ""
    async for event in delegate.run(judge_prompt):
        if hasattr(event, "text"):
            response += event.text

    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        return json.loads(response[start:end])
    except (ValueError, Exception):
        return {"winner": "tie", "score_a": 5, "score_b": 5, "reasoning": "parse error"}


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Position bias test
# ════════════════════════════════════════════════════════════════════════


async def measure_position_bias() -> float:
    pairs = [
        {
            "prompt": "What is the capital of Singapore?",
            "good": "Singapore is a city-state; the entire country is the capital.",
            "bad": "Probably somewhere in Asia. I don't remember exactly.",
        },
        {
            "prompt": "How does public key cryptography work?",
            "good": "Each party has a public key and a private key. Messages "
            "encrypted with the public key can only be decrypted by the "
            "matching private key.",
            "bad": "It uses two keys somehow.",
        },
        {
            "prompt": "Summarise photosynthesis in one sentence.",
            "good": "Plants convert sunlight, water, and CO2 into glucose and "
            "oxygen using chlorophyll.",
            "bad": "Plants eat sunlight.",
        },
    ]
    consistent = 0
    for p in pairs:
        # TODO: Call llm_judge twice — once with good as A, once with good as B.
        ab = ____
        ba = ____
        ab_picks_good = ab.get("winner") == "A"
        ba_picks_good = ba.get("winner") == "B"
        consistent += int(ab_picks_good == ba_picks_good)
    return consistent / len(pairs)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Verbosity bias test
# ════════════════════════════════════════════════════════════════════════


async def measure_verbosity_bias() -> dict:
    test_prompt = "What is machine learning?"
    concise = (
        "Machine learning is a subset of AI where algorithms learn patterns from "
        "data to make predictions without explicit programming."
    )
    verbose = (
        "Machine learning is a very interesting and important field of study. "
        "It is essentially a subset of artificial intelligence. The basic idea is "
        "that instead of explicitly programming every rule, we let the computer "
        "learn from data. "
    ) * 3
    # TODO: Call llm_judge(test_prompt, concise, verbose) and return the verdict.
    verdict = ____
    return verdict


position_consistency = asyncio.run(measure_position_bias())
verbosity_verdict = asyncio.run(measure_verbosity_bias())

print(f"  Position consistency: {position_consistency:.0%}")
print(f"  Verbosity verdict:    {verbosity_verdict.get('winner')}")

assert 0.0 <= position_consistency <= 1.0
assert "winner" in verbosity_verdict
print("✓ Checkpoint 5 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Benchmarks survey
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Evaluation Benchmarks Survey")
print("=" * 70)

# TODO: Build a polars DataFrame with 8 rows covering:
#   MMLU, HellaSwag, HumanEval, MT-Bench, TruthfulQA, GSM8K, MBPP, ARC-Challenge.
#   Columns: Benchmark, Domain, Format, Measures.
benchmarks = ____

print(benchmarks)
assert benchmarks.height >= 8
print("✓ Checkpoint 6 passed\n")


# ════════════════════════════════════════════════════════════════════════
# APPLY — NUH clinical RAG assistant evaluation plan
# ════════════════════════════════════════════════════════════════════════
# Gates: safety refusal >= 85%, MMLU med >= baseline-2pp, TruthfulQA >= 80%,
#        position-swap consistency >= 75%.

print("=" * 70)
print("APPLICATION — NUH clinical RAG assistant evaluation plan")
print("=" * 70)

# TODO: Build a 5-row polars DataFrame with columns:
#   Dimension, Method, "Approval Gate", Current
# Cover Safety, Knowledge, Reasoning, Honesty, Judge Quality. Use the
# position_consistency measurement from above in the "Current" column.
eval_plan = ____

print(eval_plan)
assert eval_plan.height == 5
print("✓ Application checkpoint passed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] GRPO intuition: zero-sum group-relative advantages
  [x] LLM-as-judge with structured JSON verdicts
  [x] Measured position and verbosity biases
  [x] Surveyed the standard LLM benchmarks
  [x] Drafted an NUH clinical RAG evaluation plan with approval gates

  NEXT: Exercise 4 (RAG) grounds LLM responses in retrieved documents.
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
