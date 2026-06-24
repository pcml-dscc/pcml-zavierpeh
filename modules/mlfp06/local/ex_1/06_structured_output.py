# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1.6: Structured Output with Kaizen Signature
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Define a typed Kaizen Signature with InputField and OutputField
#   - Drive an LLM with a type-safe schema instead of free-form text
#   - Access results via attribute access (result.sentiment), not dict keys
#   - Understand why Signatures are the production standard
#
# PREREQUISITES: 01-05 complete
# ESTIMATED TIME: ~40 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from kaizen import InputField, OutputField, Signature
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent

from shared.mlfp06.ex_1 import MODEL, get_eval_docs

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Typed Signatures Beat Free-Form JSON
# ════════════════════════════════════════════════════════════════════════
# Free-form JSON fails: format drift, schema drift, silent data loss.
# Kaizen Signatures declare types; Kaizen renders them into a schema
# prompt, validates responses, retries on failure. Attribute access,
# not dict lookups. This is the production standard.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the Signature
# ════════════════════════════════════════════════════════════════════════


class ReviewExtraction(Signature):
    """Extract structured information from a movie review snippet."""

    # TODO: Declare input field review_text: str = InputField(description=...)
    ____

    # TODO: Declare output fields:
    #   sentiment: str — "positive" or "negative"
    #   confidence: float — 0.0 to 1.0
    #   key_phrases: list[str] — up to 5 phrases
    #   targets: list[str] — aspects evaluated (acting, plot, visuals)
    #   tone: str — enthusiastic, measured, disappointed, angry, sarcastic
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (run the Signature-backed agent)
# ════════════════════════════════════════════════════════════════════════


async def run_signature_extraction() -> list:
    # TODO: Construct SimpleQAAgent(signature=ReviewExtraction, model=MODEL,
    # max_llm_cost_usd=1.0)
    agent = ____

    docs = get_eval_docs().head(10)
    results = []
    # TODO: For each text, call `await agent.run(review_text=text[:800])` and
    # append to results. Print the first 3 with typed attribute access
    # (result.sentiment, result.confidence, result.key_phrases, result.tone).
    ____
    return results


print("\n" + "=" * 70)
print("  Structured Output via Kaizen Signature")
print("=" * 70)
signature_results = asyncio.run(run_signature_extraction())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(signature_results) > 0, "Task 3: Signature extraction should produce results"
sample = signature_results[0]
assert hasattr(sample, "sentiment"), "Result should have typed 'sentiment' field"
assert hasattr(sample, "confidence"), "Result should have typed 'confidence' field"
assert 0.0 <= sample.confidence <= 1.0, "Confidence should be in [0, 1]"
assert hasattr(sample, "key_phrases"), "Result should have typed 'key_phrases' field"
assert isinstance(sample.key_phrases, list), "key_phrases should be a list"
assert hasattr(sample, "tone"), "Result should have typed 'tone' field"
print(
    f"\n[ok] Checkpoint passed — Signature extraction: "
    f"sentiment='{sample.sentiment}', confidence={sample.confidence:.2f}\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE — typed field access
# ════════════════════════════════════════════════════════════════════════
avg_conf = sum(r.confidence for r in signature_results) / len(signature_results)
tones = [r.tone for r in signature_results]
print(f"\n  Avg confidence across {len(signature_results)} reviews: {avg_conf:.2f}")
print(f"  Tone distribution: {tones}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Grab Driver Incident Report Extraction
# ════════════════════════════════════════════════════════════════════════
# Grab receives ~1.2K driver incident reports/day. Each must be decomposed
# into a strongly-typed record for DataFlow + insurance-partner APIs.
# Signatures are mandatory: downstream tables have strict schemas, and
# insurance partners require compliant JSON. Free-form parsing drops
# fields silently; Signatures retry or raise a typed error.
#
# BUSINESS IMPACT: 8% -> <0.5% parse error rate at S$180/error avoids
# S$5.8M/year in rework, vs S$45K/year in LLM cost. 129x ROI.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Defined a Kaizen Signature with typed InputField + OutputField
  [x] Drove an LLM via SimpleQAAgent with the Signature schema
  [x] Accessed results via typed attributes, not dict lookups
  [x] Understood why Signatures solve free-form JSON's failure modes

  Where this goes next:
    - Exercise 2: fine-tune with LoRA adapters
    - Exercise 3: DPO — skip the reward model
    - Exercise 6: wire into PACT governance + Nexus deployment
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

# Primary lens: Output (LLM-as-judge over the classifier's predictions).
# We'd pass the predicted label + true label as (prompt, response) pairs
# and ask a judge to score coherence/faithfulness. Attention is optional
# here — only meaningful for open-weight models.
if False:  # scaffold — requires OPENAI_API_KEY + judge budget
    obs = LLMObservatory(run_id="ex_1_prompting_run")
    # Build (prompt, response) pairs from the exercise results:
    # prompts = [r["text"] for r in zero_shot_results]
    # responses = [r["pred"] for r in zero_shot_results]
    # obs.output.evaluate(prompts, responses, criteria="coherence,label_fidelity")
    print("\n── LLM Observatory Report ──")
    findings = obs.report()
    # Optional: obs.plot_dashboard().show()

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Output     (HEALTHY): judge coherence 0.91, label_fidelity 0.84
#   [?] Attention  (n/a): API-only model — lens short-circuits to UNKNOWN
#   [?] Retrieval  (n/a): no retrieval in this exercise
#   [?] Agent      (n/a): no tool-using agent in this exercise
#   [?] Alignment  (n/a): no fine-tuning signal to compare
#   [?] Governance (n/a): no PACT engine attached
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [OUTPUT LENS] judge coherence 0.91 is HEALTHY (>0.80). label_fidelity
#     0.84 means the judge thought 84% of predictions were coherent
#     labels in the allowed category set. The remaining 16% are where
#     the LLM drifted off-template ("Positive sentiment, I think" instead
#     of "positive") — a signature of under-constrained zero-shot.
#     >> Prescription: tighten the prompt (structured output in ex_1.6)
#        or add few-shot exemplars (ex_1.2).
#
#  [ATTENTION LENS] GPT-class models are API-only — the Attention lens
#     short-circuits to UNKNOWN. To actually inspect attention, switch to
#     an open-weight model (e.g. Qwen2-0.5B via transformers) and call
#     obs.attention.logit_lens(prompt=..., answer_token=...).
#
#  [OTHER LENSES] All n/a — prompting has no retrieval, no agent loop, no
#     fine-tuning pair, no governance envelope. This is exactly the
#     signature the design doc predicts for Lesson 6.1.
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
