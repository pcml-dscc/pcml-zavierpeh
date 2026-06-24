# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 6.1: Supervisor-Worker Multi-Agent Pattern
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build specialist agents with domain-specific Kaizen Signatures
#   - Orchestrate the supervisor-worker (fan-out / fan-in) pattern
#   - Fan-out: dispatch the same question to three independent specialists
#   - Fan-in: a supervisor synthesises specialist outputs into one answer
#   - Why decomposing analysis across specialists beats one mega-prompt
#   - Audit trail: every specialist's contribution is structured and traceable
#
# PREREQUISITES: Exercise 5 (BaseAgent, Signature, ReActAgent, single-agent)
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Load SQuAD 2.0 multi-domain corpus
#   2. Instantiate three specialists and a synthesis supervisor
#   3. Build the supervisor-worker async orchestrator
#   4. Run it against a real passage and inspect the audit trail
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import time

from shared.mlfp06.ex_6 import (
    OUTPUT_DIR,
    build_specialists,
    build_synthesis,
    load_squad_corpus,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Supervisor-Worker (Fan-Out / Fan-In)
# ════════════════════════════════════════════════════════════════════════
# One supervisor sits above N specialist workers. The supervisor receives
# a complex task, fans it out to specialists (each with a focused prompt
# and a structured Signature), then fans the results back in and
# synthesises them into a single decision.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load the SQuAD 2.0 corpus
# ════════════════════════════════════════════════════════════════════════

# TODO: Call load_squad_corpus() from shared.mlfp06.ex_6
passages = ____

print(f"Passages: {passages.height}, unique titles: {passages['title'].n_unique()}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert passages.height > 0, "Task 1: corpus should not be empty"
assert passages["title"].n_unique() > 10, "Corpus should span many titles"
print("✓ Checkpoint 1 passed — multi-domain corpus loaded\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the specialists and the synthesis supervisor
# ════════════════════════════════════════════════════════════════════════

# TODO: Use build_specialists() — returns (factual, semantic, structural)
factual_agent, semantic_agent, structural_agent = ____

# TODO: Use build_synthesis() to create the supervisor
synthesis_agent = ____

specialists = [factual_agent, semantic_agent, structural_agent]

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(specialists) == 3, "Task 2: should have 3 specialists"
assert synthesis_agent is not None, "Task 2: supervisor should exist"
print("✓ Checkpoint 2 passed — four agents wired\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the supervisor-worker orchestrator
# ════════════════════════════════════════════════════════════════════════


async def supervisor_worker_analysis(doc: str, question: str) -> dict:
    """Run the full fan-out / fan-in pattern for one (doc, question)."""
    t0 = time.perf_counter()

    # TODO: Fan-out — call each specialist with (document=doc, question=question).
    # Hint: BaseAgent exposes an async `run_async(**inputs)` entry point that
    # returns a dict keyed by the Signature's OutputField names.
    # Example: factual_result = await factual_agent.run_async(
    #              document=doc, question=question,
    #          )
    factual_result = ____
    semantic_result = ____
    structural_result = ____

    # TODO: Fan-in — call synthesis_agent.run_async(...) passing:
    #   document, question, factual_analysis, semantic_analysis, structural_analysis.
    # Format each *_analysis as a string summary of the matching specialist output,
    # reading specialist outputs via dict indexing — e.g. factual_result["factual_claims"].
    synthesis_result = ____

    elapsed = time.perf_counter() - t0
    return {
        "answer": synthesis_result["unified_answer"],
        "confidence": synthesis_result["confidence"],
        "reasoning": synthesis_result["reasoning_chain"],
        "factual_claims": factual_result["factual_claims"],
        "themes": semantic_result["main_themes"],
        "entities": structural_result["key_entities"],
        "latency_s": elapsed,
    }


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Run it and inspect the audit trail
# ════════════════════════════════════════════════════════════════════════

doc = passages["text"][0]
question = passages["question"][0]

# TODO: Use asyncio.run() to execute supervisor_worker_analysis(doc, question)
sv_result = ____

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert sv_result["answer"], "Task 3: should produce a unified answer"
assert 0 <= sv_result["confidence"] <= 1, "Confidence should be in [0, 1]"
assert sv_result["factual_claims"], "Factual specialist should contribute claims"
print("✓ Checkpoint 3 passed — supervisor-worker pattern complete\n")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore scenario: insurance claims triage
# ════════════════════════════════════════════════════════════════════════
# At a Singapore general insurer handling ~8,000 personal-injury claims
# per month, raising fraud detection from 88% to 95% via supervisor-worker
# decomposition yields ~S$4.4M/month in loss prevention — and a MAS-
# defensible audit trail per claim.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Specialist agents with domain-specific Signatures
  [x] Supervisor-worker pattern: fan-out + fan-in synthesis
  [x] Structured audit trail per specialist
  [x] When a focused Signature beats a mega-prompt

  Next: 02_sequential_pipeline.py — when specialists must build on
  each other rather than run independently.
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

# Primary lens: Agent Trace (inter-agent handoffs, tool latency).
# Secondary: Governance (envelope verification when a supervisor is
# governed).
if False:  # scaffold — requires a live multi-agent setup
    obs = LLMObservatory(run_id="ex_6_multiagent_run")
    # for run_id, trace in supervisor.all_traces.items():
    #     obs.agent.register_trace(trace)
    # obs.agent.handoff_summary()  # inter-agent handoffs
    print("\n── LLM Observatory Report ──")
    findings = obs.report()

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Agent      (HEALTHY): 3 workers, 7 handoffs, mean tool-call
#       latency 840ms, no stuck loops across all runs.
#   [?] Governance (UNKNOWN): no PACT engine attached in this lesson;
#       attach supervisor.audit to light up this lens.
#   [?] Output / Retrieval / Alignment / Attention (n/a)
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [AGENT LENS] 7 handoffs across 3 workers is the signature of a
#     healthy Supervisor-Worker pattern — supervisor delegates, workers
#     report back, supervisor synthesises. Mean latency 840ms per tool
#     call is dominated by LLM inference, not tool execution. Watch for:
#     (a) a worker that handoffs 0 times = it's not being used;
#     (b) latency >5s = a tool is I/O bound and needs caching.
#  [GOVERNANCE LENS] UNKNOWN is expected in ex_6 — governance shows up
#     in ex_7 where the GovernedSupervisor attaches its audit trail.
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
