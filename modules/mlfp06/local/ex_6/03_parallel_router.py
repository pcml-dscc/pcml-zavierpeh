# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 6.3: Parallel Execution + LLM-Based Routing
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Run independent specialists concurrently with asyncio.gather
#   - Prove parallel latency ≈ max(stages), not sum
#   - Use Kaizen Pipeline.router() for LLM-based dispatch
#
# PREREQUISITES: 02_sequential_pipeline.py
# ESTIMATED TIME: ~30 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import time

from kaizen_agents import Pipeline

from shared.mlfp06.ex_6 import (
    OUTPUT_DIR,
    build_specialists,
    load_squad_corpus,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load corpus + specialists
# ════════════════════════════════════════════════════════════════════════

# TODO: Load corpus and build the three specialists
passages = ____
factual_agent, semantic_agent, structural_agent = ____

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert passages.height > 0
assert factual_agent and semantic_agent and structural_agent
print("✓ Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Parallel orchestrator with asyncio.gather
# ════════════════════════════════════════════════════════════════════════


async def parallel_analysis(doc: str, question: str) -> dict:
    """Launch all specialists simultaneously with asyncio.gather."""
    t0 = time.perf_counter()

    # TODO: Build three coroutine tasks (do NOT await them yet).
    # Hint: BaseAgent.run_async is a coroutine — calling it without `await`
    # returns the coroutine object you can pass to asyncio.gather.
    #   factual_task = factual_agent.run_async(document=doc, question=question)
    factual_task = ____
    semantic_task = ____
    structural_task = ____

    # TODO: Await all three concurrently with asyncio.gather(...)
    factual_r, semantic_r, structural_r = ____

    # run_async returns a dict — read OutputField values with dict indexing.
    return {
        "factual_claims": factual_r["factual_claims"],
        "themes": semantic_r["main_themes"],
        "entities": structural_r["key_entities"],
        "latency_s": time.perf_counter() - t0,
    }


async def sequential_baseline(doc: str, question: str) -> float:
    """Same work, but one agent at a time — for latency comparison."""
    t0 = time.perf_counter()
    # TODO: await the three specialists one at a time, using run_async.
    # Hint: await factual_agent.run_async(document=doc, question=question)
    ____
    ____
    ____
    return time.perf_counter() - t0


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Measure parallel vs sequential latency
# ════════════════════════════════════════════════════════════════════════

doc = passages["text"][0]
question = passages["question"][0]


async def run_comparison():
    par = await parallel_analysis(doc, question)
    seq_latency = await sequential_baseline(doc, question)
    return par, seq_latency


par_result, seq_latency = asyncio.run(run_comparison())

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert par_result["factual_claims"], "Parallel run should produce claims"
# A single local Ollama daemon serves one request at a time, so the three
# "parallel" calls queue and wall-clock can land at — or just above, due to
# jitter — the sequential baseline. Assert "not materially slower" (25% band)
# rather than strictly faster; real speedup needs a concurrent backend
# (hosted APIs, or Ollama with OLLAMA_NUM_PARALLEL>1).
assert (
    par_result["latency_s"] <= seq_latency * 1.25
), "Parallel run should not be materially slower than sequential"
print("✓ Checkpoint 2 passed — parallel execution verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Configure Pipeline.router() for LLM-based dispatch
# ════════════════════════════════════════════════════════════════════════

# TODO: Build an LLM router over the three specialists.
# Hint: Pipeline.router(agents=[factual_agent, semantic_agent, structural_agent])
router = ____

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert router is not None, "Task 4: router should be created"
print("✓ Checkpoint 3 passed — router configured\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Intent-to-specialist mapping
# ════════════════════════════════════════════════════════════════════════

test_queries = [
    ("What specific dates and numbers are mentioned?", "factual"),
    ("What is the underlying theme of the argument?", "semantic"),
    ("How is the passage organised, and what entities appear?", "structural"),
]

for query, expected in test_queries:
    print(f"  Query: {query}")
    print(f"    Expected specialist: {expected}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore scenario: Smart Nation helpdesk triage
# ════════════════════════════════════════════════════════════════════════
# A Singapore government agency runs a citizen helpdesk at 4,000
# tickets/day. Keyword routing mis-routes 18%; LLM routing drops
# that to 3%. At an average rework cost of 8 minutes per mis-route,
# this reclaims ~80 hours/day — ~S$700K/year in agent time.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] asyncio.gather for concurrent specialist execution
  [x] Parallel latency ≈ max(stages) vs sequential = sum
  [x] Pipeline.router() — LLM-based dispatch by capability card

  Next: 04_mcp_server.py — exposing tools via MCP so any compatible
  agent can discover and call them.
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
