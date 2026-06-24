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

import matplotlib.pyplot as plt

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
#
# Non-technical analogy: a medical case conference. The GP presents a
# patient; a cardiologist, a radiologist and a pharmacologist each give
# their independent read; then the attending physician (supervisor)
# synthesises all three opinions into a treatment plan. Nobody writes
# a mega-prompt that says "be a cardiologist AND radiologist AND
# pharmacologist" — you get better judgement by keeping the specialists
# focused and letting the attending decide.
#
# WHY IT BEATS ONE MEGA-PROMPT:
#   - Each specialist has a narrow, high-signal Signature → less drift
#   - The supervisor sees STRUCTURED specialist output, not free text
#   - Audit trail: you can trace exactly which specialist said what
#   - Costs are predictable per specialist (Kaizen max_llm_cost_usd)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load the SQuAD 2.0 corpus
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Load SQuAD 2.0 Multi-Domain Corpus")
print("=" * 70)

passages = load_squad_corpus()
print(f"Passages: {passages.height}, unique titles: {passages['title'].n_unique()}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert passages.height > 0, "Task 1: corpus should not be empty"
assert passages["title"].n_unique() > 10, "Corpus should span many titles"
print("✓ Checkpoint 1 passed — multi-domain corpus loaded\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the specialists and the synthesis supervisor
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Specialist + Supervisor Agents")
print("=" * 70)

factual_agent, semantic_agent, structural_agent = build_specialists()
synthesis_agent = build_synthesis()

specialists = [factual_agent, semantic_agent, structural_agent]
print(f"Created {len(specialists)} specialists + 1 supervisor")
for agent in specialists:
    print(f"  {agent.__class__.__name__}: {agent.description}")
print(f"  {synthesis_agent.__class__.__name__}: {synthesis_agent.description}")

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

    # Fan-out: run each specialist independently
    factual_result = await factual_agent.run_async(document=doc, question=question)
    semantic_result = await semantic_agent.run_async(document=doc, question=question)
    structural_result = await structural_agent.run_async(
        document=doc, question=question
    )

    # Fan-in: supervisor synthesises the three structured outputs
    synthesis_result = await synthesis_agent.run_async(
        document=doc,
        question=question,
        factual_analysis=(
            f"Claims: {factual_result['factual_claims']}, "
            f"Evidence: {factual_result['evidence_quality']}"
        ),
        semantic_analysis=(
            f"Themes: {semantic_result['main_themes']}, "
            f"Implicit: {semantic_result['implicit_info']}"
        ),
        structural_analysis=(
            f"Structure: {structural_result['structure_type']}, "
            f"Entities: {structural_result['key_entities']}"
        ),
    )

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
print(f"Question: {question}")
print(f"Passage title: {passages['title'][0]}")

sv_result = asyncio.run(supervisor_worker_analysis(doc, question))

print(f"\nUnified answer: {sv_result['answer'][:300]}...")
print(f"Confidence: {sv_result['confidence']:.2f}")
print(f"Reasoning steps: {len(sv_result['reasoning'])}")
print(f"Latency: {sv_result['latency_s']:.1f}s")
print("\n--- Audit trail (who said what) ---")
print(f"  Factual claims (top 3): {sv_result['factual_claims'][:3]}")
print(f"  Semantic themes (top 3): {sv_result['themes'][:3]}")
print(f"  Structural entities (top 3): {sv_result['entities'][:3]}")

trace_path = OUTPUT_DIR / "ex6_supervisor_worker_trace.txt"
trace_path.write_text(
    f"Question: {question}\n\n"
    f"Unified Answer:\n{sv_result['answer']}\n\n"
    f"Confidence: {sv_result['confidence']:.2f}\n"
    f"Factual Claims: {sv_result['factual_claims']}\n"
    f"Semantic Themes: {sv_result['themes']}\n"
    f"Structural Entities: {sv_result['entities']}\n"
)
print(f"\nTrace written to: {trace_path}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert sv_result["answer"], "Task 3: should produce a unified answer"
assert 0 <= sv_result["confidence"] <= 1, "Confidence should be in [0, 1]"
assert sv_result["factual_claims"], "Factual specialist should contribute claims"
print("\n✓ Checkpoint 3 passed — supervisor-worker pattern complete\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Agent contribution and latency breakdown
# ════════════════════════════════════════════════════════════════════════
# Visual proof that the supervisor-worker pattern distributes work across
# specialists. The bar chart shows each specialist's contribution count
# and the overall latency, giving students a concrete sense of the
# fan-out / fan-in trade-off.

agents = ["Factual", "Semantic", "Structural", "Supervisor"]
contributions = [
    len(sv_result["factual_claims"]),
    len(sv_result["themes"]),
    len(sv_result["entities"]),
    len(sv_result["reasoning"]),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Left: agent contribution counts
colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
ax1.bar(agents, contributions, color=colors)
ax1.set_ylabel("Items contributed")
ax1.set_title("Specialist Contributions (Fan-Out)", fontweight="bold")
for i, c in enumerate(contributions):
    ax1.text(i, c + 0.1, str(c), ha="center", fontsize=10)

# Right: latency pie (supervisor-worker is serial here; shows time split)
total = sv_result["latency_s"]
ax2.barh(["Total latency"], [total], color="#34495e", height=0.4)
ax2.set_xlabel("Seconds")
ax2.set_title("End-to-End Latency", fontweight="bold")
ax2.text(total + 0.1, 0, f"{total:.1f}s", va="center", fontsize=11)

plt.tight_layout()
fname = OUTPUT_DIR / "ex6_supervisor_worker_viz.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore scenario: insurance claims triage
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore general insurer handles ~8,000 personal-injury
# claims a month. Each claim has a narrative report (doctor notes,
# police statement, claimant description). A single-agent triage bot
# currently reads every claim and flags suspicious ones, but misses
# ~12% of fraud cases — the agent skims narratives and locks onto
# surface keywords like "accident" without auditing the evidence.
#
# The supervisor-worker pattern fixes this:
#   Factual specialist: "What events, dates, amounts are claimed?"
#   Semantic specialist: "What is implied but not stated?"
#   Structural specialist: "Which parties are involved and how?"
#   Supervisor: merges the three into a triage recommendation
#
# EXPECTED IMPACT: at the insurer's scale, raising fraud detection from
# 88% to 95% is ~550 extra fraud cases caught per month. At an average
# fraud claim of S$8,000, that is S$4.4M/month in loss prevention —
# vastly more than the ~4× LLM cost vs the single-agent baseline.
#
# AUDIT BONUS: the Monetary Authority of Singapore (MAS) can ask
# "why was this claim flagged?" and the answer is a structured trace,
# not a black-box decision — exactly what TRM guidelines require.

print("=" * 70)
print("  SINGAPORE APPLICATION: Insurance Claims Triage")
print("=" * 70)
print(
    """
  Scale: 8,000 personal-injury claims/month
  Baseline single-agent fraud catch rate: 88%
  Supervisor-worker target:                95%
  Net additional fraud caught:             ~550 cases/month
  Average fraud claim size:                S$8,000
  Monthly loss-prevention delta:           ~S$4.4M
  LLM cost multiplier vs single-agent:     ~4× (still negligible vs gain)
  Audit trail:                             structured per-specialist outputs
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
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Specialist agents with domain-specific Signatures (Factual,
      Semantic, Structural)
  [x] Supervisor-worker pattern: fan-out to specialists, fan-in to
      the synthesis supervisor
  [x] Structured audit trail: which specialist contributed which claim
  [x] Why a focused Signature beats a mega-prompt for quality AND cost
  [x] The Singapore insurance triage scenario — quantified impact of
      multi-agent decomposition on a regulated workflow

  KEY INSIGHT: Decomposition is not about parallelism — it is about
  keeping each LLM call focused enough to produce high-signal structured
  output. The supervisor is the audit boundary; each specialist is a
  single-purpose expert whose output can be traced, tested, and priced.

  Next: 02_sequential_pipeline.py — when specialists must build on
  each other rather than run independently.
"""
)
