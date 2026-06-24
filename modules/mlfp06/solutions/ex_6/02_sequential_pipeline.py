# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 6.2: Sequential Multi-Agent Pipeline (A → B → C)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a 3-stage sequential agent pipeline where each stage feeds
#     the next (extract → interpret → synthesise)
#   - When sequential beats fan-out: downstream stages need upstream
#     structured output as input
#   - Latency model: sequential latency = SUM of stage latencies
#   - Contrast with supervisor-worker fan-out (independent specialists)
#
# PREREQUISITES: 01_supervisor_worker.py (you know the specialist agents)
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Load the corpus and wire a fresh specialist set
#   2. Instantiate the stage-2 InterpretationAgent
#   3. Build the 3-stage pipeline orchestrator
#   4. Run it and reason about sequential latency
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import time

import matplotlib.pyplot as plt

from shared.mlfp06.ex_6 import (
    InterpretationAgent,
    OUTPUT_DIR,
    build_specialists,
    build_synthesis,
    load_squad_corpus,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Sequential Pipeline (A → B → C)
# ════════════════════════════════════════════════════════════════════════
# Sometimes specialists CANNOT run in parallel because each stage
# depends on the structured output of the previous one. Example:
#
#   Stage 1 (Factual extraction):   pull raw claims from the passage
#   Stage 2 (Interpretation):       interpret stage-1 claims in context
#   Stage 3 (Synthesis):            produce final answer from stage-2
#
# Stage 2 can't start until stage 1 has produced claims. Stage 3
# can't start until stage 2 has ranked them. This is a DAG with
# one linear edge per stage.
#
# Non-technical analogy: a factory assembly line for a handmade watch.
# The case maker has to finish before the movement fitter can begin,
# who has to finish before the dial painter can begin. You cannot
# parallelise the line without losing the dependency guarantees.
#
# LATENCY TRADE-OFF:
#   Supervisor-worker (fan-out): latency ≈ max(stages)
#   Sequential pipeline:         latency ≈ sum(stages)
#
# You pay latency in exchange for the ability to CHAIN reasoning.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load corpus and specialists
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Load corpus + specialists")
print("=" * 70)

passages = load_squad_corpus()
factual_agent, _semantic_agent, _structural_agent = build_specialists()
synthesis_agent = build_synthesis()
print(f"Corpus: {passages.height} passages")
print("Specialists + synthesis supervisor instantiated")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert passages.height > 0, "Task 1: corpus should be non-empty"
assert factual_agent is not None, "Task 1: factual agent should exist"
print("✓ Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Instantiate the stage-2 interpreter
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Stage-2 InterpretationAgent")
print("=" * 70)

interpreter = InterpretationAgent()
print(f"  {interpreter.__class__.__name__}: {interpreter.description}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert interpreter is not None, "Task 2: interpreter should exist"
print("✓ Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the 3-stage sequential pipeline
# ════════════════════════════════════════════════════════════════════════


async def sequential_pipeline(doc: str, question: str) -> dict:
    """Run 3-stage sequential pipeline: extract → interpret → synthesise."""
    t0 = time.perf_counter()
    per_stage: list[tuple[str, float]] = []

    # Stage 1: Factual extraction
    s1_t = time.perf_counter()
    factual = await factual_agent.run_async(document=doc, question=question)
    per_stage.append(("factual extraction", time.perf_counter() - s1_t))

    # Stage 2: Interpretation (consumes stage-1 output)
    s2_t = time.perf_counter()
    interpreted = await interpreter.run_async(
        factual_claims=str(factual["factual_claims"]),
        document=doc,
        question=question,
    )
    per_stage.append(("contextual interpretation", time.perf_counter() - s2_t))

    # Stage 3: Synthesis (consumes stage-2 output)
    s3_t = time.perf_counter()
    final = await synthesis_agent.run_async(
        document=doc,
        question=question,
        factual_analysis=str(interpreted["interpreted_facts"]),
        semantic_analysis=str(interpreted["relevance_ranking"]),
        structural_analysis=f"Evidence quality: {factual['evidence_quality']}",
    )
    per_stage.append(("synthesis", time.perf_counter() - s3_t))

    elapsed = time.perf_counter() - t0
    return {
        "answer": final["unified_answer"],
        "confidence": final["confidence"],
        "stages": per_stage,
        "latency_s": elapsed,
    }


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Run and inspect the latency breakdown
# ════════════════════════════════════════════════════════════════════════

doc = passages["text"][0]
question = passages["question"][0]
print(f"Question: {question}")

result = asyncio.run(sequential_pipeline(doc, question))

print(f"\nAnswer: {result['answer'][:250]}...")
print(f"Confidence: {result['confidence']:.2f}")
print(f"\nPer-stage latency:")
for name, dt in result["stages"]:
    print(f"  {name:30s} {dt:5.1f}s")
print(f"  {'TOTAL (sum of stages)':30s} {result['latency_s']:5.1f}s")

trace_path = OUTPUT_DIR / "ex6_sequential_pipeline_trace.txt"
trace_path.write_text(
    f"Question: {question}\n\nAnswer:\n{result['answer']}\n\n"
    + "\n".join(f"{n}: {t:.2f}s" for n, t in result["stages"])
    + f"\nTotal: {result['latency_s']:.2f}s\n"
)
print(f"\nTrace written to: {trace_path}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert result["answer"], "Task 3: should produce a final answer"
assert len(result["stages"]) == 3, "Should log 3 stages"
print("\n✓ Checkpoint 3 passed — sequential pipeline complete\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Pipeline stage timing waterfall chart
# ════════════════════════════════════════════════════════════════════════
# Visual proof of the sequential latency model: total = sum of stages.
# The waterfall shows each stage starting where the previous one ended,
# making the dependency chain and its latency cost visually obvious.

stage_names = [name for name, _ in result["stages"]]
stage_times = [dt for _, dt in result["stages"]]
cumulative = [sum(stage_times[:i]) for i in range(len(stage_times))]

fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#3498db", "#2ecc71", "#e67e22"]
for i, (start, dur, name) in enumerate(zip(cumulative, stage_times, stage_names)):
    ax.barh(
        0,
        dur,
        left=start,
        height=0.5,
        color=colors[i],
        edgecolor="white",
        label=f"{name} ({dur:.1f}s)",
    )
    ax.text(
        start + dur / 2,
        0,
        f"{dur:.1f}s",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
    )

ax.set_yticks([])
ax.set_xlabel("Time (seconds)")
ax.set_title("Sequential Pipeline — Stage Timing Waterfall", fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, result["latency_s"] * 1.15)
plt.tight_layout()
fname = OUTPUT_DIR / "ex6_sequential_waterfall.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore scenario: regulatory filing analysis
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore-listed REIT files quarterly disclosures with
# SGX. The compliance team reviews ~40 peer filings per quarter to
# benchmark disclosure quality. Today they read end-to-end — ~90
# minutes per filing, 60 hours per quarter.
#
# A sequential pipeline reframes this as:
#   Stage 1 (Factual):  extract numeric disclosures (NAV, DPU, gearing)
#   Stage 2 (Interpret): rank which disclosures matter for this quarter
#   Stage 3 (Synthesise): produce a benchmark brief with citations
#
# Why NOT supervisor-worker here: stage 2 needs stage-1's extracted
# numbers before it can rank relevance. Running "factual" and
# "interpret" in parallel would have interpret guessing at claims
# that haven't been extracted yet — the pipeline order is
# substantive, not decorative.
#
# IMPACT: reduces the 60-hour quarterly review to a ~2-hour human
# approval pass over pipeline drafts. At a fully-loaded compliance
# analyst rate of S$120/hour, that is ~S$7,000 saved per quarter per
# analyst, AND the audit trail is structured enough to present to
# SGX if a filing is queried.

print("=" * 70)
print("  SINGAPORE APPLICATION: REIT Peer Disclosure Benchmarking")
print("=" * 70)
print(
    """
  Workload: 40 peer filings / quarter
  Baseline human review:      60 hours / quarter
  Sequential-pipeline assist: 2 hours / quarter (human approval only)
  Time saved:                 ~58 hours / quarter
  Fully-loaded analyst rate:  S$120/hour
  Quarterly savings:          ~S$7,000 per analyst
  Regulator benefit:          structured pipeline trace = SGX-defensible
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
  [x] Sequential multi-agent pipeline (A → B → C)
  [x] When to use sequential vs supervisor-worker fan-out
  [x] Latency model: sequential = sum of stages, parallel = max
  [x] Real downstream dependency: stage 2 consumes stage 1's structured
      output, stage 3 consumes stage 2's
  [x] REIT peer-filing benchmarking as a regulated Singapore use case

  KEY INSIGHT: Sequential is a choice, not a fallback. You pay
  latency to buy reasoning chain — each stage can interpret and
  REJECT upstream output, which fan-out specialists cannot do.

  Next: 03_parallel_router.py — running specialists truly in
  parallel with asyncio.gather, and letting an LLM route queries
  to the right specialist.
"""
)
