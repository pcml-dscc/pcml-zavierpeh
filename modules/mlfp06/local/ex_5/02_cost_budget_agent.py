# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 5.2: Cost-Bounded Agents (LLMCostTracker)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Attach a dollar budget to any ReActAgent via max_llm_cost_usd
#   - Watch budget enforcement stop a looping agent gracefully
#   - Understand budget hierarchy: session -> task -> step
#   - Connect agent budgets to PACT governance (Ex 7)
#
# PREREQUISITES: 01_react_agent.py (ReAct loop, tools)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load data + tools
#   2. Build two agents — one with tight budget, one with normal budget
#   3. Run both on an intentionally expensive task
#   4. Visualise: when the budget trips vs when the task completes
#   5. Apply: Singapore SME chatbot cost-safety scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from kaizen_agents.agents.specialized.react import ReActAgent

from shared.mlfp06.ex_5 import MODEL, load_hotpotqa, make_tools

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Data and tools
# ════════════════════════════════════════════════════════════════════════

# TODO: Load HotpotQA and bind tools (see 01_react_agent.py Task 1)
qa_data = ____
tools = ____
print(f"Loaded {qa_data.height} QA examples + {len(tools)} tools\n")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert qa_data.height > 0
assert len(tools) == 4
print("✓ Checkpoint 1 passed — infra ready\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why agents need a financial operating envelope
# ════════════════════════════════════════════════════════════════════════
# A looping agent can burn money fast.  LLMCostTracker is the first line
# of defence:
#
#   Session budget ($50) -> Task budget ($5) -> Step budget ($1)
#
# When the budget is exceeded, the agent is forced to produce a final
# answer and subsequent tool calls are refused.  This is the AGENT layer
# of the envelope — PACT (Ex 7) adds the ORGANISATION layer above it.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build two agents with very different budgets
# ════════════════════════════════════════════════════════════════════════

# TODO: Build a ReActAgent with max_llm_cost_usd=0.10 (intentionally tight)
low_budget_agent = ____
# TODO: Build a ReActAgent with max_llm_cost_usd=2.00 (enough to complete)
normal_agent = ____

print(f"Low-budget agent:    $0.10 (will likely trip)")
print(f"Normal-budget agent: $2.00 (should complete)")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert low_budget_agent is not None
assert normal_agent is not None
print("\n✓ Checkpoint 2 passed — two budgeted agents built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Run both on an intentionally expensive task
# ════════════════════════════════════════════════════════════════════════

EXPENSIVE_TASK = """Perform an exhaustive analysis of the dataset:
1. Get summary statistics
2. Count all question types
3. Count all difficulty levels
4. Search for 10 different topics (churn, fraud, growth, retention, AI,
   governance, policy, climate, risk, regulation)
5. Find the longest documents
6. Synthesise a multi-paragraph report."""


async def run_with_budget(agent: ReActAgent, label: str) -> tuple[str, bool]:
    """Run the agent and return (output_preview, tripped_budget)."""
    print(f"--- {label} ---")
    try:
        # TODO: Await agent.run(EXPENSIVE_TASK)
        result = ____
        output = str(result)[:300] if result else "No output"
        print(f"  Completed. Result preview: {output}...")
        return output, False
    except Exception as e:
        msg = str(e)
        print(f"  Budget tripped: {msg[:200]}")
        return msg, True


async def compare_budgets():
    print("Running tight-budget agent on expensive task...")
    low_output, low_tripped = await run_with_budget(low_budget_agent, "LOW $0.10")

    print("\nRunning normal-budget agent on the same task...")
    normal_output, normal_tripped = await run_with_budget(normal_agent, "NORMAL $2.00")

    return low_output, low_tripped, normal_output, normal_tripped


low_output, low_tripped, normal_output, normal_tripped = asyncio.run(compare_budgets())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert low_output is not None
assert normal_output is not None
print("\n✓ Checkpoint 3 passed — both agents ran to completion or trip\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the outcome table
# ════════════════════════════════════════════════════════════════════════

import polars as pl

outcomes = pl.DataFrame(
    {
        "Agent": ["low_budget", "normal_budget"],
        "Budget USD": [0.10, 2.00],
        "Budget Tripped": [low_tripped, normal_tripped],
        "Output Preview": [
            low_output[:80].replace("\n", " ") + "...",
            normal_output[:80].replace("\n", " ") + "...",
        ],
    }
)
print("=" * 70)
print("  Budget enforcement comparison")
print("=" * 70)
print(outcomes)

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert outcomes.height == 2
print("\n✓ Checkpoint 4 passed — outcome table visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore SME chatbot cost safety
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Singapore SME deploys a support chatbot.  An attacker
# crafts a prompt that loops the agent.  Without a budget: ~$200 per
# attacker session.  With max_llm_cost_usd=$0.25: $0.25 per attacker,
# 800x reduction in denial-of-wallet exposure.  MAS FSM-GL-04 requires
# fraud-resistant cost controls on AI customer interactions.

print("=" * 70)
print("  KEY TAKEAWAY: Every production agent needs a budget")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Attached max_llm_cost_usd to a ReActAgent
  [x] Observed budget enforcement stop a looping agent
  [x] Understood the session -> task -> step budget hierarchy
  [x] Quantified the denial-of-wallet risk for production chatbots
  [x] Chose a budget using the 95th-percentile-times-two rule

  Next: 03_structured_agent.py switches from tool-using ReAct to
  typed structured output via BaseAgent + Signature...
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

# Primary lens: Agent Trace (TAOD capture, tool-call success, stuck-loop
# detection). Secondary: Output (final answer quality).
if False:  # scaffold — requires a live Delegate + API key
    obs = LLMObservatory(delegate=react_agent, run_id="ex_5_agent_run")
    # Re-run the agent under the lens:
    # import asyncio
    # trace = asyncio.run(obs.agent.capture_run(react_agent, task=prompt))
    # obs.output.evaluate(prompts=[prompt], responses=[trace.final_answer])
    print("\n── LLM Observatory Report ──")
    findings = obs.report()

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Agent      (HEALTHY): 5 TAOD steps, tool-call success 1.00,
#       no stuck loops, total cost $0.017 (budget $2.00).
#   [✓] Output     (HEALTHY): judge faithfulness 0.89 on final answer.
#   [?] Retrieval / Alignment / Governance / Attention (n/a)
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [AGENT LENS] 5 TAOD steps for a multi-hop question is the healthy
#     signature — general (data_summary) -> specific (run_query) ->
#     targeted (search_documents) -> grounded (lookup_answer) ->
#     synthesis. The BAD signature would be the same tool called with
#     the same args 3+ times ("stuck loop") or a step count of 1
#     (skipped the tools entirely). The loop detector in AgentDiagnostics
#     flags both.
#     >> Prescription (if stuck): tighten the tool docstrings, the LLM
#        is guessing because the tools don't advertise what they do.
#  [OUTPUT LENS] Faithfulness 0.89 on the final answer confirms the
#     agent's synthesis used the observations rather than fabricating.
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
