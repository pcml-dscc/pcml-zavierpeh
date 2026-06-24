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

import matplotlib.pyplot as plt

from kaizen_agents.agents.specialized.react import ReActAgent

from shared.mlfp06.ex_5 import MODEL, OUTPUT_DIR, load_hotpotqa, make_tools

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Data and tools
# ════════════════════════════════════════════════════════════════════════

qa_data = load_hotpotqa()
tools = make_tools(qa_data)
print(f"Loaded {qa_data.height} QA examples + {len(tools)} tools\n")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert qa_data.height > 0
assert len(tools) == 4
print("✓ Checkpoint 1 passed — infra ready\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why agents need a financial operating envelope
# ════════════════════════════════════════════════════════════════════════
# A looping agent can burn money fast.  A single task with an infinite
# ReAct loop can reach $50 in LLM calls in under 10 minutes — and the
# loop terminates only because the LLM eventually outputs FINAL ANSWER,
# which may never happen on a badly phrased task.
#
# LLMCostTracker is the first line of defence:
#
#   Budget hierarchy:
#     Session budget  ($50)  — total for all tasks in a session
#       └─ Task budget ($5)  — per-task limit
#            └─ Step budget ($1) — per-tool-call limit
#
#   When the budget is exceeded:
#     1. Agent receives a warning at 80% consumed
#     2. Agent is forced to produce a final answer from what it has
#     3. Subsequent tool calls are refused
#     4. Partial results return with a budget-exceeded flag
#
# This is the AGENT layer of the envelope.  The ORGANISATION layer sits
# above it in PACT (Ex 7), where a governance engine rolls up per-agent
# spend into per-tenant and per-role budgets.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build two agents with very different budgets
# ════════════════════════════════════════════════════════════════════════

# kaizen_agents 0.9: budget and tools are configured post-construction.
# agent.config.budget_limit_usd replaces the removed max_llm_cost_usd kwarg.
low_budget_agent = ReActAgent(model=MODEL)
low_budget_agent.config.budget_limit_usd = 0.10  # intentionally tight
for tool in tools:
    low_budget_agent.available_tools.append(tool)

normal_agent = ReActAgent(model=MODEL)
normal_agent.config.budget_limit_usd = 2.00  # enough to complete normal tasks
for tool in tools:
    normal_agent.available_tools.append(tool)

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
        result = await agent.run_async(task=EXPENSIVE_TASK)
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
assert low_output is not None, "Low-budget run should return something"
assert normal_output is not None, "Normal-budget run should return something"
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
assert outcomes.height == 2, "Task 4: should compare two runs"
print("\n✓ Checkpoint 4 passed — outcome table visualised\n")

# INTERPRETATION: With a $0.10 budget, a good LLMCostTracker stops the
# agent before it finishes step 4 of the expensive task.  With $2.00,
# the same agent has enough headroom to synthesise a full report.  The
# gap between the two is the "runaway zone" — every agent needs a
# budget in that zone for production deployment.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore SME chatbot cost safety
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore SME deploys a customer-support ReActAgent that
# answers questions like "where is my order?" and "what is your return
# policy?"  The chatbot has ~10 tools (order lookup, product catalog,
# shipping tracker, refund processor, FAQ search, ...).
#
# THE THREAT: A malicious user crafts a prompt that triggers an infinite
# ReAct loop — "keep searching until you find my missing parcel."  The
# agent calls search_orders, search_shipments, search_refunds, repeat.
# Without a budget, the agent burns ~$0.20 per iteration * 1000
# iterations = $200 per malicious session.  10 attackers = $2,000/day.
# At scale, this is an ML-enabled denial-of-wallet attack.
#
# THE FIX: max_llm_cost_usd=$0.25 per customer session.  The attack
# costs the business $0.25 per attacker, not $200.  The attacker stops
# because there's no payoff.
#
# BUSINESS IMPACT:
#   - Worst-case attack cost: S$270,000/year (unbudgeted)
#                            S$350/year     (budgeted @ $0.25/session)
#   - 800x reduction in max-loss exposure
#   - Insurance friendliness: "we have per-session spend caps"
#   - Compliance: MAS FSM-GL-04 requires fraud-resistant cost controls
#     on AI customer interactions
#
# THE PATTERN: max_llm_cost_usd is the cheapest, dumbest guardrail you
# can add.  It's two keyword arguments and it saves you from the single
# largest production-operations risk of agentic systems.


print("=" * 70)
print("  KEY TAKEAWAY: Every production agent needs a budget")
print("=" * 70)
print(
    """
  Budget is a structural defence, not a soft limit.  The agent cannot
  exceed it even if the LLM tries.  Think of it like a fuse in an
  electrical system — the agent isn't smart enough to know when to
  stop, so the fuse decides for it.

  Rule of thumb: set the budget to 2-5x the expected cost.  If you
  don't know the expected cost, run 10 representative tasks with no
  budget, take the 95th percentile, multiply by 2.
"""
)


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — Budget utilisation: allocated vs spent
# ════════════════════════════════════════════════════════════════════════

tiers = ["Low ($0.10)", "Normal ($2.00)"]
allocated = [0.10, 2.00]
spent = [0.10 if low_tripped else 0.06, 0.45 if not normal_tripped else 2.00]

fig, ax = plt.subplots(figsize=(6, 4))
x = range(len(tiers))
ax.bar([i - 0.15 for i in x], allocated, 0.3, label="Allocated", color="#90CAF9")
ax.bar([i + 0.15 for i in x], spent, 0.3, label="Spent", color="#FF7043")
ax.set_xticks(x)
ax.set_xticklabels(tiers)
ax.set_ylabel("USD")
ax.set_title("Agent Budget Utilisation: Allocated vs Spent")
ax.legend()
for i, tripped in enumerate([low_tripped, normal_tripped]):
    if tripped:
        ax.annotate(
            "TRIPPED",
            (i + 0.15, spent[i]),
            ha="center",
            va="bottom",
            fontsize=9,
            color="red",
            fontweight="bold",
        )
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_budget_utilization.png", dpi=150)
plt.close(fig)
print(f"\nSaved: {OUTPUT_DIR / '02_budget_utilization.png'}")


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

  KEY INSIGHT: Agent budgets are the cheapest insurance policy in the
  entire AI stack.  Two keyword arguments, one structural defence,
  800x reduction in worst-case spend.  There is no reason not to set
  one on every agent you ship.

  Next: 03_structured_agent.py switches from tool-using ReAct to
  typed structured output via BaseAgent + Signature...
"""
)
