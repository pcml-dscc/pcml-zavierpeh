# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 5.1: ReActAgent — Tool-Using Autonomous Reasoning
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a ReActAgent with the Thought -> Action -> Observation loop
#   - Hand the agent tools with structured docstrings (tools as API)
#   - Run multi-step analysis where the LLM chooses the tool order
#   - Inspect and interpret the reasoning trace for quality signals
#   - Understand function-calling protocol (auto / required / specific)
#
# PREREQUISITES: MLFP06 Ex 1-4 (Delegate, Signature, prompt engineering)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Load HotpotQA multi-hop dataset + bind tools
#   2. Build a ReActAgent with a cost budget
#   3. Run a multi-step analysis task
#   4. Visualise the reasoning trace
#   5. Apply: Singapore banking research analyst scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import matplotlib.pyplot as plt

from kaizen_agents.agents.specialized.react import ReActAgent

from shared.mlfp06._ollama_bootstrap import OLLAMA_BASE_URL
from shared.mlfp06.ex_5 import (
    MODEL,
    OUTPUT_DIR,
    load_hotpotqa,
    make_tools,
    print_tool_registry,
    tool_schemas,
)

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and bind tools
# ════════════════════════════════════════════════════════════════════════

qa_data = load_hotpotqa()
tools = make_tools(qa_data)
print_tool_registry(tools)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert qa_data.height > 0, "Task 1: dataset should not be empty"
assert len(tools) == 4, "Task 1: should have 4 tools bound to qa_data"
assert all(callable(t) for t in tools), "All tools should be callable"
assert all(
    t.__doc__ for t in tools
), "All tools need docstrings — the agent reads them as API documentation"
print("\n✓ Checkpoint 1 passed — 4 tools registered with HotpotQA\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — The ReAct Loop
# ════════════════════════════════════════════════════════════════════════
# ReAct = Reasoning + Acting.  The agent interleaves thinking and doing:
#
#   1. THOUGHT: "I need to understand the dataset first."
#   2. ACTION:  data_summary(dataset_name="qa_data")
#   3. OBSERVATION: <tool output: 500 rows, 5 columns, avg_len=1832...>
#   4. THOUGHT: "Now I know columns.  Let me count question types."
#   5. ACTION:  run_query("count question types")
#   6. OBSERVATION: <comparison: 243, bridge: 257>
#   7. THOUGHT: "I have enough — let me synthesise the answer."
#   8. FINAL ANSWER: <synthesised response>
#
# Unlike an if-else pipeline, the agent decides WHICH tool and WHAT
# arguments at each step.  The loop is autonomous — no human choreography.
#
# ANALOGY: A research analyst with a filing cabinet.  You ask "which
# clients are at risk of churn?"  The analyst doesn't follow a script.
# They open the cabinet, pull a summary, realise they need the activity
# log, pull that, cross-reference — each step informed by the last.
# ReAct is that analyst, but the filing cabinet is your tool list.
#
# WHY IT MATTERS: Business questions rarely decompose into fixed pipelines.
# ReAct lets one agent handle a whole class of questions without you
# writing a new pipeline for each.  One agent, N questions.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the ReActAgent
# ════════════════════════════════════════════════════════════════════════

# kaizen_agents 0.9: ReActAgent no longer accepts `tools=` or budget
# kwargs in the constructor. Tools are registered post-construction
# via agent.available_tools, and budget via agent.config.budget_limit_usd.
# This separation is intentional — tool registration and budget are
# configuration concerns, not identity concerns.
react_agent = ReActAgent(model=MODEL)
# ReActConfig defaults to provider="openai" and reads KAIZEN_* env vars, so wire
# it to the local Ollama daemon explicitly. config is mutable post-construction
# (same pattern as budget_limit_usd below). kaizen 2.28: run_async() also needs
# use_async_llm=True or it raises "Agent not configured for async mode".
react_agent.config.llm_provider = "ollama"
react_agent.config.base_url = OLLAMA_BASE_URL
react_agent.config.use_async_llm = True
react_agent.config.budget_limit_usd = 2.0
for tool in tools:
    react_agent.available_tools.append(tool)
print(f"ReActAgent built:")
print(f"  Model:   {MODEL}")
print(f"  Tools:   {[t.__name__ for t in tools]}")
print(f"  Budget:  ${react_agent.config.budget_limit_usd:.2f} (hard stop)")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert react_agent is not None, "Task 2: agent should be created"
print("\n✓ Checkpoint 2 passed — ReActAgent ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train (run) the agent on a multi-step task
# ════════════════════════════════════════════════════════════════════════
# "Train" here means exercise the agent end-to-end — LLM agents are not
# gradient-trained at runtime; the "training" is the reasoning trajectory.


async def run_multi_step_analysis() -> object:
    """Let the ReActAgent answer a multi-step HotpotQA research question."""
    sample_q = qa_data["question"][0]
    task = f"""Analyse the HotpotQA multi-hop reasoning dataset to understand
its structure and answer the question: "{sample_q}"

Steps:
1. Get a dataset summary to understand the columns and types
2. Count question types (comparison vs bridge) and difficulty levels
3. Search for documents relevant to the question above
4. Look up the ground-truth answer
5. Synthesise your findings into a clear report."""

    print(f"Task: {task[:200]}...\n")
    # kaizen_agents 0.9: ReActAgent.run() is synchronous;
    # use run_async() for the async path.
    result = await react_agent.run_async(task=task)

    if hasattr(result, "content"):
        output = result.content
    elif isinstance(result, str):
        output = result
    else:
        output = str(result)
    print(f"Agent output (first 500 chars):\n{output[:500]}...")
    return result


analysis_result = asyncio.run(run_multi_step_analysis())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert analysis_result is not None, "Task 3: analysis should produce a result"
print("\n✓ Checkpoint 3 passed — multi-step analysis complete\n")

# INTERPRETATION: The agent autonomously sequenced the tool calls.  A good
# agent goes general -> specific (summary, then query, then search).  A
# poor agent repeats the same call or skips to the answer without looking
# at the data — a signal the prompt or tools need work.


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the reasoning trace + function-calling protocol
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  Trace inspection — what quality looks like")
print("=" * 70)
print(
    """
Good trace signals:
  ✓ Logical step order (general → specific)
  ✓ No redundant tool calls
  ✓ Arguments match the tool schema
  ✓ Final answer synthesises ALL observations, not just the last one

Bad trace signals:
  ✗ Random tool ordering
  ✗ Same tool called twice with identical args
  ✗ Arguments that don't match schema
  ✗ Final answer ignores some observations
"""
)

# Visualise the structured JSON schemas the agent actually sees.
schemas = tool_schemas(tools)
print(f"Function-calling schemas ({len(schemas)} tools):")
for s in schemas:
    print(f"  {s['name']:20s} params={list(s['parameters']['properties'].keys())}")

print(
    """
Function-calling protocol (tool_choice):
  auto      — model decides whether to call a tool or respond directly
  required  — model MUST call at least one tool (force data grounding)
  specific  — pin to one named function (pipeline step)

Parallel calls — some models fire multiple tools per turn, e.g.:
  Turn 1: [search_documents("churn"), run_query("count types")]
Both execute simultaneously and results return together.
"""
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert len(schemas) == 4, "Task 4: should generate schemas for all 4 tools"
assert all(
    "name" in s and "parameters" in s for s in schemas
), "Every schema needs name + parameters"
print("✓ Checkpoint 4 passed — trace interpretation and schemas visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore banking research analyst
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A private bank in Singapore hires a research desk to answer
# wealth-advisor questions like "which regions had the most regulatory
# changes last quarter?"  Each question decomposes into search, filter,
# count, and synthesis — the same shape as the HotpotQA task above.
#
# BEFORE REACT: The bank hires 4 research analysts at ~S$120K/year each
# (S$480K/year) to answer ~2,000 such questions per year.  That's
# ~S$240 per answer, 1-2 day turnaround, and answers go stale before
# advisors use them.
#
# WITH REACT: A single ReActAgent with the bank's internal search,
# summary, and lookup tools answers the same questions in ~15 seconds.
# LLM cost at ~$0.05 per task * 2,000 = $100/year.  Advisors get
# real-time answers during client calls.
#
# BUSINESS IMPACT:
#   - Research cost:      S$480,000 -> S$100  (~5,000x reduction)
#   - Latency:            1-2 days -> 15 seconds
#   - Advisor experience: stale reports -> live conversation support
#   - Analysts pivot to   higher-value qualitative work (fund manager
#                         interviews, thesis development)
#
# THE RISK: An unconstrained agent can still loop and spend $100 on one
# question.  Technique 2 (02_cost_budget_agent.py) adds the guardrail.


print("=" * 70)
print("  KEY TAKEAWAY: ReActAgent turns a pipeline into a conversation")
print("=" * 70)
print(
    """
  Before: one pipeline per question shape, brittle and expensive.
  After:  one agent + N tools, answers any question that composes them.

  The tool docstring is now your most important artifact.  It's the
  contract the LLM reads to decide what to do next.  Precise tool docs
  = accurate agents.  Vague tool docs = wrong tool, wrong arguments,
  wasted budget.
"""
)


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — Agent reasoning step profile
# ════════════════════════════════════════════════════════════════════════

questions = ["Q1: Multi-hop", "Q2: Comparison", "Q3: Bridge"]
step_counts = [5, 3, 4]
latencies_s = [12.4, 8.1, 10.7]

fig, ax1 = plt.subplots(figsize=(8, 4))
x = range(len(questions))
bars = ax1.bar(x, step_counts, color="#2196F3", alpha=0.8, label="Reasoning steps")
ax1.set_ylabel("Reasoning Steps", color="#2196F3")
ax1.set_xticks(x)
ax1.set_xticklabels(questions, rotation=15, ha="right")

ax2 = ax1.twinx()
ax2.plot(x, latencies_s, "o-", color="#FF5722", linewidth=2, label="Latency (s)")
ax2.set_ylabel("Latency (s)", color="#FF5722")

ax1.set_title("ReAct Agent: Steps & Latency per Question Type")
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_react_steps.png", dpi=150)
plt.close(fig)
print(f"\nSaved: {OUTPUT_DIR / '01_react_steps.png'}")


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
  [x] Built a ReActAgent with tools and a cost budget
  [x] Ran a multi-step task where the LLM chose the tool order
  [x] Inspected the reasoning trace for quality signals
  [x] Understood function-calling protocol and parallel calls
  [x] Mapped the technique to a Singapore private-banking use case

  KEY INSIGHT: Agents are LLMs with the ability to call functions.
  No new AI — just LLMs that observe and act instead of just responding.
  The novelty is that YOU design the tool surface; the LLM orchestrates.

  Next: 02_cost_budget_agent.py adds the guardrail that prevents a
  looping agent from spending $100 on a $0.05 task...
"""
)
