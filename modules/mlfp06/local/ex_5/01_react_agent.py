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

from kaizen_agents.agents.specialized.react import ReActAgent

from shared.mlfp06._ollama_bootstrap import OLLAMA_BASE_URL
from shared.mlfp06.ex_5 import (
    MODEL,
    load_hotpotqa,
    make_tools,
    print_tool_registry,
    tool_schemas,
)

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and bind tools
# ════════════════════════════════════════════════════════════════════════

# TODO: Call load_hotpotqa() to get the QA DataFrame
qa_data = ____
# TODO: Call make_tools(qa_data) to bind the tools and receive the tool list
tools = ____
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
#   1. THOUGHT -> 2. ACTION -> 3. OBSERVATION -> repeat -> FINAL ANSWER
#
# Unlike an if-else pipeline, the agent decides WHICH tool and WHAT
# arguments at each step.  The loop is autonomous — no human choreography.
#
# WHY IT MATTERS: Business questions rarely decompose into fixed pipelines.
# ReAct lets one agent handle a whole class of questions without you
# writing a new pipeline for each.  One agent, N questions.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the ReActAgent
# ════════════════════════════════════════════════════════════════════════

# TODO: Instantiate ReActAgent with model=MODEL
#       (kaizen_agents 0.9: tools and budget are set AFTER construction,
#        not via constructor kwargs — see the provided wiring below)
react_agent = ____
# LLM wiring (provided): ReActConfig defaults to provider="openai", so point it
# at the local Ollama daemon. kaizen 2.28 also needs use_async_llm=True for the
# run_async() call in TASK 3. config is mutable post-construction.
react_agent.config.llm_provider = "ollama"
react_agent.config.base_url = OLLAMA_BASE_URL
react_agent.config.use_async_llm = True
react_agent.config.budget_limit_usd = 2.0
for tool in tools:
    react_agent.available_tools.append(tool)
print(f"ReActAgent built:")
print(f"  Model:   {MODEL}")
print(f"  Tools:   {[t.__name__ for t in tools]}")
print(f"  Budget:  $2.00 (hard stop)")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert react_agent is not None, "Task 2: agent should be created"
print("\n✓ Checkpoint 2 passed — ReActAgent ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train (run) the agent on a multi-step task
# ════════════════════════════════════════════════════════════════════════


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
    # TODO: Await react_agent.run(task) to execute the ReAct loop
    result = ____

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
"""
)

# TODO: Call tool_schemas(tools) to generate the JSON Schema descriptors
schemas = ____
print(f"Function-calling schemas ({len(schemas)} tools):")
for s in schemas:
    print(f"  {s['name']:20s} params={list(s['parameters']['properties'].keys())}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert len(schemas) == 4, "Task 4: should generate schemas for all 4 tools"
assert all(
    "name" in s and "parameters" in s for s in schemas
), "Every schema needs name + parameters"
print("\n✓ Checkpoint 4 passed — trace interpretation and schemas visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore banking research analyst
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A private bank in Singapore hires a research desk to answer
# wealth-advisor questions.  Before ReAct: 4 analysts at ~S$120K each
# (S$480K/year) to answer 2,000 questions/year.  With ReActAgent:
# ~$100/year in LLM cost.  5,000x cost reduction, 1-2 days -> 15 sec.
#
# THE RISK: An unconstrained agent can still loop and spend $100 on one
# question.  Technique 2 (02_cost_budget_agent.py) adds the guardrail.

print("=" * 70)
print("  KEY TAKEAWAY: ReActAgent turns a pipeline into a conversation")
print("=" * 70)


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

  Next: 02_cost_budget_agent.py adds the guardrail that prevents a
  looping agent from spending $100 on a $0.05 task...
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
