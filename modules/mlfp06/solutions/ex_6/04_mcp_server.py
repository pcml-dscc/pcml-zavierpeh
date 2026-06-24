# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 6.4: MCP Server — Exposing Tools to External Agents
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Understand Model Context Protocol (MCP) as a standard way to
#     expose tools to any MCP-compatible agent
#   - Define MCP tools with JSON-schema parameters and typed handlers
#   - Register tools on an MCPServer and inspect its capability surface
#   - Understand the transport split: stdio (local) vs HTTP/SSE (remote)
#
# PREREQUISITES: 03_parallel_router.py
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load the shared corpus (tools will search it)
#   2. Define three MCP tool handlers (analyse_passage, search_corpus, stats)
#   3. Register each handler on an MCPServer with the @server.tool decorator,
#      attaching a JSON input schema + tool annotations
#   4. Inspect the server's capability surface and verify 3 tools registered
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from kailash_mcp import MCPServer, StructuredTool, ToolAnnotation

from shared.mlfp06.ex_6 import OUTPUT_DIR, load_squad_corpus


# ════════════════════════════════════════════════════════════════════════
# THEORY — Model Context Protocol
# ════════════════════════════════════════════════════════════════════════
# MCP is a small protocol that lets an AI agent discover and call
# tools exposed by a server. A "tool" is a function plus a JSON
# schema describing its parameters plus a natural-language
# description. Any MCP-compatible agent — Claude, GPT, a Kaizen
# agent, an IDE plugin — can list the server's tools and invoke
# them.
#
# Non-technical analogy: MCP is USB for AI agents. Your specialists
# are devices; MCP is the plug. Any agent that speaks MCP can
# discover what's attached and use it, without custom glue code
# per agent type.
#
# COMPONENTS:
#   - Server:    registers tools with schemas, listens on a transport
#   - Transport: stdio for local subprocesses, HTTP/SSE for remote
#   - Tool:      handler function + JSON schema + description
#   - Resource:  read-only data the agent can access
#
# HOW YOU REGISTER A TOOL (kailash-mcp):
#   MCPServer exposes a `@server.tool(...)` decorator. You decorate a
#   plain Python function; kailash-mcp reads its name, docstring, and
#   type hints to build the tool's capability card. To describe the
#   parameters precisely, pair it with `@structured_tool(input_schema=...)`
#   — a JSON Schema — and a `ToolAnnotation` that records hints such as
#   "read-only" or "estimated duration" for the calling agent.
#
# WHY THIS MATTERS FOR MULTI-AGENT:
# MCP lets your agents share tools without hard-coding imports. A
# Kaizen supervisor can call an MCP tool exposed by a Python
# service; a Claude Desktop agent can call the same tool; an IDE
# plugin can call it. One registration, many consumers.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load the corpus (tools will operate on it)
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Load SQuAD corpus for MCP tools")
print("=" * 70)

passages = load_squad_corpus()
print(
    f"Corpus: {passages.height} passages across "
    f"{passages['title'].n_unique()} titles"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert passages.height > 0
print("✓ Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Create the server, then define + register three tool handlers
# ════════════════════════════════════════════════════════════════════════
# In kailash-mcp you do not build separate "tool" objects and hand them to
# the server — you register a plain function ON the server with the
# @server.tool() decorator. So the server comes first. We declare it on the
# stdio transport (the default for local subprocess agents); HTTP/SSE is a
# constructor flag away when you need remote agents.

print("=" * 70)
print("TASK 2: Create MCPServer and register three tools")
print("=" * 70)

mcp_server = MCPServer(name="mlfp06-analysis-server", transport="stdio")

# A JSON Schema describes each tool's parameters so any MCP-compatible agent
# knows how to call it. ToolAnnotation records behavioural hints (read-only,
# rough latency) that help an agent decide whether/when to call the tool.
ANALYSE_SCHEMA = {
    "type": "object",
    "properties": {
        "passage": {"type": "string", "description": "Text to analyse"},
        "analysis_type": {
            "type": "string",
            "enum": ["factual", "semantic", "structural"],
        },
    },
    "required": ["passage"],
}
SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query"},
        "top_k": {"type": "integer", "description": "Max results", "default": 3},
    },
    "required": ["query"],
}
STATS_SCHEMA = {"type": "object", "properties": {}}

READ_ONLY = ToolAnnotation(is_read_only=True, estimated_duration=0.1)


@mcp_server.tool()
def analyse_passage(passage: str, analysis_type: str = "factual") -> str:
    """Analyse a text passage from a factual, semantic, or structural perspective.

    Args:
        passage: The text to analyse.
        analysis_type: One of 'factual', 'semantic', 'structural'.

    Returns:
        Analysis results as formatted text.
    """
    return (
        f"Analysis ({analysis_type}) of {len(passage)}-char passage: "
        f"[would run {analysis_type}_agent against the passage]"
    )


@mcp_server.tool()
def search_corpus(query: str, top_k: int = 3) -> str:
    """Search the document corpus for passages matching a query.

    Args:
        query: Search query text.
        top_k: Maximum results to return.

    Returns:
        Matching passages with their titles.
    """
    query_lower = query.lower()
    scored = []
    for row in passages.iter_rows(named=True):
        score = sum(1 for w in query_lower.split() if w in row["text"].lower())
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [f"[{row['title']}] {row['text'][:200]}..." for _, row in scored[:top_k]]
    return "\n\n".join(results) if results else "No matches found."


@mcp_server.tool()
def get_corpus_stats() -> str:
    """Get statistics about the available document corpus.

    Returns:
        Corpus statistics including size, topics, and coverage.
    """
    return (
        f"Corpus: {passages.height} passages, "
        f"{passages['title'].n_unique()} unique topics\n"
        f"First 10 topics: {passages['title'].unique().to_list()[:10]}"
    )


# The @server.tool decorator returns the function unchanged, so each handler
# is BOTH registered on the server AND directly callable here for testing.
# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert get_corpus_stats().startswith("Corpus:")
assert "[" in search_corpus("what", top_k=2) or "No matches" in search_corpus(
    "xyzxyz", top_k=2
)
print("✓ Checkpoint 2 passed — 3 handlers registered and callable\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Attach JSON schemas as StructuredTool capability cards
# ════════════════════════════════════════════════════════════════════════
# A StructuredTool bundles the input schema + annotation into one object —
# the "capability card" an agent reads before deciding to call the tool.
# Here we build one card per registered tool so we can list the full surface.

print("=" * 70)
print("TASK 3: Build StructuredTool capability cards (schemas + annotations)")
print("=" * 70)

tool_cards = {
    "analyse_passage": StructuredTool(
        input_schema=ANALYSE_SCHEMA, annotations=READ_ONLY
    ),
    "search_corpus": StructuredTool(input_schema=SEARCH_SCHEMA, annotations=READ_ONLY),
    "get_corpus_stats": StructuredTool(
        input_schema=STATS_SCHEMA, annotations=READ_ONLY
    ),
}

for name, card in tool_cards.items():
    handler = mcp_server.get_tool_by_name(name)
    desc = (handler["original_function"].__doc__ or "").strip().splitlines()[0]
    print(f"  {name}: {desc}")
    print(f"      params: {list(card.input_schema['properties'])}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert len(tool_cards) == 3
assert all(isinstance(c, StructuredTool) for c in tool_cards.values())
print("\n✓ Checkpoint 3 passed — 3 capability cards defined\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Inspect the server's capability surface
# ════════════════════════════════════════════════════════════════════════
# The server already holds every tool we decorated. We never start a live
# listener here — we introspect the registered surface, which is exactly
# what an agent's `list_tools` call returns over the wire.

print("=" * 70)
print("TASK 4: MCP Server — inspect and verify capability surface")
print("=" * 70)

stats = mcp_server.get_server_stats()
registered_names = sorted(mcp_server._tool_registry.keys())

print(f"Server: {mcp_server.name}")
print(f"Registered tools ({stats['tools']['registered_tools']}): {registered_names}")
print("\nTransport options:")
print("  stdio:    for local subprocess agent connections")
print("  HTTP/SSE: for remote/network agent connections")

print("\nTool test — get_corpus_stats():")
print(get_corpus_stats())

trace_path = OUTPUT_DIR / "ex6_mcp_server_trace.txt"
trace_path.write_text(
    f"Server: {mcp_server.name}\n"
    f"Tools: {registered_names}\n"
    f"\n{get_corpus_stats()}\n"
)
print(f"\nTrace written to: {trace_path}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert mcp_server is not None
assert stats["tools"]["registered_tools"] == 3
assert registered_names == ["analyse_passage", "get_corpus_stats", "search_corpus"]
print("\n✓ Checkpoint 4 passed — MCP server exposes 3 tools\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Tool call frequency bar chart
# ════════════════════════════════════════════════════════════════════════
# In production, an MCP server logs every tool invocation. This chart
# simulates what that dashboard looks like — showing which tools are
# most frequently called. Skewed distributions reveal over-reliance
# on a single tool (risk) or unused tools (dead code).

tool_names = registered_names
# Simulated production call counts (realistic distribution: search >> stats).
# Order matches registered_names: analyse_passage, get_corpus_stats, search_corpus.
simulated_calls = [45, 15, 120]

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#3498db", "#2ecc71", "#e67e22"]
bars = ax.bar(tool_names, simulated_calls, color=colors)
ax.set_ylabel("Invocations (simulated 24h)")
ax.set_title("MCP Tool Call Frequency — Production Dashboard", fontweight="bold")
for bar, count in zip(bars, simulated_calls):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        count + 2,
        str(count),
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
ax.set_ylim(0, max(simulated_calls) * 1.15)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex6_mcp_tool_frequency.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore scenario: shared analysis tools across MAS teams
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore financial institution runs three separate
# AI teams — retail credit, corporate credit, and AML/transaction
# monitoring. Each team has built its own ad-hoc "analyse document"
# and "search knowledge base" helpers, each wired to a different
# LLM framework. When the bank wants to reuse retail credit's
# "analyse doc" tool from the AML team, engineering cost is ~3
# person-weeks per integration because every pair of teams invents
# its own glue code.
#
# MCP replaces the glue: retail credit exposes its tools on an
# MCPServer once. AML points its Kaizen supervisor at the server
# and lists tools at runtime. Corporate credit points a Claude
# Desktop workflow at the same server. One registration, three
# consumers, zero custom glue.
#
# IMPACT:
#   Integrations needed per new tool:    3 teams × 3 pairs = 9
#   Legacy glue cost per integration:    ~3 person-weeks
#   Legacy total per new tool:           ~27 person-weeks
#   MCP cost per new tool:               ~1 person-week (one registration)
#   Savings per tool:                    ~26 person-weeks × S$3,500/week
#                                        ≈ S$91K per tool shared
#   Regulatory bonus: MAS TRM audit sees ONE set of tool schemas,
#   ONE audit log, ONE access-control surface — not nine.

print("=" * 70)
print("  SINGAPORE APPLICATION: Shared MCP Tools Across Bank AI Teams")
print("=" * 70)
print(
    """
  Teams: retail credit, corporate credit, AML monitoring
  Legacy integration cost per new tool:  ~27 person-weeks
  MCP integration cost per new tool:     ~1 person-week
  Savings per shared tool:                ~S$91K
  MAS TRM audit surface:                  1 (schemas, logs, ACLs) — not 9
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
  [x] MCP as the "USB for AI agents": one server, many consumers
  [x] @server.tool registration: handler + JSON schema + annotation
  [x] StructuredTool capability cards and server introspection
  [x] Transport trade-off: stdio (local) vs HTTP/SSE (remote)
  [x] Singapore bank scenario: MCP collapses N×M glue to 1×N
      registrations

  KEY INSIGHT: The moment you expose a tool via MCP, its capability
  card becomes discoverable by EVERY MCP-compatible agent — that is
  the leverage point. Don't write tools that only your own supervisor
  can call. Write tools that any agent can call.

  Next: 05_memory_and_security.py — wiring agent memory (short-term,
  long-term, entity) and guarding against multi-agent attack patterns.
"""
)
