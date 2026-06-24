# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.3: Multi-Channel Serving with Nexus + RBAC/JWT
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Register a handler with Nexus for API + CLI + MCP simultaneously
#   - Wrap an async handler in a single-node WorkflowBuilder for Nexus
#   - Validate JWTs via middleware and extract an RBAC role claim
#   - Apply rate limiting as the first line of defence against abuse
#   - Visualise the middleware stack order
#   - Apply multi-channel serving to a Singapore government service bot
#
# PREREQUISITES: Exercise 8.2 (governance pipeline)
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Rebuild the governed stack via build_capstone_stack(engine)
#   2. Wrap serve_qa in a WorkflowBuilder + register with Nexus
#   3. Demonstrate JWT validation and RBAC role extraction
#   4. Apply a sliding-window rate limiter
#   5. Visualise the middleware order and apply to GovTech scenario
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from kailash.workflow.builder import WorkflowBuilder
from nexus import Nexus
from pact import GovernanceEngine, load_org_yaml

from shared.mlfp06.ex_8 import (
    OUTPUT_DIR,
    RateLimiter,
    SimpleJWTAuth,
    build_capstone_stack,
    handle_qa,
    run_async,
    write_org_yaml,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — One handler, three channels
# ════════════════════════════════════════════════════════════════════════
# Nexus is Kailash's multi-channel deployment layer. One registered
# workflow is simultaneously exposed as API + CLI + MCP. The governance
# envelope lives INSIDE the handler the workflow wraps — there is no
# "trusted channel" vs "untrusted channel" divergence.
#
# Nexus registration contract: Nexus.register(name, workflow) where
# the second argument is a BUILT Workflow, not a bare async function.
# We wrap serve_qa in a single-node WorkflowBuilder so the same handler
# runs on every channel.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Rebuild the governed stack
# ════════════════════════════════════════════════════════════════════════

org_path = write_org_yaml()
loaded = load_org_yaml(org_path)
governance_engine = GovernanceEngine(loaded.org_definition)

# TODO: build the 3-tier stack via build_capstone_stack(governance_engine)
agents_by_role, tiers = ____

print("Governed stack rebuilt:")
for tier in tiers:
    print(
        f"  {tier.role:6s} -> budget=${tier.budget_usd:>5.1f}  "
        f"clearance={tier.clearance}"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 1: three tiers should exist"
print("\u2713 Checkpoint 1 passed — governed stack rebuilt\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Register a handler with Nexus for API + CLI + MCP
# ════════════════════════════════════════════════════════════════════════


async def serve_qa(question: str, role: str = "qa") -> dict:
    """The single handler Nexus exposes on all three channels."""
    return await handle_qa(question, role=role, agents_by_role=agents_by_role)


# Smoke-test the handler BEFORE registering — a registration failure
# is then a Nexus issue, not a handler issue.
smoke_result = run_async(serve_qa("What is machine learning?", role="qa"))
print(
    f"\nHandler smoke test: role={smoke_result.get('role')}  "
    f"governed={smoke_result.get('governed')}  "
    f"latency_ms={smoke_result.get('latency_ms', 0):.1f}"
)

# TODO: Wrap the handler in a single-node WorkflowBuilder. Construct
# a WorkflowBuilder and then register the built workflow under the
# name "capstone_serve_qa".
workflow = ____
workflow.add_node(
    "PythonCodeNode",
    "serve_qa_node",
    {
        "code": (
            "# Production body would import and call serve_qa(question, role)\n"
            "result = {'answer': f'[nexus-stub] {question}', 'role': role}\n"
        ),
    },
)

app = Nexus()
app.register("capstone_serve_qa", workflow.build())

print("\nNexus app registered:")
print("  name:     capstone_serve_qa")
print("  wraps:    serve_qa(question, role) — WorkflowBuilder single-node")
print("  channels: API + CLI + MCP (automatic)")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert app is not None, "Task 2: Nexus app should be created"
assert smoke_result.get("governed") is True
print("\u2713 Checkpoint 2 passed — Nexus multi-channel deployment wired\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — JWT validation and RBAC role extraction
# ════════════════════════════════════════════════════════════════════════

print("RBAC + JWT demonstration:")
for token in (
    "token_viewer_001",
    "token_operator_001",
    "token_auditor_001",
    "invalid_token",
):
    # TODO: validate the token via SimpleJWTAuth.validate(token)
    claims = ____
    if claims:
        print(f"  {token[:18]:18s} -> sub={claims['sub']}, role={claims['role']}")
    else:
        print(f"  {token[:18]:18s} -> REJECTED (401)")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert SimpleJWTAuth.validate("token_viewer_001") is not None
assert SimpleJWTAuth.validate("invalid_token") is None
print("\u2713 Checkpoint 3 passed — JWT + RBAC behaves correctly\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Apply a sliding-window rate limiter
# ════════════════════════════════════════════════════════════════════════

# TODO: instantiate RateLimiter with max_requests=5, window_seconds=60
limiter = ____
print("Rate limiter (5 req / 60s) for client_alice:")
for i in range(7):
    allowed = limiter.allow("client_alice")
    print(f"  Request {i + 1}: {'ALLOWED' if allowed else 'RATE LIMITED (429)'}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert not limiter.allow("client_alice"), "6th+ request should be rate-limited"
print("\u2713 Checkpoint 4 passed — rate limiter enforces the window\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise and Apply: Singapore GovTech service bot
# ════════════════════════════════════════════════════════════════════════

middleware_stack = pl.DataFrame(
    {
        "Order": [1, 2, 3, 4, 5, 6],
        "Layer": [
            "CORS",
            "Rate Limit",
            "JWT / RBAC",
            "Request Log",
            "Handler",
            "Governance",
        ],
        "Enforces": [
            "origin whitelist",
            "per-client sliding window",
            "role claim extraction",
            "structured JSON logs",
            "serve_qa()",
            "GovernedSupervisor envelope",
        ],
    }
)
middleware_stack.write_parquet(OUTPUT_DIR / "middleware_stack.parquet")
print("\nMiddleware stack order:")
print(middleware_stack)

# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Latency histogram per channel
# ════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(42)
api_latencies = rng.lognormal(mean=5.5, sigma=0.4, size=200)
cli_latencies = rng.lognormal(mean=4.8, sigma=0.3, size=200)
mcp_latencies = rng.lognormal(mean=5.2, sigma=0.5, size=200)

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(api_latencies, bins=30, alpha=0.6, color="#e74c3c", label="API (HTTP)")
ax.hist(cli_latencies, bins=30, alpha=0.6, color="#3498db", label="CLI (local)")
ax.hist(mcp_latencies, bins=30, alpha=0.6, color="#2ecc71", label="MCP (protocol)")
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Frequency")
ax.set_title("Per-Channel Latency Distribution (Simulated)", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex8_channel_latency.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# SCENARIO: Singapore GovTech ships ONE internal policy assistant used
# by ~15,000 public servants. JWTs carry a role claim ("viewer",
# "operator", "auditor"). Deploy consolidation: 3 codebases -> 1.

print("\n" + "=" * 70)
print("  APPLY — GovTech Multi-Ministry Policy Assistant")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Wrapped an async handler in a single-node WorkflowBuilder
  [x] Registered the workflow with Nexus — API + CLI + MCP at once
  [x] Validated JWTs and extracted RBAC role claims
  [x] Rate-limited per-client traffic with a sliding window
  [x] Visualised the middleware stack order
  [x] Applied multi-channel serving to a GovTech scenario

  Next: 04_drift_monitoring.py adds production drift monitoring.
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

# Primary lens: ALL SIX — the capstone wires Align + Kaizen + PACT +
# Nexus + RAG + Agents end-to-end, so every lens should be lit.
if False:  # scaffold — requires the full capstone stack
    obs = LLMObservatory(run_id="ex_8_capstone_run")
    # obs.output.evaluate(prompts=[...], responses=[...])
    # obs.retrieval.evaluate(queries=[...], retrieved_contexts=[...], answers=[...])
    # for run_id, trace in supervisor.all_traces.items():
    #     obs.agent.register_trace(trace)
    # obs.alignment.log_training_step(...)
    # obs.governance.verify_chain(audit_df)
    print("\n── LLM Observatory Report ──")
    findings = obs.report()
    # obs.plot_dashboard().show()  # all six panels at once

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad (CAPSTONE)
# ════════════════════════════════════════════════════════════════
#   [✓] Output     (HEALTHY): faithfulness 0.88, judge coherence 0.91
#   [✓] Retrieval  (HEALTHY): recall@5 = 0.79, context util 0.72
#   [✓] Agent      (HEALTHY): 14 TAOD steps, no stuck loops, cost $0.04
#   [✓] Alignment  (HEALTHY): KL 0.6 nats, win-rate 0.61 vs base
#   [!] Governance (WARNING): 1 of 8 drills escalated; budget at 71%
#       Fix: raise escalation threshold or narrow data_access envelope.
#   [?] Attention  (UNKNOWN): API-only judge/prod model — enable the
#       open-weight evaluator to light up this panel.
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [CAPSTONE COMPOSITE] The capstone is the first exercise where you
#     see the full six-lens dashboard. Five lenses GREEN + one YELLOW
#     is a realistic "ship it with a watch-item" disposition. The
#     governance WARNING is the escalation on 1/8 drills — investigate
#     which drill escalated before production rollout; that's exactly
#     the kind of pre-deploy check the dashboard is designed for.
#  [CROSS-LENS READING] Notice how each lens is answering a different
#     question: Output says "is the answer good?"; Retrieval says "did
#     we give it the right context?"; Agent says "did it use the right
#     steps?"; Alignment says "is the fine-tune pulling its weight?";
#     Governance says "did we stay inside the envelope?". A single
#     aggregate "quality score" would hide all of this.
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
