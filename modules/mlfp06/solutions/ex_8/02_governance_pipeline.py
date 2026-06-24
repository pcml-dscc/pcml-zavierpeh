# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.2: Build a Governed Agent Pipeline with PACT
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compile a PACT organisation YAML into a live GovernanceEngine
#   - Wrap a Kaizen BaseAgent in GovernedSupervisor at multiple trust tiers
#   - Configure D/T/R (Delegator/Task/Responsible) chains for three roles
#   - Visualise the governance envelope hierarchy
#   - Apply governed agents to a Singapore financial advisory scenario
#
# PREREQUISITES: Exercise 8.1 (adapter loading), MLFP06 Ex 7 (PACT intro)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Compile the PACT org YAML into a GovernanceEngine
#   2. Build the shared capstone stack via ``build_capstone_stack(engine)``
#   3. Inspect the three GovernedSupervisor tiers (qa / admin / audit)
#   4. Visualise the envelope hierarchy
#   5. Apply to a Singapore financial advisory (MAS TRM) scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from pact import GovernanceEngine, NodeType, load_org_yaml

from shared.mlfp06.ex_8 import (
    OUTPUT_DIR,
    build_capstone_stack,
    write_org_yaml,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — D/T/R and Operating Envelopes
# ════════════════════════════════════════════════════════════════════════
# PACT governance is built on three ideas:
#
#   Delegator (D) — the human or role who authorises a task
#   Task        (T) — what can be done (bounded scope)
#   Responsible (R) — the agent carrying out the task
#
# Every agent action is traceable back to a human Delegator. There is
# no "rogue agent" path. Each D/T/R chain carries an operating envelope:
# a budget (max $/request), an allowed action list, and a confidentiality
# clearance. The GovernanceEngine's default is fail-closed: if a request
# falls outside any envelope, it is rejected, not "best-effort handled".
#
# In modern PACT the wrapper is `GovernedSupervisor` from kaizen_agents.
# It takes budget, tools, and clearance, and builds a proper
# 5-dimensional ConstraintEnvelope internally. `build_capstone_stack`
# deduplicates the 3-tier construction block so every capstone
# technique file wires the same stack the same way.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Compile the org YAML into a GovernanceEngine
# ════════════════════════════════════════════════════════════════════════

org_path = write_org_yaml()
loaded = load_org_yaml(org_path)
governance_engine = GovernanceEngine(loaded.org_definition)
compiled_org = governance_engine.get_org()
# In modern pact, every node is a NodeType.DEPARTMENT | NodeType.TEAM | NodeType.ROLE.
# "Agents" are ROLE nodes whose address contains a team ("-T<n>" segment) — i.e. a
# Responsible sitting under a team under a department. Department heads are ROLE
# nodes WITHOUT a "-T" segment in their address (e.g. "D1-R1").
n_agents = sum(
    1
    for n in compiled_org.nodes.values()
    if n.node_type == NodeType.ROLE and "-T" in n.address and not n.is_vacant
)
n_roles = sum(1 for n in compiled_org.nodes.values() if n.node_type == NodeType.ROLE)
n_departments = sum(
    1 for n in compiled_org.nodes.values() if n.node_type == NodeType.DEPARTMENT
)
print(
    f"Compiled org: {n_departments} departments, "
    f"{n_roles} roles total ({n_agents} responsible agents)"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert n_roles >= 3, "Task 1: compiled org should carry 3+ Responsible roles"
print("\u2713 Checkpoint 1 passed — governance compiled\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the shared 3-tier governed stack
# ════════════════════════════════════════════════════════════════════════
#
# `build_capstone_stack(engine)` attaches a `ConstraintEnvelopeConfig`
# to each Responsible role (qa / admin / audit) and returns a
# `GovernedSupervisor` for each — plus the tier metadata so this file
# can read budgets / tools / clearance without re-deriving them.

agents_by_role, tiers = build_capstone_stack(governance_engine)
print("Governed agent tiers (from build_capstone_stack):")
for tier in tiers:
    gs = agents_by_role[tier.role]
    env = gs.envelope
    print(
        f"  {tier.role:6s} -> {tier.address:14s}  "
        f"budget=${env.financial.max_spend_usd:>5.1f}  "
        f"clearance={env.confidentiality_clearance.name:<12s}  "
        f"tools={len(env.operational.allowed_actions)}"
    )

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 2: should build three tiers"
assert agents_by_role["qa"].envelope.financial.max_spend_usd == 1.0
assert agents_by_role["admin"].envelope.financial.max_spend_usd == 10.0
assert agents_by_role["audit"].envelope.financial.max_spend_usd == 50.0
print("\u2713 Checkpoint 2 passed — three governed tiers wired\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Verify fail-closed on an out-of-envelope action
# ════════════════════════════════════════════════════════════════════════
#
# `build_capstone_stack` attached envelopes to the 3 Responsible role
# addresses. Ask the engine to verify an action the qa tier does NOT
# have in its allowed_actions — it MUST be denied.
#
# NOTE: `engine.verify_action()` on an address with no attached envelope
# auto-approves. We verify against the qa tier's attached address so
# the envelope is the source of restriction, not a missing clearance.

denied = governance_engine.verify_action(
    role_address="D1-R1-T1-R1",  # qa tier
    action="update_model",  # admin-only tool
    context={"cost": 0.10},
)
print(
    f"qa tier asks to update_model: "
    f"{'DENIED (correct)' if not denied.allowed else 'ALLOWED (BUG!)'}  "
    f"level={denied.level}"
)

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert not denied.allowed, "Task 3: qa tier MUST NOT be allowed to update_model"
assert denied.level == "blocked"
print("\u2713 Checkpoint 3 passed — fail-closed on envelope violation\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the envelope hierarchy
# ════════════════════════════════════════════════════════════════════════

envelope_table = pl.DataFrame(
    {
        "Role": [t.role for t in tiers],
        "Clearance": [t.clearance for t in tiers],
        "Budget (USD)": [t.budget_usd for t in tiers],
        "Allowed tools": [
            "answer+search",
            "+update+metrics+drift",
            "+audit_log+report",
        ],
        "Fail mode": ["closed", "closed", "closed"],
    }
)
envelope_table.write_parquet(OUTPUT_DIR / "governance_envelopes.parquet")
print("\nGovernance envelope hierarchy:")
print(envelope_table)

# INTERPRETATION: The envelope is monotonic — admin is a superset of
# qa, audit is a superset of admin. Monotonic tightening prevents
# privilege escalation between tiers: no governed agent can expand
# its own envelope. A higher tier is the only path to more capability,
# and that tier has its own audit log.


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Governance tier comparison chart
# ════════════════════════════════════════════════════════════════════════
# Grouped bar chart comparing the three governance tiers across budget,
# clearance level, and tool count. Makes the monotonic escalation
# visible at a glance.

tier_names = [t.role for t in tiers]
budgets = [t.budget_usd for t in tiers]
clearance_numeric = [1, 2, 3]  # public=1, internal=2, restricted=3 (teaching scale)
tool_counts = [len(t.tools) for t in tiers]

fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(tier_names))
width = 0.25

# Normalise to 0-1 for visual comparison
max_budget = max(budgets)
ax.bar(
    [i - width for i in x],
    [b / max_budget for b in budgets],
    width,
    label="Budget (normalised)",
    color="#3498db",
)
ax.bar(
    x,
    [c / 3.0 for c in clearance_numeric],
    width,
    label="Clearance level",
    color="#e67e22",
)
ax.bar(
    [i + width for i in x],
    [t / max(tool_counts) for t in tool_counts],
    width,
    label="Tool count (normalised)",
    color="#2ecc71",
)

ax.set_xticks(list(x))
ax.set_xticklabels(tier_names, fontsize=11)
ax.set_ylabel("Normalised value (0-1)")
ax.set_title("Governance Tier Comparison — Monotonic Escalation", fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.2)

for i, (b, c, t) in enumerate(zip(budgets, ["pub", "int", "restr"], tool_counts)):
    ax.text(i - width, b / max_budget + 0.03, f"${b:g}", ha="center", fontsize=8)
    ax.text(i, clearance_numeric[i] / 3.0 + 0.03, c, ha="center", fontsize=8)
    ax.text(i + width, t / max(tool_counts) + 0.03, f"{t}", ha="center", fontsize=8)

ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex8_governance_tiers.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore Financial Advisory (MAS TRM)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A MAS-regulated wealth management firm operates a
# retail-facing advisory bot (qa tier), an internal portfolio-ops
# dashboard (admin tier), and a compliance audit console (audit tier).
# MAS TRM 7.5 requires full traceability of every model-produced
# recommendation that reaches a retail customer.
#
# BUSINESS IMPACT: One un-logged retail advisory incident triggers a
# MAS investigation costing S$150,000 in external legal fees plus
# potential license suspension. The governed-agent envelope prevents
# the retail tier from invoking portfolio-ops tools (tool restriction)
# and caps retail-facing cost at US$1/request (budget enforcement).
# The audit tier's envelope has the access needed to satisfy a MAS
# Section 27 production order without extending retail or ops
# clearance.

print("\n" + "=" * 70)
print("  APPLY — MAS-Regulated Wealth Advisory")
print("=" * 70)
print(
    """
  qa tier     (retail customers):
    - answer + search only, US$1 budget cap
    - PUBLIC clearance (no portfolio PII)
    - fail_mode=closed → no envelope, no response

  admin tier  (portfolio ops):
    - +update_model +view_metrics +monitor_drift
    - US$10 budget; RESTRICTED clearance (aggregated positions)

  audit tier  (compliance + MAS liaison):
    - +access_audit_log +generate_report
    - US$50 budget; RESTRICTED clearance (individual trades)

  Un-logged advisory incident cost avoided: ~S$150,000
  Tier-jumping blocked structurally (not "hopefully by prompt")
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
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Compiled a PACT org YAML into a live GovernanceEngine
  [x] Built the shared 3-tier capstone stack via build_capstone_stack
  [x] Verified fail-closed on an out-of-envelope action
  [x] Visualised the monotonic envelope hierarchy
  [x] Applied governed agents to a MAS-regulated scenario

  KEY INSIGHT: Governance is not a layer you bolt onto a working
  agent — it is a WRAPPER that runs before the agent's first token
  is generated. GovernedSupervisor makes the envelope the primary
  interface; the base agent is an implementation detail the executor
  callback wraps on demand.

  Next: 03_multichannel_serving.py deploys these tiers via Nexus.
"""
)
