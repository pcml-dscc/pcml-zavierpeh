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
#   Delegator (D) — the human or role who authorises a task
#   Task        (T) — what can be done (bounded scope)
#   Responsible (R) — the agent carrying out the task
# Every agent action is traceable back to a human Delegator. Each
# D/T/R chain carries an operating envelope: budget, allowed actions,
# confidentiality clearance. Default is fail-closed.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Compile the org YAML into a GovernanceEngine
# ════════════════════════════════════════════════════════════════════════

org_path = write_org_yaml()
# TODO: load the org YAML + construct GovernanceEngine from its definition
#   Hint: pact.load_org_yaml(path) returns a LoadedOrg; pass
#         loaded.org_definition to GovernanceEngine(...).
loaded = ____
governance_engine = ____
compiled_org = governance_engine.get_org()

# Count the Responsible agents vs total ROLE nodes. Agents are ROLE
# nodes whose address contains a "-T" segment (they sit under a team).
n_agents = sum(
    1
    for n in compiled_org.nodes.values()
    if n.node_type == NodeType.ROLE and "-T" in n.address and not n.is_vacant
)
n_roles = sum(1 for n in compiled_org.nodes.values() if n.node_type == NodeType.ROLE)
print(f"Compiled org: {n_roles} roles ({n_agents} responsible agents)")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert n_roles >= 3, "Task 1: compiled org should carry 3+ Responsible roles"
print("\u2713 Checkpoint 1 passed — governance compiled\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the shared 3-tier governed stack
# ════════════════════════════════════════════════════════════════════════
#
# build_capstone_stack(engine) attaches a ConstraintEnvelopeConfig to
# each Responsible role (qa / admin / audit) and returns a
# GovernedSupervisor for each — plus the tier metadata so this file
# can read budgets / tools / clearance without re-deriving them.

# TODO: call build_capstone_stack(governance_engine). It returns a
# 2-tuple (agents_by_role, tiers).
agents_by_role, tiers = ____
print("Governed agent tiers:")
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
# Ask the engine to verify an action the qa tier does NOT have in its
# allowed_actions. It MUST be denied. Note: verify_action on a role
# with no attached envelope auto-approves — we verify against the qa
# tier's attached address so the envelope is the source of restriction.

# TODO: call governance_engine.verify_action on role "D1-R1-T1-R1"
# (qa tier), action "update_model" (admin-only), with context={"cost": 0.10}
denied = ____
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
# privilege escalation between tiers.


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Governance tier comparison chart
# ════════════════════════════════════════════════════════════════════════

tier_names = [t.role for t in tiers]
budgets = [t.budget_usd for t in tiers]
clearance_numeric = [1, 2, 3]
tool_counts = [len(t.tools) for t in tiers]

fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(tier_names))
width = 0.25
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
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex8_governance_tiers.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore Financial Advisory (MAS TRM)
# ════════════════════════════════════════════════════════════════════════
# A MAS-regulated wealth firm runs retail-facing advisory (qa),
# portfolio ops (admin), and compliance audit (audit). MAS TRM 7.5
# requires full traceability. Un-logged advisory incident cost:
# ~S$150,000 in external legal fees + license suspension risk.

print("\n" + "=" * 70)
print("  APPLY — MAS Wealth Advisory (tiered envelopes)")
print("=" * 70)


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

  Next: 03_multichannel_serving.py deploys these tiers via Nexus.
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
