# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.1: Organisation Definition & GovernanceEngine Compile
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Write a D/T/R (Delegator/Task/Responsible) organisation in YAML
#   - Compile that organisation with kailash-pact's GovernanceEngine
#   - Understand what compilation validates (chains, cycles, monotonic
#     clearance, budget bounds) — and what it does NOT (LLM safety)
#   - Map the abstract grammar to a real Singapore FinTech scenario
#
# PREREQUISITES: Exercise 6 (multi-agent systems)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load real adversarial prompts (used by later techniques)
#   2. Write the SG FinTech org YAML to disk
#   3. Compile with GovernanceEngine and inspect the result
#   4. Visualise the D/T/R chains as a table
#   5. Apply — why a MAS-regulated bank needs compiled governance
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from shared.mlfp06.ex_7 import (
    ORG_YAML,
    compile_governance,
    load_adversarial_prompts,
    write_org_yaml,
)

OUTPUT_DIR = Path("outputs") / "ex7_governance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# THEORY — D/T/R Accountability Grammar
# ════════════════════════════════════════════════════════════════════════
# Every autonomous action an agent takes in production MUST trace back
# to a human Delegator who authorised it. That is the D/T/R grammar:
#
#   D (Delegator):   A named human authority (chief_ml_officer,
#                    chief_risk_officer, vp_customer, ...).
#   T (Task):        A bounded scope of work the agent may perform
#                    (data_analysis, model_training, bias_audit, ...).
#   R (Responsible): The agent that executes within the envelope the
#                    delegator attached to the task.
#
# Analogy: A bank manager (D) tells a teller (R) "handle customer
# deposits up to S$10,000" (T with an envelope). If the teller approves
# a S$50,000 transfer, accountability traces back to the manager who
# set the envelope too wide — not to a nameless "the system".
#
# WHY THIS MATTERS: MAS TRM 7.5 and EU AI Act Art. 14 both require
# human oversight of AI systems. "The agent did it" is never an
# acceptable audit answer. D/T/R makes accountability structural.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Adversarial Test Prompts
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Load Adversarial Test Prompts")
print("=" * 70)

adversarial_prompts = load_adversarial_prompts(n=50)
print(f"Loaded {adversarial_prompts.height} real adversarial prompts")
print(
    f"Toxicity range: {adversarial_prompts['toxicity_score'].min():.2f} — "
    f"{adversarial_prompts['toxicity_score'].max():.2f}"
)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert adversarial_prompts.height > 0, "Task 1: adversarial prompts should load"
print("[x] Checkpoint 1 passed — adversarial test data ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Write Organisation YAML
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Write SG FinTech Org YAML (D/T/R)")
print("=" * 70)

org_yaml_path = write_org_yaml()

print(f"Organisation: SG FinTech AI Division")
print(f"Departments:  3 (ML Engineering, Risk & Compliance, Customer Intelligence)")
print(f"Agents:       6")
print(f"Delegations:  6 D/T/R chains")
print(f"YAML path:    {org_yaml_path}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
# Modern pact schema: `envelopes:` block holds the D/T/R delegation
# contracts; the old `delegations:` top-level list is gone. Each
# envelope entry is one delegation from a human (`defined_by`) to an
# agent (`target`).
assert "departments" in ORG_YAML and "envelopes" in ORG_YAML
assert org_yaml_path  # path was returned
print("[x] Checkpoint 2 passed — org YAML written\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Compile with GovernanceEngine
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Construct GovernanceEngine from org YAML")
print("=" * 70)

engine, org = compile_governance(org_yaml_path)

print(f"Compiled organisation:")
print(f"  Agents:      {org.n_agents}")
print(f"  Delegations: {org.n_delegations}")
print(f"  Departments: {org.n_departments}")
print(
    """
Compilation validates:
  - Every agent has a delegation chain to a human Delegator
  - No circular delegations (A -> B -> A)
  - Clearance levels decrease monotonically down chains
  - Budget envelopes don't exceed parent limits
What compilation does NOT validate:
  - Content safety of LLM outputs (that needs adversarial testing)
  - Runtime budget consumption (that needs the runtime wrapper)
"""
)

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert org is not None, "Task 3: compilation should succeed"
assert org.n_agents > 0, "Task 3: org should have agents"
assert org.n_delegations > 0, "Task 3: org should have delegations"
print(
    f"[x] Checkpoint 3 passed — compiled {org.n_agents} agents, "
    f"{org.n_delegations} delegations\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the D/T/R Chains
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Visualise D/T/R Chains")
print("=" * 70)

dtr_chains = pl.DataFrame(
    {
        "Delegator": [
            "chief_ml_officer",
            "chief_ml_officer",
            "chief_ml_officer",
            "chief_risk_officer",
            "chief_risk_officer",
            "vp_customer",
        ],
        "Task": [
            "data_analysis",
            "model_training",
            "model_deployment",
            "risk_assessment",
            "bias_audit",
            "customer_interaction",
        ],
        "Responsible": [
            "data_analyst",
            "model_trainer",
            "model_deployer",
            "risk_assessor",
            "bias_checker",
            "customer_agent",
        ],
        "Budget": ["$20", "$100", "$50", "$200", "$75", "$5"],
        "Clearance": [
            "internal",
            "confidential",
            "confidential",
            "restricted",
            "confidential",
            "public",
        ],
    }
)
print(dtr_chains)


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — D/T/R Org Chart (Delegator-to-Agent hierarchy)
# ════════════════════════════════════════════════════════════════════════
# Visual proof that every agent traces back to a named human Delegator.
# The tree structure makes accountability chains visually obvious — no
# agent floats without a human root.

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title(
    "D/T/R Organisation Chart — SG FinTech AI Division", fontweight="bold", fontsize=13
)

# Delegator positions (top row)
delegators = {
    "chief_ml_officer": (2, 6),
    "chief_risk_officer": (5, 6),
    "vp_customer": (8, 6),
}
# Agent positions (bottom row), keyed by delegator
agents_map = {
    "chief_ml_officer": [
        ("data_analyst", 0.5, 3.5),
        ("model_trainer", 2, 3.5),
        ("model_deployer", 3.5, 3.5),
    ],
    "chief_risk_officer": [("risk_assessor", 4.5, 3.5), ("bias_checker", 5.5, 3.5)],
    "vp_customer": [("customer_agent", 8, 3.5)],
}

for delegator, (dx, dy) in delegators.items():
    ax.text(
        dx,
        dy,
        delegator.replace("_", "\n"),
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="#3498db", edgecolor="white", alpha=0.9
        ),
        color="white",
    )
    for agent_name, ax_, ay in agents_map[delegator]:
        ax.text(
            ax_,
            ay,
            agent_name.replace("_", "\n"),
            ha="center",
            va="center",
            fontsize=7,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#2ecc71",
                edgecolor="white",
                alpha=0.8,
            ),
            color="white",
        )
        ax.annotate(
            "",
            xy=(ax_, ay + 0.8),
            xytext=(dx, dy - 0.8),
            arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5),
        )

# Legend
ax.text(1, 1.5, "D = Delegator (human)", fontsize=9, color="#3498db", fontweight="bold")
ax.text(
    1, 0.8, "R = Responsible (agent)", fontsize=9, color="#2ecc71", fontweight="bold"
)
ax.text(
    5,
    1.5,
    f"Agents: {org.n_agents}  |  Delegations: {org.n_delegations}",
    fontsize=10,
    color="#2c3e50",
)

plt.tight_layout()
fname = OUTPUT_DIR / "ex7_dtr_org_chart.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: MAS-Regulated Bank
# ════════════════════════════════════════════════════════════════════════
#
# SCENARIO: A Singapore retail bank deploys an AI agent platform for
# customer service, fraud triage, and risk reporting. MAS TRM 7.5
# mandates an auditable trail for every automated decision that
# affects a customer. The compliance team asks one question during the
# annual audit: "For every AI-taken action this year, show me which
# human authorised that class of action."
#
# Without compiled D/T/R governance, the bank's answer is "well, the
# ML team wrote the agents, so... them, I guess?" — which fails the
# audit because no specific human is accountable for any specific
# action. With compile_governance() (load_org_yaml + GovernanceEngine) running on boot, every
# agent action carries a delegation chain. The auditor's question is
# answered by a 1-line query against the audit trail.
#
# BUSINESS IMPACT: A failed MAS TRM audit triggers remediation orders,
# capital add-ons, and in severe cases, a pause on the affected
# business line. Singapore banks regularly face S$500K–S$5M in
# remediation costs per finding. One compiled org YAML eliminates an
# entire class of findings.

print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Compilation Is the Structural Audit Gate")
print("=" * 70)
print(
    f"  {org.n_agents} agents, {org.n_delegations} delegations, "
    f"all traced to named humans."
)
print("  The compilation step is the difference between 'we have")
print("  governance' and 'we can prove governance ran at boot'.")


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

# Primary lens: Governance (audit chain, envelope breach scan, verdict
# distribution, budget consumption). Secondary: Agent Trace.
if False:  # scaffold — requires a PACT GovernanceEngine or governed supervisor
    obs = LLMObservatory(governance=None, run_id="ex_7_governance_run")
    # obs.governance.verify_chain(audit_df)
    # obs.governance.budget_consumption()
    # obs.governance.negative_drills([...])  # envelope breach attempts
    print("\n── LLM Observatory Report ──")
    findings = obs.report()

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [✓] Governance (HEALTHY): audit chain intact (0 breaks), 128
#       actions recorded, 2 blocks + 1 escalate, budget at 34% of cap.
#   [!] Governance (WARNING on negative drills): 4/5 drills blocked,
#       1 drill succeeded ("approaching cap on financial envelope").
#       Fix: tighten budget envelope from $50 -> $20 per run.
#   [✓] Agent      (HEALTHY): 12 TAOD steps, no stuck loops.
#   [?] Output / Retrieval / Alignment / Attention (n/a)
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [GOVERNANCE LENS] Audit chain intact = every action's hash chains
#     into the next (Merkle-style). A broken chain means a row was
#     inserted / modified out-of-band — the flight recorder's integrity
#     is compromised. 2 blocks + 1 escalate on 128 actions is healthy
#     enforcement pressure. The negative-drill WARN is the important
#     one: we threw 5 attacks at the envelope, one succeeded because
#     the financial cap was loose.
#     >> Prescription: the drill that succeeded tells you which envelope
#        dimension to tighten. Don't just lower the cap — add a
#        derivative rule ("halt if cost doubles within 10s").
#  [AGENT LENS] Clean trace under governance confirms the envelope
#     didn't block legitimate work (no escalations on normal actions).
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Wrote a D/T/R organisation definition in YAML
  [x] Compiled it with compile_governance() -> GovernanceEngine(load_org_yaml)
  [x] Understood what compilation validates (and what it doesn't)
  [x] Mapped the grammar to a Singapore retail bank audit scenario

  KEY INSIGHT: Governance is engineering, not philosophy. If your
  governance story cannot be compiled and validated at boot, it is
  a slide deck, not a control.

  Next: 02_envelopes.py adds operating envelopes + monotonic
  tightening on top of the compiled org.
"""
)
