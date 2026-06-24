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
#   - Understand what compilation validates and what it does not
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

import polars as pl

from shared.mlfp06.ex_7 import (
    ORG_YAML,
    compile_governance,
    load_adversarial_prompts,
    write_org_yaml,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — D/T/R Accountability Grammar
# ════════════════════════════════════════════════════════════════════════
# D (Delegator):   A named human authority
# T (Task):        A bounded scope of work
# R (Responsible): The agent that executes within the task envelope
#
# Every autonomous action MUST trace back to a human Delegator.
# "The agent did it" is never an acceptable audit answer.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load adversarial prompts (for later techniques)
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Load Adversarial Test Prompts")
print("=" * 70)

# TODO: Call load_adversarial_prompts(n=50) and bind to adversarial_prompts
adversarial_prompts = ____

print(f"Loaded {adversarial_prompts.height} adversarial prompts")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert adversarial_prompts.height > 0
print("[x] Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Write the canonical org YAML to disk
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Write SG FinTech Org YAML")
print("=" * 70)

# TODO: Call write_org_yaml() to materialise ORG_YAML to a temp file.
#       Bind the returned path to org_yaml_path.
org_yaml_path = ____

print(f"YAML path: {org_yaml_path}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
# Modern pact schema: `envelopes:` block holds the D/T/R delegation
# contracts; the old `delegations:` top-level list is gone. Each
# envelope entry is one delegation from a human to an agent.
assert "departments" in ORG_YAML and "envelopes" in ORG_YAML
assert org_yaml_path
print("[x] Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Compile the organisation with GovernanceEngine
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Construct GovernanceEngine from org YAML")
print("=" * 70)

# TODO: Compile the YAML. Use compile_governance(org_yaml_path).
#       It returns (engine, org).
engine, org = ____

print(f"Agents: {org.n_agents}  Delegations: {org.n_delegations}")

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert org is not None
assert org.n_agents > 0
assert org.n_delegations > 0
print("[x] Checkpoint 3 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the D/T/R chains as a polars DataFrame
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Visualise D/T/R Chains")
print("=" * 70)

# TODO: Construct a polars DataFrame with columns
#       Delegator, Task, Responsible, Budget, Clearance
#       for the 6 delegations defined in ORG_YAML.
dtr_chains = ____

print(dtr_chains)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: MAS-Regulated Bank Audit
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore retail bank deploys AI agents and MUST answer,
# for every automated decision affecting a customer, which human
# authorised that class of action. Without compiled D/T/R, the answer
# is "the ML team, I guess" — which fails MAS TRM 7.5. With compiled
# governance, the answer is a 1-line query against the audit trail.
#
# BUSINESS IMPACT: A failed MAS TRM audit triggers remediation orders
# and capital add-ons — typically S$500K–S$5M per finding.

print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Compilation Is the Structural Audit Gate")
print("=" * 70)


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
  [x] Visualised the delegation chains
  [x] Mapped the grammar to a MAS TRM audit scenario

  Next: 02_envelopes.py adds operating envelopes + monotonic tightening.
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
