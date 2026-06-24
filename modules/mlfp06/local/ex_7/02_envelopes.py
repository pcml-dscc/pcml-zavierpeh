# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.2: Operating Envelopes & Monotonic Tightening
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build `ConstraintEnvelopeConfig` objects across all 5 canonical
#     dimensions (Financial, Operational, Temporal, Data Access,
#     Communication)
#   - Express the monotonic-tightening rule structurally via
#     `RoleEnvelope.validate_tightening()` — no hand-rolled integer
#     comparisons
#   - Detect privilege-escalation attempts at envelope-compile time
#
# PREREQUISITES: 01_org_compile.py
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Build each agent's full 5-dimension operating envelope
#   2. Verify monotonic tightening for every delegation chain
#   3. Simulate a privilege-escalation attempt (caught structurally)
#   4. Apply — IMDA AI Verify self-assessment
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl
from kailash.trust.pact.envelopes import MonotonicTighteningError
from pact import (
    CommunicationConstraintConfig,
    ConfidentialityLevel,
    ConstraintEnvelopeConfig,
    DataAccessConstraintConfig,
    FinancialConstraintConfig,
    OperationalConstraintConfig,
    RoleEnvelope,
    TemporalConstraintConfig,
)

from shared.mlfp06.ex_7 import CLEARANCE_LEVELS, compile_governance

engine, org = compile_governance()


# ════════════════════════════════════════════════════════════════════════
# THEORY — The 5 Canonical Envelope Dimensions
# ════════════════════════════════════════════════════════════════════════
# Financial, Operational, Temporal, Data Access, Communication. The
# envelope ONLY restricts — it never grants. Clearance rides on top
# as a confidentiality classifier.
#
# ── Sidebar: Canonical PACT clearance hierarchy ────────────────────
# The course teaches a 4-level lattice (public < internal <
# confidential < restricted). Canonical PACT ships 5 levels
# (PUBLIC < RESTRICTED < CONFIDENTIAL < SECRET < TOP_SECRET). The
# string `"internal"` is a historical alias of `"restricted"`.
# ────────────────────────────────────────────────────────────────────


# Address map (dash-delimited D/T/R grammar). Department heads use
# "D<n>-R<n>"; agents use "D<n>-R<n>-T<n>-R<n>".
AGENT_ADDRESSES: dict[str, str] = {
    "data_analyst": "D1-R1-T1-R1",
    "model_trainer": "D1-R1-T2-R1",
    "model_deployer": "D1-R1-T3-R1",
    "risk_assessor": "D2-R1-T1-R1",
    "bias_checker": "D2-R1-T2-R1",
    "customer_agent": "D3-R1-T1-R1",
}
DELEGATOR_ADDRESSES: dict[str, str] = {
    "chief_ml_officer": "D1-R1",
    "chief_risk_officer": "D2-R1",
    "vp_customer": "D3-R1",
}


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build Each Agent's Full 5-Dimension Envelope
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Construct ConstraintEnvelopeConfig (all 5 dimensions)")
print("=" * 70)


def make_envelope(
    *,
    envelope_id: str,
    description: str,
    clearance: ConfidentialityLevel,
    max_spend_usd: float,
    allowed_actions: list[str],
    read_paths: list[str],
    write_paths: list[str],
    allowed_channels: list[str],
) -> ConstraintEnvelopeConfig:
    """Build a 5-dimension envelope.

    Populate EVERY dimension explicitly so the structural guarantee
    is visible at the call site. Hint: the five configs you need are
    `FinancialConstraintConfig`, `OperationalConstraintConfig`,
    `TemporalConstraintConfig`, `DataAccessConstraintConfig`, and
    `CommunicationConstraintConfig`.
    """
    # TODO: Return a ConstraintEnvelopeConfig with:
    #   id=envelope_id, description=description,
    #   confidentiality_clearance=clearance,
    #   financial=FinancialConstraintConfig(max_spend_usd=...),
    #   operational=OperationalConstraintConfig(
    #       allowed_actions=..., blocked_actions=[]),
    #   temporal=TemporalConstraintConfig(blackout_periods=[]),
    #   data_access=DataAccessConstraintConfig(
    #       read_paths=..., write_paths=..., blocked_data_types=[]),
    #   communication=CommunicationConstraintConfig(
    #       allowed_channels=...),
    #   max_delegation_depth=3.
    return ____


# TODO: Build the 6 agent envelopes. Each agent takes a subset of its
#       department head's authority (strict subset on every dimension).
envelopes_by_role: dict[str, ConstraintEnvelopeConfig] = ____

# TODO: Attach each envelope to the engine via `engine.set_role_envelope`.
ROLE_TO_DELEGATOR: dict[str, str] = {
    "data_analyst": "chief_ml_officer",
    "model_trainer": "chief_ml_officer",
    "model_deployer": "chief_ml_officer",
    "risk_assessor": "chief_risk_officer",
    "bias_checker": "chief_risk_officer",
    "customer_agent": "vp_customer",
}
for role_id, env in envelopes_by_role.items():
    role_env = ____
    engine.set_role_envelope(role_env)

envelope_table = pl.DataFrame(
    {
        "Agent": list(envelopes_by_role.keys()),
        "Clearance": [
            env.confidentiality_clearance.name.lower()
            for env in envelopes_by_role.values()
        ],
        "Max $": [env.financial.max_spend_usd for env in envelopes_by_role.values()],
    }
)
print(envelope_table)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert envelope_table.height == 6
for env in envelopes_by_role.values():
    assert env.financial is not None
    assert env.operational is not None
    assert env.temporal is not None
    assert env.data_access is not None
    assert env.communication is not None
print("\n[x] Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Verify Monotonic Tightening Structurally
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Monotonic Tightening via RoleEnvelope.validate_tightening()")
print("=" * 70)

# TODO: Build one "head" envelope per department — the widest envelope
#       the department may ever hand out. Children must be strict
#       subsets on every dimension.
head_envelopes: dict[str, ConstraintEnvelopeConfig] = ____

delegation_chains: list[tuple[str, str]] = [
    ("chief_ml_officer", "data_analyst"),
    ("chief_ml_officer", "model_trainer"),
    ("chief_ml_officer", "model_deployer"),
    ("chief_risk_officer", "risk_assessor"),
    ("chief_risk_officer", "bias_checker"),
    ("vp_customer", "customer_agent"),
]

all_valid = True
for delegator, agent in delegation_chains:
    try:
        # TODO: Call RoleEnvelope.validate_tightening with parent_envelope
        #       and child_envelope kwargs.
        ____
        print(f"  [ok] {delegator} -> {agent}")
    except MonotonicTighteningError as exc:
        all_valid = False
        print(f"  [VIOLATION] {delegator} -> {agent}: {exc}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert all_valid
print("\n[x] Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Simulate a Privilege-Escalation Attempt
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Privilege-Escalation Attempt")
print("=" * 70)

# TODO: Build a rogue envelope that exceeds vp_customer on at least
#       TWO dimensions (e.g. financial + operational + clearance).
rogue_child = ____

escalation_caught = False
violation_reason: str | None = None
try:
    RoleEnvelope.validate_tightening(
        parent_envelope=head_envelopes["vp_customer"],
        child_envelope=rogue_child,
    )
except MonotonicTighteningError as exc:
    escalation_caught = True
    violation_reason = str(exc)

print(f"  Attempt: vp_customer -> customer_agent (ROGUE)")
print(f"  Result:  {'REJECTED' if escalation_caught else 'ACCEPTED (bug!)'}")
if violation_reason:
    print(f"  Reason:  {violation_reason[:180]}")

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert escalation_caught
print("\n[x] Checkpoint 3 passed — privilege escalation caught\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Apply: IMDA AI Verify Self-Assessment
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore e-commerce platform preparing an IMDA AI Verify
# self-assessment must evidence that every AI agent has a least-
# privilege envelope that cannot be escalated. With compiled envelopes
# the evidence is automated; without, it drifts from code within weeks.
#
# BUSINESS IMPACT: IMDA AI Verify is increasingly a procurement filter
# for Singapore government and GLC-linked tenders worth S$1M–S$10M
# annually. A platform that cannot produce the evidence is excluded.

print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Envelopes = Structural Least-Privilege")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built ConstraintEnvelopeConfig across all 5 canonical dimensions
  [x] Verified monotonic tightening via RoleEnvelope.validate_tightening
  [x] Caught a privilege-escalation attempt structurally
  [x] Mapped envelopes to IMDA AI Verify evidence

  Next: 03_budget_access.py combines budget cascading with access control.
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
