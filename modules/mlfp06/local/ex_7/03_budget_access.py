# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.3: Budget Cascading & Access Control Decisions
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Cascade budget from parent agents to children using
#     `TeachingBudgetTracker`
#   - Detect overspend attempts before they reach the LLM
#   - Exercise `engine.verify_action(role_address, action, context)`
#     on allow AND deny paths
#   - Understand why deny-path coverage is non-negotiable
#
# PREREQUISITES: 02_envelopes.py
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Allocate & consume budget across a 3-agent hierarchy
#   2. Attempt an overspend and verify it is denied
#   3. Visualise spend vs. allocation
#   4. Run 10 engine.verify_action() tests — both allow and deny
#   5. Apply — MAS TRM cost controls
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

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

from shared.mlfp06.ex_7 import TeachingBudgetTracker, compile_governance

engine, org = compile_governance()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Budget as a Governance Primitive
# ════════════════════════════════════════════════════════════════════════
# A bug or prompt injection can drain your API credits in minutes.
# The first public "agent spent $10K in an afternoon" incident was a
# missing budget envelope. Hard budget cascading is the structural fix.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build the budget hierarchy
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Build Budget Hierarchy")
print("=" * 70)

# TODO: Create a TeachingBudgetTracker with total_budget=500.0
tracker = ____

# TODO: Allocate $20 to data_analyst, $100 to model_trainer, $50 to model_deployer
____
____
____

# TODO: Simulate a day's work — three model_trainer tasks at
#       $30, $30, $25; one data_analyst at $8; one model_deployer at $15.
____
____
____
____
____

print(tracker.summary())

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert tracker.remaining("model_trainer") == 15.0
print("\n[x] Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Overspend must be denied
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Overspend Attempt")
print("=" * 70)

# model_trainer has $15 remaining; try to spend $50.
# TODO: Call tracker.spend("model_trainer", 50.0) and bind to overspend_ok.
overspend_ok = ____

print(f"  Overspend $50: {'ALLOWED (bug!)' if overspend_ok else 'DENIED (correct)'}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert not overspend_ok
assert tracker.remaining("model_trainer") == 15.0
print("\n[x] Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise spend vs. allocation
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Visual Proof")
print("=" * 70)

# TODO: For each row in tracker.summary(), print an ASCII bar of
#       length 40 proportional to spent/allocated.
summary = tracker.summary()
print(f"  {'agent':<18} {'alloc':>8} {'spent':>8}  bar")
print("  " + "-" * 55)
for row in summary.iter_rows(named=True):
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Exercise engine.verify_action() (allow + deny cases)
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 4: Access Control via engine.verify_action()")
print("=" * 70)

# Dash-delimited D/T/R addresses (Shard 3 convention).
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
ROLE_TO_DELEGATOR: dict[str, str] = {
    "data_analyst": "chief_ml_officer",
    "model_trainer": "chief_ml_officer",
    "model_deployer": "chief_ml_officer",
    "risk_assessor": "chief_risk_officer",
    "bias_checker": "chief_risk_officer",
    "customer_agent": "vp_customer",
}


def _envelope(
    envelope_id: str,
    clearance: ConfidentialityLevel,
    max_spend_usd: float,
    allowed_actions: list[str],
) -> ConstraintEnvelopeConfig:
    """Minimal 5-dimension envelope for Task 4's access probe."""
    # TODO: Return a ConstraintEnvelopeConfig with all 5 dimensions.
    #       Financial: max_spend_usd={max_spend_usd}
    #       Operational: allowed_actions={allowed_actions}
    #       Temporal / DataAccess / Communication: permissive defaults.
    return ____


# TODO: Build envelopes_by_role for all 6 agents.
envelopes_by_role: dict[str, ConstraintEnvelopeConfig] = ____

# TODO: Attach each envelope via engine.set_role_envelope(RoleEnvelope(...)).
for role_id, env in envelopes_by_role.items():
    role_env = ____
    engine.set_role_envelope(role_env)


def test_access_control() -> list[dict]:
    test_cases: list[tuple[str, str, dict, bool]] = [
        ("model_trainer", "read_data", {"cost": 0.10}, True),
        ("model_trainer", "deploy_model", {"cost": 0.10}, False),
        ("customer_agent", "search_faq", {"cost": 0.01}, True),
        ("customer_agent", "read_data", {"cost": 0.10}, False),
        ("risk_assessor", "audit_model", {"cost": 0.50}, True),
        ("risk_assessor", "deploy_model", {"cost": 0.10}, False),
        ("model_deployer", "deploy_model", {"cost": 0.10}, True),
        ("data_analyst", "read_data", {"cost": 500.0}, False),
        ("bias_checker", "run_fairness_check", {"cost": 0.25}, True),
        ("customer_agent", "answer_question", {"cost": 100.0}, False),
    ]

    results = []
    for role_id, action, context, expected in test_cases:
        # TODO: Call engine.verify_action(
        #           role_address=AGENT_ADDRESSES[role_id],
        #           action=action, context=context)
        verdict = ____
        actual = verdict.allowed
        results.append(
            {"agent": role_id, "action": action, "expected": expected, "actual": actual}
        )
        status = "[ok]" if actual == expected else "[FAIL]"
        print(
            f"  {status} {role_id:<14} {action:<20} "
            f"{'ALLOW' if expected else 'DENY':<7} "
            f"{'ALLOW' if actual else 'DENY':<7}"
        )
    return results


access_results = test_access_control()

# ── Checkpoint 4 ────────────────────────────────────────────────────────
assert len(access_results) >= 10
assert all(r["actual"] == r["expected"] for r in access_results)
print(f"\n[x] Checkpoint 4 passed — {len(access_results)} verdicts\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: MAS TRM Cost Controls
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore digital bank's 24/7 loan triage agent must
# have a hard cap on per-decision compute spend under MAS TRM 5.8.
# Cascading budgets + engine.verify_action() make the cap structural.
#
# BUSINESS IMPACT: A runaway agent can easily produce a S$50K–S$200K
# API bill. Hard envelopes close the blast radius to the per-task cap.

print("=" * 70)
print("  KEY TAKEAWAY: Budget + Access Control Together")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Cascaded budget across a hierarchy via TeachingBudgetTracker
  [x] Caught an overspend attempt
  [x] Ran 10 engine.verify_action() tests across allow + deny paths
  [x] Mapped budgets to MAS TRM cost controls

  Next: 04_runtime_audit.py wires GovernedSupervisor into a live
  LLM workflow and tests it against real adversarial prompts.
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
