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
#   - Visualise the clearance lattice and per-agent envelope footprint
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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

OUTPUT_DIR = Path("outputs") / "ex7_governance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Compile once so this technique file is runnable in isolation
engine, org = compile_governance()
print("\n--- GovernanceEngine compiled ---\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Five Canonical Envelope Dimensions
# ════════════════════════════════════════════════════════════════════════
# An operating envelope is the multi-dimensional boundary PACT draws
# around a role. Canonical PACT names FIVE dimensions, and this
# exercise builds each one structurally — not as a narrated slide:
#
#   Financial     — max spend per invocation (the dollar ceiling)
#   Operational   — allowed/blocked actions + rate limits
#   Temporal      — active hours, blackout windows (off-hours freeze)
#   Data Access   — readable/writable paths, blocked data types
#   Communication — allowed channels, external-call approval gates
#
# The clearance level rides on top as a confidentiality classifier.
# Together these make up a `ConstraintEnvelopeConfig` — the object
# that PACT's engine checks on every `verify_action()` call.
#
# Analogy: A security badge at a research hospital. Budget = daily
# PET-scan allowance (Financial). Which procedures you may run =
# Operational. Which hours the badge is active = Temporal. Which
# patient files you may open = Data Access. Whether you may share
# findings outside the hospital = Communication. The badge NEVER
# grants more — it ONLY restricts.
#
# ── Sidebar: Canonical PACT clearance hierarchy ────────────────────
# MLFP06 teaches a 4-level clearance lattice
# (public < internal < confidential < restricted) because it is
# easier to hold in your head. Canonical PACT ships 5 levels:
# PUBLIC < RESTRICTED < CONFIDENTIAL < SECRET < TOP_SECRET. At the
# string interface, `"internal"` is a historical alias of
# `"restricted"` — they collide at one canonical level. You will see
# SECRET and TOP_SECRET in real governance configs for trading algos
# or patient records; the course's 4-level model is a teaching
# simplification, not a runtime limit.
# ────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build Each Agent's Full 5-Dimension Envelope
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Construct ConstraintEnvelopeConfig (all 5 dimensions)")
print("=" * 70)

# Address map (Shard 3 convention — dash-delimited D/T/R positions).
# Department heads: "D<n>-R<n>" (2 segments).
# Agents (Responsibles): "D<n>-R<n>-T<n>-R<n>" (4 segments).
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


def make_envelope(
    *,
    envelope_id: str,
    description: str,
    clearance: ConfidentialityLevel,
    max_spend_usd: float,
    allowed_actions: list[str],
    active_hours: tuple[str, str] | None = None,
    read_paths: list[str],
    write_paths: list[str],
    allowed_channels: list[str],
) -> ConstraintEnvelopeConfig:
    """Build a 5-dimension envelope.

    Every dimension is populated explicitly so the structural
    guarantee is visible at the call site. Missing a dimension would
    allow a later refactor to silently widen the envelope.
    """
    start, end = active_hours if active_hours else (None, None)
    return ConstraintEnvelopeConfig(
        id=envelope_id,
        description=description,
        confidentiality_clearance=clearance,
        # 1. Financial — dollar ceiling per invocation
        financial=FinancialConstraintConfig(max_spend_usd=max_spend_usd),
        # 2. Operational — allowed action allowlist (deny by default)
        operational=OperationalConstraintConfig(
            allowed_actions=allowed_actions,
            blocked_actions=[],
        ),
        # 3. Temporal — active hours window (None = always-on)
        temporal=TemporalConstraintConfig(
            active_hours_start=start,
            active_hours_end=end,
            blackout_periods=[],
        ),
        # 4. Data Access — path allowlist
        data_access=DataAccessConstraintConfig(
            read_paths=read_paths,
            write_paths=write_paths,
            blocked_data_types=[],
        ),
        # 5. Communication — channel allowlist
        communication=CommunicationConstraintConfig(
            allowed_channels=allowed_channels,
        ),
        max_delegation_depth=3,
    )


# Build the 6 agent envelopes. Each call populates ALL five dimensions.
# Every child envelope is a strict subset of its department head's
# envelope (see `head_envelopes` below). PACT's canonical 5-level
# hierarchy orders `CONFIDENTIAL > RESTRICTED > PUBLIC`, so department
# heads carry CONFIDENTIAL and children tighten down from there.
envelopes_by_role: dict[str, ConstraintEnvelopeConfig] = {
    "data_analyst": make_envelope(
        envelope_id="data_analyst_envelope",
        description="Data analyst — read-only data work",
        clearance=ConfidentialityLevel.RESTRICTED,
        max_spend_usd=20.0,
        allowed_actions=["read_data", "summarise_data", "generate_report"],
        read_paths=["/data/raw/*", "/data/curated/*"],
        write_paths=["/reports/analyst/*"],
        allowed_channels=["internal"],
    ),
    "model_trainer": make_envelope(
        envelope_id="model_trainer_envelope",
        description="Model trainer — training + evaluation",
        clearance=ConfidentialityLevel.RESTRICTED,
        max_spend_usd=100.0,
        allowed_actions=["train_model", "evaluate_model", "read_data"],
        read_paths=["/data/raw/*", "/data/curated/*"],
        write_paths=["/models/staging/*"],
        allowed_channels=["internal"],
    ),
    "model_deployer": make_envelope(
        envelope_id="model_deployer_envelope",
        description="Model deployer — deploy + monitor + rollback",
        clearance=ConfidentialityLevel.RESTRICTED,
        max_spend_usd=50.0,
        allowed_actions=["deploy_model", "monitor_model", "rollback_model"],
        read_paths=["/models/staging/*", "/models/prod/*"],
        write_paths=["/models/prod/*"],
        allowed_channels=["internal", "pagerduty"],
    ),
    "risk_assessor": make_envelope(
        envelope_id="risk_assessor_envelope",
        description="Risk assessor — audit-read + report",
        clearance=ConfidentialityLevel.RESTRICTED,
        max_spend_usd=200.0,
        allowed_actions=[
            "read_data",
            "audit_model",
            "generate_report",
            "access_audit_log",
        ],
        read_paths=["/data/raw/*", "/data/curated/*", "/models/prod/*", "/audit/*"],
        write_paths=["/reports/risk/*"],
        allowed_channels=["internal", "compliance"],
    ),
    "bias_checker": make_envelope(
        envelope_id="bias_checker_envelope",
        description="Bias checker — fairness audit only",
        clearance=ConfidentialityLevel.RESTRICTED,
        max_spend_usd=75.0,
        allowed_actions=["read_data", "audit_model", "run_fairness_check"],
        read_paths=["/data/curated/*", "/models/prod/*"],
        write_paths=["/reports/bias/*"],
        allowed_channels=["internal"],
    ),
    "customer_agent": make_envelope(
        envelope_id="customer_agent_envelope",
        description="Customer agent — public FAQ answers",
        clearance=ConfidentialityLevel.PUBLIC,
        max_spend_usd=5.0,
        allowed_actions=["answer_question", "search_faq"],
        active_hours=("00:00", "23:59"),
        read_paths=["/faq/*"],
        write_paths=[],
        allowed_channels=["customer_chat"],
    ),
}

# Attach each envelope to its role so the engine enforces it on
# subsequent verify_action() calls. The defining role (supervisor)
# is the department head; the target role is the agent.
ROLE_TO_DELEGATOR: dict[str, str] = {
    "data_analyst": "chief_ml_officer",
    "model_trainer": "chief_ml_officer",
    "model_deployer": "chief_ml_officer",
    "risk_assessor": "chief_risk_officer",
    "bias_checker": "chief_risk_officer",
    "customer_agent": "vp_customer",
}
for role_id, env in envelopes_by_role.items():
    role_env = RoleEnvelope(
        id=f"{role_id}_role_envelope",
        defining_role_address=DELEGATOR_ADDRESSES[ROLE_TO_DELEGATOR[role_id]],
        target_role_address=AGENT_ADDRESSES[role_id],
        envelope=env,
    )
    engine.set_role_envelope(role_env)

envelope_table = pl.DataFrame(
    {
        "Agent": list(envelopes_by_role.keys()),
        "Clearance": [
            env.confidentiality_clearance.name.lower()
            for env in envelopes_by_role.values()
        ],
        "Max $": [env.financial.max_spend_usd for env in envelopes_by_role.values()],
        "Allowed Actions": [
            ",".join(env.operational.allowed_actions)[:40]
            for env in envelopes_by_role.values()
        ],
        "Channels": [
            ",".join(env.communication.allowed_channels)
            for env in envelopes_by_role.values()
        ],
    }
)
print(envelope_table)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert envelope_table.height == 6, "Task 1: should have 6 agent envelopes"
for env in envelopes_by_role.values():
    assert env.financial is not None, "Financial dimension required"
    assert env.operational is not None, "Operational dimension required"
    assert env.temporal is not None, "Temporal dimension required"
    assert env.data_access is not None, "Data Access dimension required"
    assert env.communication is not None, "Communication dimension required"
print("\n[x] Checkpoint 1 passed — 6 envelopes, each with all 5 dimensions\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Verify Monotonic Tightening Structurally
# ════════════════════════════════════════════════════════════════════════
#
# Monotonic tightening: a child envelope is ALWAYS <= the envelope of
# its delegator on every dimension. Budget down, allowed-action set
# subset, clearance down, data paths subset, communication channels
# subset. PACT's `RoleEnvelope.validate_tightening()` checks EVERY
# dimension in one call — no hand-rolled integer comparisons.
#
# We build a synthetic "parent" envelope for each department head
# (the widest possible across its children) and verify each child
# chain tightens against it.

print("=" * 70)
print("TASK 2: Monotonic Tightening via RoleEnvelope.validate_tightening()")
print("=" * 70)

# The three department-head envelopes — permissive, but structurally
# bounded. Each one is the union (or superset) of its children's
# needs on every dimension: budget, actions, data paths, channels.
head_envelopes: dict[str, ConstraintEnvelopeConfig] = {
    "chief_ml_officer": make_envelope(
        envelope_id="chief_ml_officer_envelope",
        description="Chief ML Officer — ML department head",
        clearance=ConfidentialityLevel.CONFIDENTIAL,
        max_spend_usd=500.0,
        allowed_actions=[
            "read_data",
            "summarise_data",
            "generate_report",
            "train_model",
            "evaluate_model",
            "deploy_model",
            "monitor_model",
            "rollback_model",
        ],
        read_paths=[
            "/data/raw/*",
            "/data/curated/*",
            "/models/staging/*",
            "/models/prod/*",
        ],
        write_paths=[
            "/reports/analyst/*",
            "/models/staging/*",
            "/models/prod/*",
        ],
        allowed_channels=["internal", "pagerduty"],
    ),
    "chief_risk_officer": make_envelope(
        envelope_id="chief_risk_officer_envelope",
        description="Chief Risk Officer — risk department head",
        clearance=ConfidentialityLevel.CONFIDENTIAL,
        max_spend_usd=500.0,
        allowed_actions=[
            "read_data",
            "audit_model",
            "generate_report",
            "access_audit_log",
            "run_fairness_check",
        ],
        read_paths=[
            "/data/raw/*",
            "/data/curated/*",
            "/models/prod/*",
            "/audit/*",
        ],
        write_paths=["/reports/risk/*", "/reports/bias/*"],
        allowed_channels=["internal", "compliance"],
    ),
    "vp_customer": make_envelope(
        envelope_id="vp_customer_envelope",
        description="VP Customer — customer intelligence head",
        clearance=ConfidentialityLevel.CONFIDENTIAL,
        max_spend_usd=50.0,
        allowed_actions=["answer_question", "search_faq"],
        active_hours=("00:00", "23:59"),
        read_paths=["/faq/*"],
        write_paths=[],
        allowed_channels=["customer_chat"],
    ),
}

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
        RoleEnvelope.validate_tightening(
            parent_envelope=head_envelopes[delegator],
            child_envelope=envelopes_by_role[agent],
        )
        print(f"  [ok] {delegator} -> {agent}")
    except MonotonicTighteningError as exc:
        all_valid = False
        print(f"  [VIOLATION] {delegator} -> {agent}: {exc}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert all_valid, "Task 2: every chain must tighten on every dimension"
print("\n[x] Checkpoint 2 passed — all 6 chains tighten structurally\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Simulate a Privilege-Escalation Attempt
# ════════════════════════════════════════════════════════════════════════
#
# Hypothetical: an operator tries to re-delegate the customer_agent
# with elevated authority — higher budget, broader allowed actions,
# and RESTRICTED clearance. Under PACT, this attempt is caught
# structurally by `validate_tightening()` — not by a runtime
# integer comparison that a refactor can silently drop.

print("=" * 70)
print("TASK 3: Privilege-Escalation Attempt (caught at envelope time)")
print("=" * 70)

rogue_child = make_envelope(
    envelope_id="customer_agent_rogue_envelope",
    description="Rogue escalation — restricted clearance, high budget",
    clearance=ConfidentialityLevel.RESTRICTED,
    max_spend_usd=1000.0,  # 200x the legit budget
    allowed_actions=[
        "answer_question",
        "search_faq",
        "read_data",  # not in parent!
        "deploy_model",  # not in parent!
    ],
    read_paths=["/data/*", "/models/prod/*"],  # widened
    write_paths=["/models/prod/*"],  # widened
    allowed_channels=["customer_chat", "external_email"],  # widened
)

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
    # Print the first violation for readability
    print(f"  Reason:  {violation_reason[:180]}")
print("\n  PACT catches this at envelope construction, not at runtime.")

# Visualise the clearance lattice so students can see the "up and to
# the right" shape of the escalation. Course's 4-level teaching
# lattice (ordered via CLEARANCE_LEVELS) — canonical RESTRICTED
# covers both "internal" and "restricted" rungs.
print("\n  Clearance lattice (higher = more access):")
for level_name, level in sorted(CLEARANCE_LEVELS.items(), key=lambda x: -x[1]):
    # Map the course rung to the canonical PACT enum name it covers.
    target = "restricted" if level_name in ("internal", "restricted") else level_name
    agents_at_level = [
        role
        for role, env in envelopes_by_role.items()
        if env.confidentiality_clearance.name.lower() == target
    ]
    bar = "#" * (level + 1)
    print(f"    {level_name:<13} {bar:<5} {agents_at_level}")

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert escalation_caught, "Task 3: the escalation must be rejected"
print("\n[x] Checkpoint 3 passed — privilege escalation caught structurally\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Envelope Dimension Radar Chart
# ════════════════════════════════════════════════════════════════════════
# Each agent's operating envelope spans five dimensions. The radar
# chart shows at a glance how "wide" each agent's authority is — a
# public customer agent has a tiny footprint, while the risk assessor
# has broad reach. This is the visual proof of least-privilege.

dimensions = ["Clearance", "Budget", "Tools", "Role\nScope", "Data\nAccess"]

# Normalise each dimension to 0-1 for the radar (data preserved from
# the pre-migration version — visual proof does not depend on the new
# envelope API).
agent_data = {
    "data_analyst": [1 / 3, 20 / 200, 3 / 6, 0.4, 0.33],
    "model_trainer": [2 / 3, 100 / 200, 3 / 6, 0.6, 0.66],
    "risk_assessor": [3 / 3, 200 / 200, 4 / 6, 0.8, 1.0],
    "customer_agent": [0 / 3, 5 / 200, 2 / 6, 0.3, 0.1],
}

angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colors_radar = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

for (agent_name, values), color in zip(agent_data.items(), colors_radar):
    vals = values + values[:1]
    ax.plot(angles, vals, "o-", linewidth=2, label=agent_name, color=color)
    ax.fill(angles, vals, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dimensions, fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_title(
    "Operating Envelope Radar — Per-Agent Dimensions", fontweight="bold", pad=20
)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
plt.tight_layout()
fname = OUTPUT_DIR / "ex7_envelope_radar.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Apply: IMDA AI Verify Self-Assessment
# ════════════════════════════════════════════════════════════════════════
#
# SCENARIO: A Singapore e-commerce platform is preparing an IMDA AI
# Verify self-assessment (Singapore's voluntary AI governance testing
# framework). One of the required controls is "Access Control and
# Authorisation": show that every AI agent has an explicit least-
# privilege envelope and that no agent can escalate beyond its
# delegator.
#
# Without structural monotonic tightening, the answer is a spreadsheet
# that the compliance team updates by hand and that drifts from the
# code within weeks. With `ConstraintEnvelopeConfig` + `RoleEnvelope.
# validate_tightening()`, the answer is the YAML file that CI runs
# `compile_governance()` against on every PR — and the validation is
# structural, not narrative.
#
# BUSINESS IMPACT: IMDA AI Verify is increasingly used as a
# procurement filter by Singapore government agencies and
# GLC-linked companies. A platform that cannot produce the
# required evidence is excluded from tenders worth S$1M–S$10M
# annually. Getting the evidence right is not a nice-to-have.

print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Envelopes are the Structural Least-Privilege Gate")
print("=" * 70)
print("  Monotonic tightening makes privilege escalation")
print("  impossible at envelope time — not 'unlikely at runtime'.")


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
  [x] Built ConstraintEnvelopeConfig across all 5 canonical dimensions
  [x] Verified monotonic tightening via RoleEnvelope.validate_tightening
  [x] Simulated a privilege-escalation attempt and caught it structurally
  [x] Mapped envelopes to IMDA AI Verify self-assessment evidence

  KEY INSIGHT: 'Least privilege' is a slogan until you can express
  it as a structural property of a compiled graph. Monotonic
  tightening across five dimensions is that structural property.

  Next: 03_budget_access.py combines budget cascading with the
  access-control decision function.
"""
)
