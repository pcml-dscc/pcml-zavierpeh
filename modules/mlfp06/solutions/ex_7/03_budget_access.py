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
#   - Detect overspend attempts BEFORE they reach the LLM
#   - Exercise `engine.verify_action(role_address, action, context)`
#     across allow AND deny paths — the canonical PACT decision call
#   - Understand why test coverage MUST include the deny path
#
# PREREQUISITES: 02_envelopes.py
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Allocate & consume budget across a 3-agent hierarchy
#   2. Attempt an overspend and verify it is denied
#   3. Visualise spend vs. allocation
#   4. Exercise engine.verify_action() across 10 allow+deny cases
#   5. Apply — MAS TRM budget and cost controls
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
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

OUTPUT_DIR = Path("outputs") / "ex7_governance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

engine, org = compile_governance()
print("\n--- GovernanceEngine compiled ---\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Budget Is a Governance Primitive
# ════════════════════════════════════════════════════════════════════════
# An LLM call has a dollar cost. An autonomous agent can make many
# LLM calls per task. Without a hard budget per envelope, a bug — or
# a prompt injection — can drain your API credits in minutes. The
# first publicly reported "agent spent $10K in an afternoon" incident
# was a missing budget envelope on a customer-support agent.
#
# Budget cascading enforces the same hierarchy as clearance:
#
#   ml_director total budget:  $500
#     ├─ data_analyst:   $20/task
#     ├─ model_trainer:  $100/task
#     └─ model_deployer: $50/task
#
# After 3 training runs of $30 each: model_trainer spent $90. Next
# $50 request: DENIED (would exceed $100 allocation).


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build a Budget Hierarchy
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Build Budget Hierarchy")
print("=" * 70)

tracker = TeachingBudgetTracker(total_budget=500.0)
tracker.allocate("data_analyst", 20.0)
tracker.allocate("model_trainer", 100.0)
tracker.allocate("model_deployer", 50.0)

# Simulate a realistic day's work
tracker.spend("model_trainer", 30.0)  # Training task 1
tracker.spend("model_trainer", 30.0)  # Training task 2
tracker.spend("model_trainer", 25.0)  # Training task 3
tracker.spend("data_analyst", 8.0)  # Analysis task
tracker.spend("model_deployer", 15.0)  # Deployment

print("Budget summary after a day's work:")
print(tracker.summary())

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert tracker.remaining("model_trainer") == 15.0
print("\n[x] Checkpoint 1 passed — budget hierarchy populated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Overspend Attempt (Must Be Denied)
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Overspend Attempt")
print("=" * 70)

# model_trainer has $15 remaining. Try to spend $50.
overspend_ok = tracker.spend("model_trainer", 50.0)
print(
    f"  Overspend $50 on model_trainer (remaining $15): "
    f"{'ALLOWED (bug!)' if overspend_ok else 'DENIED (correct)'}"
)

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert not overspend_ok, "Task 2: overspend must be denied"
assert tracker.remaining("model_trainer") == 15.0, "$15 should still remain"
print("\n[x] Checkpoint 2 passed — overspend caught\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise Spend vs. Allocation
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Visual Proof — Spend vs. Allocation")
print("=" * 70)

summary = tracker.summary()
print()
print(f"  {'agent':<18} {'alloc':>8} {'spent':>8}  bar")
print(f"  {'-' * 55}")
for row in summary.iter_rows(named=True):
    alloc = row["allocated"]
    spent = row["consumed"]
    pct = int(40 * spent / alloc) if alloc > 0 else 0
    bar = "#" * pct + "." * (40 - pct)
    print(f"  {row['agent']:<18} ${alloc:>7.0f} ${spent:>7.0f}  {bar}")
print()


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Exercise engine.verify_action() Across Allow + Deny Cases
# ════════════════════════════════════════════════════════════════════════
#
# A governance system that has only been tested on the allow path is
# a system that has not been tested at all. Every access-control test
# suite MUST include both allow and deny cases, because the failure
# mode you care about is "deny silently became allow after refactor".
#
# Modern PACT exposes a single decision entry point:
#   engine.verify_action(role_address, action, context)
# It returns a `GovernanceVerdict` carrying `.allowed`, `.level`, and
# `.reason`. The role address is the dash-delimited D/T/R grammar.

print("=" * 70)
print("TASK 4: Access Control via engine.verify_action() (Allow + Deny)")
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
    """Minimal 5-dimension envelope for the access-table probe.

    The financial and operational dimensions are the ones Task 4
    actually exercises; the other three are populated with permissive
    defaults so every envelope is structurally complete.
    """
    return ConstraintEnvelopeConfig(
        id=envelope_id,
        description=envelope_id,
        confidentiality_clearance=clearance,
        financial=FinancialConstraintConfig(max_spend_usd=max_spend_usd),
        operational=OperationalConstraintConfig(
            allowed_actions=allowed_actions,
            blocked_actions=[],
        ),
        temporal=TemporalConstraintConfig(blackout_periods=[]),
        data_access=DataAccessConstraintConfig(
            read_paths=["/*"],
            write_paths=[],
            blocked_data_types=[],
        ),
        communication=CommunicationConstraintConfig(allowed_channels=["internal"]),
        max_delegation_depth=3,
    )


envelopes_by_role: dict[str, ConstraintEnvelopeConfig] = {
    "data_analyst": _envelope(
        "data_analyst_envelope",
        ConfidentialityLevel.RESTRICTED,
        20.0,
        ["read", "read_data", "summarise_data", "generate_report"],
    ),
    "model_trainer": _envelope(
        "model_trainer_envelope",
        ConfidentialityLevel.RESTRICTED,
        100.0,
        ["read", "read_data", "train_model", "evaluate_model"],
    ),
    "model_deployer": _envelope(
        "model_deployer_envelope",
        ConfidentialityLevel.RESTRICTED,
        50.0,
        ["deploy_model", "monitor_model", "rollback_model"],
    ),
    "risk_assessor": _envelope(
        "risk_assessor_envelope",
        ConfidentialityLevel.RESTRICTED,
        200.0,
        [
            "read",
            "read_data",
            "audit_model",
            "generate_report",
            "access_audit_log",
        ],
    ),
    "bias_checker": _envelope(
        "bias_checker_envelope",
        ConfidentialityLevel.RESTRICTED,
        75.0,
        ["read_data", "audit_model", "run_fairness_check"],
    ),
    "customer_agent": _envelope(
        "customer_agent_envelope",
        ConfidentialityLevel.PUBLIC,
        5.0,
        ["answer_question", "search_faq"],
    ),
}

# Attach each envelope to its role so verify_action() enforces it.
for role_id, env in envelopes_by_role.items():
    role_env = RoleEnvelope(
        id=f"{role_id}_role_envelope",
        defining_role_address=DELEGATOR_ADDRESSES[ROLE_TO_DELEGATOR[role_id]],
        target_role_address=AGENT_ADDRESSES[role_id],
        envelope=env,
    )
    engine.set_role_envelope(role_env)


def test_access_control() -> list[dict]:
    # (role_id, action, context, expected_allowed, reason)
    test_cases: list[tuple[str, str, dict, bool, str]] = [
        ("model_trainer", "read_data", {"cost": 0.10}, True, "Within envelope"),
        (
            "model_trainer",
            "deploy_model",
            {"cost": 0.10},
            False,
            "deploy_model not in allowed_actions",
        ),
        (
            "customer_agent",
            "search_faq",
            {"cost": 0.01},
            True,
            "Within envelope",
        ),
        (
            "customer_agent",
            "read_data",
            {"cost": 0.10},
            False,
            "read_data not in allowed_actions",
        ),
        (
            "risk_assessor",
            "audit_model",
            {"cost": 0.50},
            True,
            "Within envelope",
        ),
        (
            "risk_assessor",
            "deploy_model",
            {"cost": 0.10},
            False,
            "deploy_model not in allowed_actions",
        ),
        (
            "model_deployer",
            "deploy_model",
            {"cost": 0.10},
            True,
            "Within envelope",
        ),
        (
            "data_analyst",
            "read_data",
            {"cost": 500.0},  # cost > $20 financial cap
            False,
            "Budget exceeded ($500 > $20)",
        ),
        (
            "bias_checker",
            "run_fairness_check",
            {"cost": 0.25},
            True,
            "Within envelope",
        ),
        (
            "customer_agent",
            "answer_question",
            {"cost": 100.0},  # cost > $5 financial cap
            False,
            "Budget exceeded ($100 > $5)",
        ),
    ]

    results: list[dict] = []
    print(f"  {'Agent':<16} {'Action':<22} {'Expected':<9} {'Actual':<9}")
    print("  " + "-" * 60)

    for role_id, action, context, expected, _hint in test_cases:
        address = AGENT_ADDRESSES[role_id]
        verdict = engine.verify_action(
            role_address=address,
            action=action,
            context=context,
        )
        actual = verdict.allowed
        match = actual == expected
        status = "[ok]" if match else "[FAIL]"
        results.append(
            {
                "agent": role_id,
                "action": action,
                "expected": expected,
                "actual": actual,
                "match": match,
                "reason": verdict.reason,
            }
        )
        print(
            f"  {status} {role_id:<14} {action:<20} "
            f"{'ALLOW' if expected else 'DENY':<7} "
            f"{'ALLOW' if actual else 'DENY':<7}"
        )
        if not actual:
            print(f"      reason: {verdict.reason[:80]}")

    return results


access_results = test_access_control()

# ── Checkpoint 4 ────────────────────────────────────────────────────────
assert len(access_results) >= 10, "Task 4: should test at least 10 cases"
mismatches = [r for r in access_results if not r["match"]]
assert not mismatches, (
    f"Task 4: {len(mismatches)} verdicts did not match expectation — "
    f"{[r['action'] for r in mismatches]}"
)
deny_count = sum(1 for r in access_results if not r["expected"])
print(
    f"\n[x] Checkpoint 4 passed — {len(access_results)} verdicts "
    f"({deny_count} deny-path)\n"
)


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Budget cascade bar chart + access decision heatmap
# ════════════════════════════════════════════════════════════════════════
# Two panels: (1) stacked bars showing allocated vs consumed budget per
# agent — the remaining gap is visual headroom; (2) heatmap of access
# decisions showing allow (green) and deny (red) across agents/actions.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: budget cascade
agents_budget = summary["agent"].to_list()
allocated = summary["allocated"].to_list()
consumed = summary["consumed"].to_list()
remaining = summary["remaining"].to_list()
x = range(len(agents_budget))

ax1.bar(x, consumed, color="#e74c3c", label="Consumed", width=0.5)
ax1.bar(
    x,
    remaining,
    bottom=consumed,
    color="#2ecc71",
    label="Remaining",
    width=0.5,
    alpha=0.7,
)
ax1.set_xticks(list(x))
ax1.set_xticklabels([a.replace("_", "\n") for a in agents_budget], fontsize=8)
ax1.set_ylabel("USD")
ax1.set_title("Budget Cascade — Spend vs Allocation", fontweight="bold")
ax1.legend(fontsize=8)
for i, (c, a) in enumerate(zip(consumed, allocated)):
    ax1.text(i, a + 1, f"${a:.0f}", ha="center", fontsize=8, color="#2c3e50")

# Right: access decision matrix heatmap
unique_agents = sorted(set(r["agent"] for r in access_results))
unique_actions = sorted(set(r["action"] for r in access_results))
matrix = []
for agent in unique_agents:
    row = []
    for action in unique_actions:
        match = [
            r for r in access_results if r["agent"] == agent and r["action"] == action
        ]
        if match:
            row.append(1 if match[0]["match"] else 0)
        else:
            row.append(0.5)  # not tested
    matrix.append(row)

from matplotlib.colors import ListedColormap

cmap = ListedColormap(["#e74c3c", "#bdc3c7", "#2ecc71"])
im = ax2.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax2.set_xticks(range(len(unique_actions)))
ax2.set_xticklabels(
    [a.replace("_", "\n") for a in unique_actions], fontsize=7, rotation=45, ha="right"
)
ax2.set_yticks(range(len(unique_agents)))
ax2.set_yticklabels([a.replace("_", "\n") for a in unique_agents], fontsize=8)
ax2.set_title("Access Decision Matrix", fontweight="bold")

plt.tight_layout()
fname = OUTPUT_DIR / "ex7_budget_access_viz.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: MAS TRM Cost Control
# ════════════════════════════════════════════════════════════════════════
#
# SCENARIO: A Singapore digital bank's AI-powered loan triage agent
# runs 24/7 against customer applications. MAS TRM Section 5.8 (Cost
# Controls) requires evidence that automated systems have a hard cap
# on per-decision compute spend AND that the cap cannot be silently
# raised by an engineer.
#
# Without cascading budgets, the cap lives in a config file that a
# developer can edit. With PACT-compiled envelopes + verify_action,
# the cap lives in the governance graph, which is validated at boot
# AND at every access decision. A PR that attempts to raise the cap
# is visible in version control, reviewable, and blocked by CI if
# monotonic tightening is violated.
#
# BUSINESS IMPACT: A single runaway agent incident — say, 24 hours
# of compounding LLM calls against a loan queue — can produce a
# S$50K–S$200K API bill plus remediation overhead. Hard envelopes
# close the blast radius to the per-task budget, typically S$0.10–
# S$5 per decision.

print("=" * 70)
print("  KEY TAKEAWAY: Budget + Access Control Together")
print("=" * 70)
print("  Budget cascading prevents runaway spend; access control")
print("  prevents out-of-scope actions. You need BOTH — a within-")
print("  budget agent doing the wrong thing is still a breach.")


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
  [x] Cascaded budget across a 3-agent hierarchy via TeachingBudgetTracker
  [x] Demonstrated a rejected overspend attempt
  [x] Ran 10 engine.verify_action() tests covering allow AND deny paths
  [x] Visualised the spend bars to see remaining headroom
  [x] Mapped budget envelopes to MAS TRM cost controls

  KEY INSIGHT: You have not tested a governance system until you
  have tested the denials. "Happy path works" is not evidence.

  Next: 04_runtime_audit.py wires GovernedSupervisor into a live
  LLM workflow and tests it against real adversarial prompts.
"""
)
