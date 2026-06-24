# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 4: PACT Governance for a Production Agent Fleet

Complete the `solve()` function. Read problem.md for the full specification —
the exact envelopes to build, the 10 verify_action cases (in order), and the
privilege-escalation test. This task is fully deterministic (no LLM).
"""
from __future__ import annotations

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

from shared.mlfp06.ex_7 import compile_governance

# D/T/R addresses for the four agent roles and their delegators.
AGENT_ADDRESSES = {
    "data_analyst": "D1-R1-T1-R1",
    "model_trainer": "D1-R1-T2-R1",
    "risk_assessor": "D2-R1-T1-R1",
    "customer_agent": "D3-R1-T1-R1",
}
DELEGATOR_ADDRESSES = {
    "chief_ml_officer": "D1-R1",
    "chief_risk_officer": "D2-R1",
    "vp_customer": "D3-R1",
}

# The 10 verify_action probes, in grading order (role, action, cost).
CASES = [
    ("data_analyst", "read_data", 0.10),
    ("data_analyst", "deploy_model", 0.10),
    ("data_analyst", "read_data", 50.0),
    ("model_trainer", "train_model", 5.0),
    ("model_trainer", "deploy_model", 1.0),
    ("risk_assessor", "audit_model", 0.50),
    ("risk_assessor", "access_audit_log", 1.0),
    ("customer_agent", "search_faq", 0.01),
    ("customer_agent", "read_data", 0.10),
    ("customer_agent", "answer_question", 100.0),
]


def solve() -> dict:
    """Compile the org, attach envelopes, run verify_action, test escalation.

    See problem.md for the exact envelope specs and the return contract:
        {"org_stats": {...}, "verdicts": [bool x10], "escalation_caught": bool}
    """
    # TODO 1: Compile the organisation with compile_governance() -> (engine, org).
    #         Read org.n_agents / org.n_delegations / org.n_departments.

    # TODO 2: Build a ConstraintEnvelopeConfig for each of the four roles
    #         (all 5 dimensions populated). Attach each to the engine via
    #         engine.set_role_envelope(RoleEnvelope(...)) using the addresses
    #         above. Budgets + allowed actions are in problem.md.

    # TODO 3: For each case in CASES (in order), call
    #         engine.verify_action(role_address=..., action=..., context={"cost": ...})
    #         and append verdict.allowed (a bool) to a verdicts list.

    # TODO 4: Build a CONFIDENTIAL parent envelope for vp_customer and a rogue
    #         RESTRICTED child that escalates budget + actions. Call
    #         RoleEnvelope.validate_tightening(parent_envelope=..., child_envelope=...)
    #         inside try/except MonotonicTighteningError; set escalation_caught.

    return {
        "org_stats": {"n_agents": 0, "n_delegations": 0, "n_departments": 0},
        "verdicts": [],
        "escalation_caught": False,
    }


if __name__ == "__main__":
    print(solve())
