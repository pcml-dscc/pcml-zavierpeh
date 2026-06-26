# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 4: PACT Governance for a Production Agent Fleet

Complete the `solve()` function. Read problem.md for the full specification —
the exact envelopes to build, the 10 verify_action cases (in order), and the
privilege-escalation test. This task is fully deterministic (no LLM).
"""
from __future__ import annotations

import os
import tempfile

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

from shared.mlfp06.ex_7 import ORG_YAML, compile_governance

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


def _envelope(
    envelope_id: str,
    clearance: ConfidentialityLevel,
    max_spend_usd: float,
    allowed_actions: list[str],
) -> ConstraintEnvelopeConfig:
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


def solve() -> dict:
    """Compile the org, attach envelopes, run verify_action, test escalation.

    See problem.md for the exact envelope specs and the return contract:
        {"org_stats": {...}, "verdicts": [bool x10], "escalation_caught": bool}
    """
    yaml_path = os.path.join(tempfile.gettempdir(), "sg_fintech_org_assessment.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(ORG_YAML)
    engine, org = compile_governance(yaml_path)
    org_stats = {
        "n_agents": int(org.n_agents),
        "n_delegations": int(org.n_delegations),
        "n_departments": int(org.n_departments),
    }

    role_specs = {
        "data_analyst": (
            "chief_ml_officer",
            ConfidentialityLevel.RESTRICTED,
            20.0,
            ["read_data", "summarise_data", "generate_report"],
        ),
        "model_trainer": (
            "chief_ml_officer",
            ConfidentialityLevel.RESTRICTED,
            100.0,
            ["train_model", "evaluate_model", "read_data"],
        ),
        "risk_assessor": (
            "chief_risk_officer",
            ConfidentialityLevel.RESTRICTED,
            200.0,
            ["read_data", "audit_model", "generate_report", "access_audit_log"],
        ),
        "customer_agent": (
            "vp_customer",
            ConfidentialityLevel.PUBLIC,
            5.0,
            ["answer_question", "search_faq"],
        ),
    }

    for role, (delegator, clearance, budget, actions) in role_specs.items():
        engine.set_role_envelope(
            RoleEnvelope(
                id=f"{role}_role_envelope",
                defining_role_address=DELEGATOR_ADDRESSES[delegator],
                target_role_address=AGENT_ADDRESSES[role],
                envelope=_envelope(f"{role}_envelope", clearance, budget, actions),
            )
        )

    verdicts = [
        bool(
            engine.verify_action(
                role_address=AGENT_ADDRESSES[role],
                action=action,
                context={"cost": cost},
            ).allowed
        )
        for role, action, cost in CASES
    ]

    parent_envelope = _envelope(
        "vp_customer_parent",
        ConfidentialityLevel.CONFIDENTIAL,
        50.0,
        ["answer_question", "search_faq"],
    )
    rogue_child = _envelope(
        "rogue_customer_child",
        ConfidentialityLevel.RESTRICTED,
        1000.0,
        ["answer_question", "search_faq", "read_data", "deploy_model"],
    )
    try:
        RoleEnvelope.validate_tightening(
            parent_envelope=parent_envelope,
            child_envelope=rogue_child,
        )
        escalation_caught = False
    except MonotonicTighteningError:
        escalation_caught = True

    return {
        "org_stats": org_stats,
        "verdicts": verdicts,
        "escalation_caught": escalation_caught,
    }


if __name__ == "__main__":
    print(solve())
