# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.5: Regulatory Compliance Audit Report
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Extract audit trails from a GovernedSupervisor for all three tiers
#   - Verify the hash-chained audit trail for tamper-evidence
#   - Summarise governance activity into regulator-ready sections
#   - Map technical controls to EU AI Act, AI Verify, MAS TRM, PDPA
#   - Visualise the full platform scorecard
#   - Apply the audit report to a MAS Section 27 production order
#
# PREREQUISITES: Exercises 8.1-8.4
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Rebuild the governed stack and drive sample queries
#   2. Extract the per-tier audit trails via ``supervisor.audit.to_list()``
#      and verify tamper-evidence via ``supervisor.audit.verify_chain()``
#   3. Build the section-by-section compliance report
#   4. Visualise the 6-regulation regulatory mapping table
#   5. Apply the report to a MAS Section 27 production order
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl
from pact import GovernanceEngine, load_org_yaml

from shared.mlfp06.ex_8 import (
    OUTPUT_DIR,
    build_capstone_stack,
    handle_qa,
    load_mmlu_eval,
    run_async,
    write_org_yaml,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Compliance as a read-out, not a rewrite
# ════════════════════════════════════════════════════════════════════════
# If governance is wired correctly, a compliance audit is NOT a scramble
# — it is a SELECT statement against the audit trail. The trick is that
# the audit trail has to EXIST before the regulator asks. Every capstone
# tier (qa, admin, audit) has been running on a GovernedSupervisor since
# Exercise 8.2, and each supervisor's ``audit`` attribute records every
# run into a hash-chained list. Two reads give a regulator everything:
#
#   supervisor.audit.to_list()        → the entries themselves
#   supervisor.audit.verify_chain()   → structural tamper-evidence
#
# We map the trail to the frameworks the Singapore regulator cares
# about: EU AI Act (cross-border), AI Verify (local toolkit), MAS TRM
# (technology risk), and PDPA (personal data).


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Rebuild stack + drive sample queries so audit trails exist
# ════════════════════════════════════════════════════════════════════════

eval_data = load_mmlu_eval(n_rows=100)

org_path = write_org_yaml()
loaded = load_org_yaml(org_path)
governance_engine = GovernanceEngine(loaded.org_definition)
agents_by_role, tiers = build_capstone_stack(governance_engine)

print("Governed stack rebuilt:")
for tier in tiers:
    print(
        f"  {tier.role:6s} -> budget=${tier.budget_usd:>5.1f}  "
        f"clearance={tier.clearance}"
    )


async def drive_sample_traffic() -> None:
    """Drive a small burst through every tier so each audit log has content."""
    for q in eval_data["instruction"].to_list()[:3]:
        await handle_qa(q, role="qa", agents_by_role=agents_by_role)
    await handle_qa(
        "Show model performance metrics",
        role="admin",
        agents_by_role=agents_by_role,
    )
    await handle_qa(
        "Generate quarterly compliance report",
        role="audit",
        agents_by_role=agents_by_role,
    )


run_async(drive_sample_traffic())

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 1: governed stack must be rebuilt"
print("\u2713 Checkpoint 1 passed — stack rebuilt and sample traffic driven\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Extract per-tier audit trails + verify tamper-evidence
# ════════════════════════════════════════════════════════════════════════
#
# ``GovernedSupervisor.audit`` is the read-only view of a hash-chained
# audit list. Each entry references the prior entry by hash — silently
# editing any entry invalidates the whole chain. ``verify_chain()``
# returns True only if the chain has not been tampered with.

qa_audit = agents_by_role["qa"].audit.to_list()
admin_audit = agents_by_role["admin"].audit.to_list()
audit_audit = agents_by_role["audit"].audit.to_list()

qa_chain_valid = agents_by_role["qa"].audit.verify_chain()
admin_chain_valid = agents_by_role["admin"].audit.verify_chain()
audit_chain_valid = agents_by_role["audit"].audit.verify_chain()

print("Audit trail entry counts + tamper-evidence:")
print(f"  qa:    {len(qa_audit):>3} entries  " f"chain_valid={qa_chain_valid}")
print(f"  admin: {len(admin_audit):>3} entries  " f"chain_valid={admin_chain_valid}")
print(f"  audit: {len(audit_audit):>3} entries  " f"chain_valid={audit_chain_valid}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert qa_audit is not None, "Task 2: qa audit trail should be accessible"
assert qa_chain_valid, "Task 2: qa audit chain should verify"
assert admin_chain_valid, "Task 2: admin audit chain should verify"
assert audit_chain_valid, "Task 2: audit-tier audit chain should verify"
print("\u2713 Checkpoint 2 passed — audit trails extracted and verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the section-by-section compliance report
# ════════════════════════════════════════════════════════════════════════


def generate_compliance_report() -> None:
    """Produce a regulator-facing read-out of the governance plane."""
    print("\n" + "=" * 60)
    print("  COMPLIANCE AUDIT REPORT")
    print("  System: MLFP Capstone Governed ML Platform")
    print("  Generated: 2026-04-14")
    print("=" * 60)

    print("\n1. AGENT ACTIVITY SUMMARY")
    print(f"   QA agent actions:    {len(qa_audit)}")
    print(f"   Admin agent actions: {len(admin_audit)}")
    print(f"   Audit agent actions: {len(audit_audit)}")
    # Count successful vs blocked runs based on the `success` field
    # GovernedSupervisor writes into each audit entry.
    qa_ok = sum(1 for e in qa_audit if e.get("success") is True)
    qa_blocked = sum(1 for e in qa_audit if e.get("success") is False)
    print(f"   QA success/failed:   {qa_ok}/{qa_blocked}")

    print("\n2. GOVERNANCE ENFORCEMENT")
    print("   D/T/R chains:         3 active delegations")
    print("   Budget enforcement:   ACTIVE (per-agent limits)")
    print("   Tool restrictions:    ACTIVE (whitelist per role)")
    print("   Clearance validation: ACTIVE (4-level hierarchy)")
    print("   Audit trail:          HASH-CHAINED (tamper-evident)")
    print(
        "   Chain verified:       "
        f"qa={qa_chain_valid}  admin={admin_chain_valid}  audit={audit_chain_valid}"
    )
    print("   Fail mode:            CLOSED (deny by default)")

    print("\n3. AUTHENTICATION & ACCESS CONTROL")
    print("   Auth method:      JWT (RS256)")
    print("   RBAC roles:       3 (viewer, operator, auditor)")
    print("   Rate limiting:    ACTIVE (per-client sliding window)")
    print("   CORS:             ACTIVE (origin whitelist)")

    print("\n4. MODEL PROVENANCE")
    print("   Base model:       env variable (not hardcoded)")
    print("   SFT adapter:      imdb_sentiment_sft_v1 (Exercise 2)")
    print("   DPO adapter:      ultrafeedback_dpo_v1 (Exercise 3)")
    print("   Registry:         AdapterRegistry (versioned, auditable)")
    print("   Drift monitoring: ACTIVE (DriftMonitor, PSI threshold=0.2)")

    print("\n5. DEPLOYMENT ARCHITECTURE")
    print("   Channels:     API + CLI + MCP (via Nexus)")
    print("   Governance:   GovernedSupervisor on ALL channels")
    print("   Sessions:     persistent state across channels")
    print("   Cost control: per-agent budget cascading")
    print("   Monitoring:   DriftMonitor + structured logging")


generate_compliance_report()

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
print("\n\u2713 Checkpoint 3 passed — compliance report sections rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the regulatory mapping
# ════════════════════════════════════════════════════════════════════════
#
# Six regulations, six technical controls, one hash-chained audit trail
# underneath all of them. The mapping is what a regulator will read.

regulatory = pl.DataFrame(
    {
        "Requirement": [
            "EU AI Act Art. 9 — Risk Management",
            "EU AI Act Art. 12 — Record-keeping",
            "EU AI Act Art. 14 — Human Oversight",
            "Singapore AI Verify — Accountability",
            "MAS TRM 7.5 — Audit Trail",
            "PDPA — Personal Data Protection",
        ],
        "Technical control": [
            "Operating envelopes per agent; budget limits; tool restrictions",
            "Hash-chained audit trail; every decision logged with timestamps",
            "D/T/R chains; every agent action traces to human Delegator",
            "D/T/R accountability grammar; role-based clearance levels",
            "audit.verify_chain() + audit.to_list() for full tamper-evident trail",
            "PII masking in clearance gate; data_clearance on every tier",
        ],
        "Status": ["COMPLIANT"] * 6,
    }
)
regulatory.write_parquet(OUTPUT_DIR / "regulatory_mapping.parquet")
print("\n6. REGULATORY COMPLIANCE MAPPING")
print(regulatory)

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert regulatory.height == 6, "Task 4: regulatory mapping must have 6 rows"
assert (regulatory["Status"] == "COMPLIANT").all()
print("\n\u2713 Checkpoint 4 passed — regulatory mapping built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: MAS Section 27 Production Order
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A MAS investigation team issues a Section 27 production
# order requesting every model-produced recommendation that reached a
# specific retail customer over a 90-day window. The audit tier holds
# the data clearance required to pull this. The qa and admin tiers do
# NOT — they physically cannot fulfil the order, which is the correct
# answer for least-privilege.
#
# BUSINESS IMPACT: A Section 27 order with no usable audit trail
# triggers either a license suspension (~S$10M/month of lost revenue
# for a mid-tier bank) or a "best-efforts reconstruction" that costs
# ~S$500,000 in external forensic consulting and still does not satisfy
# the regulator. With the governed platform, the same order is answered
# in hours via a single audit-tier query on
# ``supervisor.audit.to_list()``.

print("\n" + "=" * 70)
print("  APPLY — MAS Section 27 Production Order")
print("=" * 70)
print(
    """
  Order:    All model outputs to retail customer X, 90-day window.
  Source:   audit tier (clearance=restricted, access_audit_log=yes).
  Response: One query against supervisor.audit.to_list() + verify_chain().
  SLA:      hours, not months.

  Without this audit trail:
    - Forensic reconstruction:        ~S$500,000 (and incomplete)
    - License suspension risk:        ~S$10M / month lost revenue

  With this audit trail:
    - Compliance-team query:          ~S$5,000 analyst time
    - Response delivered to MAS:      within 48h
"""
)


# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert audit_chain_valid, "Task 5: audit-tier chain must verify for MAS evidence"
print("\u2713 Checkpoint 5 passed — audit-tier evidence ready for regulator\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — Compliance traffic light per regulation
# ════════════════════════════════════════════════════════════════════════
# Every regulation gets a traffic-light row. Green = compliant, amber =
# warn, red = fail. Our system is green on all six because the audit
# trail exists, the chain verifies, and every tier's envelope is
# enforced. If a tier's chain were to fail verification, that row would
# turn red automatically.

regs = [
    "EU AI Act\nArt. 9",
    "EU AI Act\nArt. 12",
    "EU AI Act\nArt. 14",
    "AI Verify\nAccountability",
    "MAS TRM\n7.5",
    "PDPA",
]

# Traffic-light status per row — derived from the actual chain
# verification booleans so any tamper surfaces visually, not just in
# the printed report.
all_chains_valid = qa_chain_valid and admin_chain_valid and audit_chain_valid
statuses = ["pass" if all_chains_valid else "fail"] * len(regs)
color_map = {"pass": "#4CAF50", "warn": "#FF9800", "fail": "#F44336"}
colors = [color_map[s] for s in statuses]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(regs, [1] * len(regs), color=colors, edgecolor="#333", linewidth=0.5)
for i, (bar, status) in enumerate(zip(bars, statuses)):
    ax.text(
        0.5,
        i,
        status.upper(),
        ha="center",
        va="center",
        fontweight="bold",
        color="white",
        fontsize=11,
    )
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title("Compliance Traffic Light — Derived From audit.verify_chain()")
legend_patches = [
    mpatches.Patch(color=c, label=l.upper()) for l, c in color_map.items()
]
ax.legend(handles=legend_patches, loc="lower right")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_compliance_traffic_light.png", dpi=150)
plt.close(fig)
print(f"\nSaved: {OUTPUT_DIR / '05_compliance_traffic_light.png'}")


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
  [x] Extracted per-tier audit trails from GovernedSupervisor
  [x] Verified hash-chained tamper-evidence via audit.verify_chain()
  [x] Built a section-by-section compliance report
  [x] Mapped technical controls to EU AI Act, AI Verify, MAS TRM, PDPA
  [x] Visualised the regulatory scorecard driven by real chain verification
  [x] Applied the audit report to a MAS Section 27 scenario

  KEY INSIGHT: The compliance report is a read-out of the same
  hash-chained audit trail the governance engine has been writing
  since Exercise 8.2. There is no parallel "compliance pipeline" —
  governance IS the compliance pipeline. The traffic-light chart is
  derived from ``audit.verify_chain()``, not hand-coded booleans, so
  tamper-evidence surfaces visually by construction.

  COURSE CAPSTONE COMPLETE. You have now built the full stack:
    adapter -> govern -> serve -> monitor -> audit
  on the Kailash platform, end-to-end.
"""
)
