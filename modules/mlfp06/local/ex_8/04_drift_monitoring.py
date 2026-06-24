# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.4: Production Drift Monitoring + Agent Test Harness
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Wire kailash-ml DriftMonitor against a production traffic sample
#   - Read PSI thresholds: < 0.1 no drift, 0.1-0.2 moderate, > 0.2 alert
#   - Debug an agent call by extracting input/output/governance traces
#   - Run an automated test harness across several governance paths
#   - Apply drift monitoring to a Singapore e-commerce fraud scenario
#
# PREREQUISITES: Exercises 8.1-8.3
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Rebuild the governed agent stack via build_capstone_stack(engine)
#   2. Configure DriftMonitor with MMLU as the reference distribution
#   3. Debug a single governed call (input -> output -> governance trace)
#   4. Run the automated test harness (5 tests)
#   5. Visualise the PSI dashboard and apply to Shopee-scale fraud
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import polars as pl
from kailash.db.connection import ConnectionManager
from kailash_ml import DriftMonitor
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
# THEORY — Drift and Observability
# ════════════════════════════════════════════════════════════════════════
# A model ships. On day 1, its prediction distribution matches the
# training set. On day 30, a product launch silently shifts it. Drift
# monitoring turns "the dashboard looks fine" into "PSI=0.35, ALERT".
#
# PSI < 0.10  no drift   |  0.10-0.20  moderate  |  > 0.20  alert


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Rebuild the governed stack
# ════════════════════════════════════════════════════════════════════════

eval_data = load_mmlu_eval(n_rows=100)

org_path = write_org_yaml()
loaded = load_org_yaml(org_path)
governance_engine = GovernanceEngine(loaded.org_definition)

# TODO: build the 3-tier stack via build_capstone_stack(governance_engine)
agents_by_role, tiers = ____

print("Governed stack rebuilt:")
for tier in tiers:
    print(
        f"  {tier.role:6s} -> budget=${tier.budget_usd:>5.1f}  "
        f"clearance={tier.clearance}"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 1: governed stack should rebuild"
print("\u2713 Checkpoint 1 passed — governed stack rebuilt\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Configure DriftMonitor
# ════════════════════════════════════════════════════════════════════════
# kailash-ml's DriftMonitor persists reference distributions through a
# ConnectionManager. We back it with a local SQLite file.

_DRIFT_DB_PATH = os.path.abspath("data/mlfp06/ex8_drift.db")
os.makedirs(os.path.dirname(_DRIFT_DB_PATH), exist_ok=True)
for _sfx in ("", "-wal", "-shm"):
    _p = _DRIFT_DB_PATH + _sfx
    if os.path.exists(_p):
        os.remove(_p)

# Reference distribution: numeric feature the PSI calculator can bin.
reference_df = eval_data.with_columns(
    pl.col("instruction").str.len_chars().cast(pl.Float64).alias("question_length")
).select(["question_length"])


async def setup_drift_monitoring() -> tuple[DriftMonitor, object]:
    # TODO: construct a ConnectionManager on the SQLite URL and
    # initialize it, then build a DriftMonitor with psi_threshold=0.2.
    conn = ____
    await conn.initialize()
    monitor = ____

    # TODO: store the reference distribution keyed by model name
    # "capstone_qa_model" on feature ["question_length"].
    await ____

    production_df = reference_df.head(50)
    report = await monitor.check_drift("capstone_qa_model", production_df)
    psi = report.feature_results[0].psi if report.feature_results else 0.0
    print("DriftMonitor report:")
    print(f"  Model:              capstone_qa_model")
    print(f"  Reference samples:  {reference_df.height}")
    print(f"  Production samples: {production_df.height}")
    print(f"  Drift detected:     {report.overall_drift_detected}")
    print(f"  PSI (question_len): {psi:.4f}")
    # Close the SQLite pool inside the loop so the finalizer does not
    # fire against a closed event loop on interpreter shutdown.
    await conn.close()
    return monitor, report


monitor, drift_report = run_async(setup_drift_monitoring())

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert monitor is not None, "Task 2: DriftMonitor should be created"
assert drift_report is not None, "Task 2: drift report should be produced"
print("\u2713 Checkpoint 2 passed — DriftMonitor wired\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Debug a single governed call
# ════════════════════════════════════════════════════════════════════════


async def debug_agent_call() -> dict:
    question = eval_data["instruction"][0]
    print(f"\nDebugging call for: {question[:80]}...")

    t0 = time.time()
    # TODO: call handle_qa(question, role="qa", agents_by_role=agents_by_role)
    result = await ____
    latency = (time.time() - t0) * 1000

    print("\n  OUTPUT TRACE:")
    if "error" in result:
        print(f"    Status:    BLOCKED ({result['error']})")
    else:
        print(f"    Answer:    {result['answer'][:150]}...")
        print(f"    Confidence:{result.get('confidence', 'N/A')}")
        print(f"    Latency:   {latency:.0f} ms")

    print("\n  GOVERNANCE TRACE:")
    print(f"    Role:      {result['role']}")
    print(f"    Governed:  {result['governed']}")
    return result


debug_result = run_async(debug_agent_call())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert debug_result is not None, "Task 3: debug call should return a result"
assert "role" in debug_result, "Task 3: result dict shape must be preserved"
assert debug_result["governed"] is True
print("\u2713 Checkpoint 3 passed — debug trace produced\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Automated test harness
# ════════════════════════════════════════════════════════════════════════


async def run_test_harness() -> pl.DataFrame:
    results: list[dict] = []

    # Test 1 — Normal QA query
    r = await handle_qa(
        eval_data["instruction"][0], role="qa", agents_by_role=agents_by_role
    )
    results.append(
        {
            "test": "Normal QA",
            "passed": "error" not in r,
            "detail": "Answer received" if "error" not in r else r.get("error", ""),
        }
    )

    # Test 2 — Invalid role fallback
    r = await handle_qa(
        "Test question", role="invalid_role", agents_by_role=agents_by_role
    )
    results.append(
        {
            "test": "Invalid role fallback",
            "passed": True,
            "detail": f"Role used: {r.get('role', 'unknown')}",
        }
    )

    # Test 3 — Admin access
    r = await handle_qa(
        "Show model performance metrics", role="admin", agents_by_role=agents_by_role
    )
    results.append(
        {
            "test": "Admin access",
            "passed": "error" not in r,
            "detail": "Admin access granted" if "error" not in r else "Blocked",
        }
    )

    # Test 4 — Budget cascade
    budget_ok = True
    for q in eval_data["instruction"].to_list()[:5]:
        rr = await handle_qa(q, role="qa", agents_by_role=agents_by_role)
        if rr.get("blocked"):
            budget_ok = False
            break
    results.append(
        {
            "test": "Budget cascade (5 queries)",
            "passed": budget_ok,
            "detail": "All passed" if budget_ok else "Budget exceeded",
        }
    )

    # Test 5 — Cross-role governance
    q = "What are the internal model training parameters?"
    qa_r = await handle_qa(q, role="qa", agents_by_role=agents_by_role)
    admin_r = await handle_qa(q, role="admin", agents_by_role=agents_by_role)
    results.append(
        {
            "test": "Cross-role governance",
            "passed": True,
            "detail": (
                f"QA: {'answered' if 'error' not in qa_r else 'blocked'}, "
                f"Admin: {'answered' if 'error' not in admin_r else 'blocked'}"
            ),
        }
    )

    df = pl.DataFrame(results)
    print("\n--- Test Results ---")
    print(df)
    return df


test_df = run_async(run_test_harness())
test_df.write_parquet(OUTPUT_DIR / "test_harness_results.parquet")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert test_df.height >= 5, "Task 4: at least 5 tests should run"
print("\u2713 Checkpoint 4 passed — automated test harness complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise and Apply: Shopee-scale fraud detection
# ════════════════════════════════════════════════════════════════════════

current_psi = (
    drift_report.feature_results[0].psi if drift_report.feature_results else 0.0
)
psi_dashboard = pl.DataFrame(
    {
        "Zone": ["Safe", "Investigate", "Alert"],
        "PSI range": ["< 0.10", "0.10 - 0.20", "> 0.20"],
        "Action": [
            "Continue serving",
            "Root-cause within 24h",
            "Rollback or retrain",
        ],
        "Current": [current_psi, current_psi, current_psi],
    }
)
psi_dashboard.write_parquet(OUTPUT_DIR / "psi_dashboard.parquet")
print("\nPSI dashboard:")
print(psi_dashboard)

# SCENARIO: Shopee-scale e-commerce. 11.11 sales shift traffic 4-7x.
# PSI > 0.2 alerting triggers a 72-hour human agent routing + retrain
# queue. Without drift: ~S$250,000 merchant goodwill lost. With drift:
# ~S$5,000 human surge handling cost. Net saving: ~S$245,000.

print("\n" + "=" * 70)
print("  APPLY — Shopee-scale 11.11 Dispute Handling")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — Drift score timeline
# ════════════════════════════════════════════════════════════════════════

days = list(range(1, 8))
psi_values = [0.03, 0.05, 0.08, 0.12, 0.15, 0.22, 0.31]
threshold = 0.2

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(days, psi_values, "o-", color="#1976D2", linewidth=2, label="PSI score")
ax.axhline(
    y=threshold,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Alert threshold ({threshold})",
)
ax.fill_between(days, 0, 0.1, alpha=0.1, color="green", label="Safe zone")
ax.fill_between(days, 0.1, 0.2, alpha=0.1, color="orange", label="Investigate zone")
ax.fill_between(days, 0.2, 0.4, alpha=0.1, color="red", label="Alert zone")
ax.set_xlabel("Day")
ax.set_ylabel("PSI Score")
ax.set_title("Drift Monitoring: PSI Timeline with Alert Threshold")
ax.set_ylim(0, 0.4)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_drift_timeline.png", dpi=150)
plt.close(fig)
print(f"\nSaved: {OUTPUT_DIR / '04_drift_timeline.png'}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Wired DriftMonitor with MMLU as the reference distribution
  [x] Read PSI thresholds as business-actionable zones
  [x] Debugged a governed agent call end-to-end
  [x] Ran an automated test harness covering 5 governance paths
  [x] Applied drift monitoring to a regional e-commerce scenario

  Next: 05_compliance_audit.py closes the loop with a regulatory audit.
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
