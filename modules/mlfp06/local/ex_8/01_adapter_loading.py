# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.1: Load Fine-Tuned Model from AdapterRegistry
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Use kailash-align's AdapterRegistry as the source of truth for
#     model provenance and version control
#   - Load a LoRA adapter through AlignmentPipeline (SFT + DPO merge)
#   - Fall back safely when a specific adapter is unavailable
#   - Visualise the adapter catalogue as a registry table
#   - Apply adapter loading to a Singapore HR compliance scenario
#
# PREREQUISITES: MLFP06 Ex 2 (SFT LoRA), Ex 3 (DPO alignment)
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Load MMLU evaluation data for downstream monitoring
#   2. Query AdapterRegistry and pick the best available adapter
#   3. Instantiate AlignmentPipeline in inference mode
#   4. Visualise the adapter catalogue
#   5. Apply to a Singapore HR compliance QA scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from kailash_align import AdapterRegistry, AlignmentConfig, AlignmentPipeline

from shared.mlfp06.ex_8 import (
    OUTPUT_DIR,
    load_mmlu_eval,
    run_async,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Why an AdapterRegistry?
# ════════════════════════════════════════════════════════════════════════
# A production LLM platform trains many adapters: SFT for domain tone,
# DPO for preference alignment, SLERP merges for combining both. The
# AdapterRegistry is the Git for model weights — it stores what each
# adapter was trained on, which base model it attaches to, who owns
# it, and how recently it was evaluated. Hardcoding a filesystem path
# silently ships the wrong weights tomorrow; the registry turns model
# loading into a named lookup against an audited catalogue.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load MMLU evaluation data
# ════════════════════════════════════════════════════════════════════════

# TODO: Call load_mmlu_eval with n_rows=100
eval_data = ____

print(f"\nEvaluation data (MMLU): {eval_data.shape}")
print(f"Subjects: {eval_data['subject'].n_unique()}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert eval_data.height > 0, "Task 1: MMLU should load at least 1 row"
assert "instruction" in eval_data.columns, "Task 1: expected 'instruction' column"
print("\u2713 Checkpoint 1 passed — evaluation data loaded\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Query AdapterRegistry and pick the best adapter
# ════════════════════════════════════════════════════════════════════════


async def pick_best_adapter() -> tuple[AdapterRegistry, dict]:
    """Return the registry and the best available adapter metadata."""
    # TODO: instantiate AdapterRegistry()
    registry = ____

    # TODO: call registry.list_adapters() (async)
    adapters = ____
    print(f"Available adapters: {len(adapters)}")

    best_adapter: dict = {}
    for candidate in (
        "sg_domain_slerp_merge_v1",
        "ultrafeedback_dpo_v1",
        "imdb_sentiment_sft_v1",
    ):
        try:
            # TODO: call registry.get_adapter(candidate) (async)
            found = ____
            if found:
                best_adapter = found
                break
        except Exception:
            continue
    return registry, best_adapter


registry, best_adapter = run_async(pick_best_adapter())

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert registry is not None, "Task 2: registry should be accessible"
print(f"\u2713 Checkpoint 2 passed — adapter: {best_adapter.get('name', 'none')}\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the AlignmentPipeline for inference
# ════════════════════════════════════════════════════════════════════════

# TODO: Build an AlignmentPipeline with an AlignmentConfig whose
#       method="inference" and adapter_path=best_adapter.get("adapter_path", "")
inference_pipeline = ____

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert inference_pipeline is not None, "Task 3: pipeline should be created"
print("\u2713 Checkpoint 3 passed — AlignmentPipeline ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the adapter catalogue
# ════════════════════════════════════════════════════════════════════════


async def snapshot_catalogue() -> pl.DataFrame:
    # TODO: list adapters, return as polars DataFrame with
    #       columns: name, method, base_model
    ____


catalogue = run_async(snapshot_catalogue())
catalogue.write_parquet(OUTPUT_DIR / "adapter_catalogue.parquet")
print("Adapter catalogue:")
print(catalogue)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore HR Compliance QA
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore SME (200 employees) runs an internal HR policy
# assistant fine-tuned on MOM guidelines. The base model is general; the
# SFT adapter teaches Singapore employment law (CPF, Work Pass, notice);
# DPO alignment filters out legally risky phrasing.
#
# BUSINESS IMPACT: Legal review externally is ~S$400/query. A governed
# adapter serving 200 queries/month at S$0.08/query = S$16 vs S$80,000
# — a ~5,000x reduction, with full provenance from the registry.

print("\n" + "=" * 70)
print("  APPLY — Singapore HR Policy Assistant")
print("=" * 70)
print(
    f"""
  Active adapter: {best_adapter.get('name', 'none')}
  Method:         {best_adapter.get('method', '-')}

  Legal-review baseline:  S$80,000/month
  Governed-adapter cost:  S$16/month
  Saving:                 ~5,000x reduction
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Queried AdapterRegistry as the catalogue of trained adapters
  [x] Chose the best available adapter with graceful fallback
  [x] Built an AlignmentPipeline for inference
  [x] Snapshotted the registry for rollback visibility
  [x] Applied adapter loading to a Singapore HR scenario

  Next: 02_governance_pipeline.py wraps this adapter in PACT controls.
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
