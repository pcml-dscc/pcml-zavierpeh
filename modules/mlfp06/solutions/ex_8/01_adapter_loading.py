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
#   - Load a LoRA adapter through AlignmentServing (SFT + DPO merge)
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
#   3. Build an AlignmentServing stack bound to the adapter registry
#   4. Visualise the adapter catalogue
#   5. Apply to a Singapore HR compliance QA scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl

from kailash_align import AdapterRegistry
from kailash_align.config import ServingConfig
from kailash_align.serving import AlignmentServing

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
# adapter was trained on, which base model it attaches to, who owns it,
# and how recently it was evaluated.
#
# Hardcoding a path like "./models/imdb_sft_v1" is the autoencoder-style
# identity mistake of deployment: it looks fine today and silently
# ships the wrong weights tomorrow. The registry turns model loading
# into a named lookup against an audited catalogue.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load MMLU evaluation data
# ════════════════════════════════════════════════════════════════════════

eval_data = load_mmlu_eval(n_rows=100)

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
    registry = AdapterRegistry()
    adapters = await registry.list_adapters()
    print(f"Available adapters: {len(adapters)}")
    for a in adapters:
        print(f"  {a.get('name', '?'):35s} method={a.get('method', '?')}")

    # Preferred order: SLERP merge (Ex 2+3 combined) > DPO > SFT.
    best_adapter: dict = {}
    for candidate in (
        "sg_domain_slerp_merge_v1",
        "ultrafeedback_dpo_v1",
        "imdb_sentiment_sft_v1",
    ):
        try:
            found = await registry.get_adapter(candidate)
            if found:
                best_adapter = found
                break
        except Exception:
            continue

    if not best_adapter:
        print("  Note: no prior adapter found; running un-adapted.")
    return registry, best_adapter


registry, best_adapter = run_async(pick_best_adapter())

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert registry is not None, "Task 2: registry should be accessible"
print(
    f"\u2713 Checkpoint 2 passed — selected adapter: {best_adapter.get('name', 'none')}\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the AlignmentServing stack for deployment
# ════════════════════════════════════════════════════════════════════════
# kailash-align 0.3+ moved the inference / deployment path out of
# AlignmentPipeline (which is training-only: sft, dpo, kto, orpo, grpo,
# ppo, rloo, online_dpo, sft_then_dpo) and into AlignmentServing, which
# handles GGUF export, Ollama, and vLLM targets. Adapter lookup still
# comes from the AdapterRegistry; AlignmentServing consumes the registry
# by reference so `deploy(adapter_name=...)` resolves the path at call
# time instead of being baked into the config. This is the canonical
# "load-for-inference" shape in the modern align stack.

serving_config = ServingConfig(
    target="ollama",
    quantization="q4_k_m",
    validate_gguf=False,  # skip GGUF validation in course smoke-test
)
serving = AlignmentServing(adapter_registry=registry, config=serving_config)

print(f"Built AlignmentServing stack (target={serving_config.target})")
print(f"  Resolved adapter from registry: {best_adapter.get('name', 'none')}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert serving is not None, "Task 3: serving stack should be created"
print("\u2713 Checkpoint 3 passed — AlignmentServing ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the adapter catalogue
# ════════════════════════════════════════════════════════════════════════


async def snapshot_catalogue() -> pl.DataFrame:
    adapters = await registry.list_adapters()
    if not adapters:
        return pl.DataFrame({"name": ["(none)"], "method": ["-"], "base_model": ["-"]})
    return pl.DataFrame(
        {
            "name": [a.get("name", "?") for a in adapters],
            "method": [a.get("method", "?") for a in adapters],
            "base_model": [a.get("base_model", "?") for a in adapters],
        }
    )


catalogue = run_async(snapshot_catalogue())
catalogue.write_parquet(OUTPUT_DIR / "adapter_catalogue.parquet")
print("Adapter catalogue:")
print(catalogue)

# INTERPRETATION: The catalogue is the single pane of glass operations
# teams use for rollback. If a new adapter ships broken, they pick the
# previous row in this table and redeploy. Without the registry, this
# roll-back is a filesystem archaeology dig.


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Adapter parameter count comparison
# ════════════════════════════════════════════════════════════════════════
# Visual proof of the parameter efficiency story: SFT/DPO adapters train
# only a fraction of the base model's parameters. The bar chart makes
# the magnitude difference viscerally obvious.

adapter_names = catalogue["name"].to_list()
# Simulated parameter counts (LoRA adapters are ~0.1-2% of base)
base_params = 7_000_000_000  # 7B base
adapter_params = {
    "imdb_sentiment_sft_v1": 4_200_000,
    "ultrafeedback_dpo_v1": 4_200_000,
    "sg_domain_slerp_merge_v1": 8_400_000,
}
param_counts = [adapter_params.get(n, 4_200_000) for n in adapter_names]

fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#3498db", "#2ecc71", "#e67e22"][: len(adapter_names)]
bars = ax.bar(adapter_names, param_counts, color=colors)
ax.axhline(
    base_params,
    color="#e74c3c",
    linestyle="--",
    linewidth=2,
    label=f"Base model: {base_params / 1e9:.0f}B params",
)
ax.set_ylabel("Trainable parameters")
ax.set_title("Adapter Parameter Count vs Base Model", fontweight="bold")
ax.set_yscale("log")
for bar, count in zip(bars, param_counts):
    pct = count / base_params * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        count * 1.5,
        f"{count / 1e6:.1f}M\n({pct:.2f}%)",
        ha="center",
        fontsize=9,
    )
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, which="both")
plt.tight_layout()
fname = OUTPUT_DIR / "ex8_adapter_params.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore HR Compliance QA
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore SME (200 employees) runs an internal HR policy
# assistant fine-tuned on MOM (Ministry of Manpower) guidelines. The
# base model is general; the SFT adapter teaches Singapore employment
# law nuances (CPF, Work Pass categories, retrenchment notice). DPO
# alignment then filters out speculative or legally risky phrasing.
#
# BUSINESS IMPACT: Legal review of HR guidance costs ~S$400/query at
# an external firm. A governed adapter serving 200 queries/month at
# S$0.08/query = S$16 vs S$80,000 — a 5,000x cost reduction, with the
# registry ensuring the HR team can prove WHICH adapter answered WHICH
# query for every compliance audit.

print("\n" + "=" * 70)
print("  APPLY — Singapore HR Policy Assistant")
print("=" * 70)
print(
    f"""
  Base model:     {best_adapter.get('base_model', 'env $DEFAULT_LLM_MODEL')}
  Active adapter: {best_adapter.get('name', 'none')}
  Method:         {best_adapter.get('method', '-')}

  Legal-review baseline:  S$80,000/month (200 queries at S$400 each)
  Governed-adapter cost:  S$16/month (200 queries at S$0.08 each)
  Saving:                 S$79,984/month (~5,000x reduction)

  Audit trail: every query logs the adapter name + version, so an
  MOM inspector can reconstruct exactly which model answered any
  policy question.
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
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Queried AdapterRegistry as the catalogue of trained adapters
  [x] Chose the best available adapter with graceful fallback
  [x] Built an AlignmentServing stack for deployment (GGUF / Ollama / vLLM)
  [x] Snapshotted the registry to a parquet for rollback visibility
  [x] Applied adapter loading to a Singapore HR compliance scenario

  KEY INSIGHT: The registry is the ONE structural defence against
  "which model is in prod right now?" panic. Everything downstream
  (governance, nexus, drift, audit) assumes you can answer that
  question in one query.

  Next: 02_governance_pipeline.py wraps this adapter in PACT controls.
"""
)
