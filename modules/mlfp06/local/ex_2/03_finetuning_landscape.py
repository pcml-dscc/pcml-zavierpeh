# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 2.3: Fine-Tuning Landscape Survey
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Survey all 10 fine-tuning techniques by mechanism and param cost
#   - Build a decision tree for picking the right technique
#   - Visualise the param-efficiency landscape
#   - Apply the decision tree to a Singapore healthcare privacy scenario
#
# PREREQUISITES: Exercises 2.1 and 2.2 (LoRA + adapters)
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. THEORY: the 10 techniques in one table
#   2. BUILD: decision tree as a plain function
#   3. VISUALISE: param-cost landscape plot
#   4. APPLY: Singapore hospital — differential privacy SFT decision
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from dotenv import load_dotenv

from shared.mlfp06.ex_2 import OUTPUT_DIR

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# THEORY — The 10 Techniques
# ════════════════════════════════════════════════════════════════════════
# Fine-tuning methods trade three knobs: param cost, quality ceiling,
# and constraints (privacy, forgetting, multi-task reuse).

# TODO: Build a list of dicts with keys {name, mechanism, params_pct, when}
# for all 10 techniques: LoRA, Adapter Layers, Prefix Tuning, Prompt Tuning,
# Full Fine-Tuning, LLRD, Progressive Freezing, Knowledge Distillation,
# DP-SGD, EWC. See the theory notes and solution for canonical values.
techniques = ____

techniques_df = pl.DataFrame(techniques)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Print the landscape table
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Fine-tuning landscape — 10 techniques")
print("=" * 70)
print(techniques_df)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert techniques_df.height == 10, "Task 1: should cover all 10 techniques"
print("✓ Checkpoint 1 passed — 10 techniques surveyed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: decision tree as a function
# ════════════════════════════════════════════════════════════════════════


def recommend_technique(
    *,
    has_sensitive_data: bool,
    gpu_memory_gb: float,
    model_size_b: float,
    num_tasks: int,
    small_dataset: bool,
    need_catastrophic_forgetting_guard: bool,
) -> str:
    """Return a recommended fine-tuning technique for the given constraints."""
    # TODO: if has_sensitive_data -> "DP-SGD"
    # TODO: if need_catastrophic_forgetting_guard and num_tasks > 1 -> "EWC"
    # TODO: if model_size_b * 8 > gpu_memory_gb -> "LoRA"  (FT needs ~8x model size)
    # TODO: if num_tasks > 3 -> "Adapter Layers"
    # TODO: if small_dataset -> "Progressive Freezing"
    # TODO: else -> "Full Fine-Tuning"
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: exercise the decision tree on 5 scenarios
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Decision tree against 5 scenarios")
print("=" * 70)

scenarios = [
    dict(
        name="Hospital notes classifier",
        has_sensitive_data=True,
        gpu_memory_gb=80,
        model_size_b=7,
        num_tasks=1,
        small_dataset=False,
        need_catastrophic_forgetting_guard=False,
    ),
    dict(
        name="12-tenant SaaS tone adaptation",
        has_sensitive_data=False,
        gpu_memory_gb=24,
        model_size_b=7,
        num_tasks=12,
        small_dataset=False,
        need_catastrophic_forgetting_guard=False,
    ),
    dict(
        name="Research lab 1B model, 40 GB GPU",
        has_sensitive_data=False,
        gpu_memory_gb=40,
        model_size_b=1,
        num_tasks=1,
        small_dataset=False,
        need_catastrophic_forgetting_guard=False,
    ),
    dict(
        name="Continual-learning robotics agent",
        has_sensitive_data=False,
        gpu_memory_gb=40,
        model_size_b=3,
        num_tasks=5,
        small_dataset=False,
        need_catastrophic_forgetting_guard=True,
    ),
    dict(
        name="Small legal firm, 200 examples",
        has_sensitive_data=False,
        gpu_memory_gb=40,
        model_size_b=3,
        num_tasks=1,
        small_dataset=True,
        need_catastrophic_forgetting_guard=False,
    ),
]

for s in scenarios:
    name = s.pop("name")
    rec = recommend_technique(**s)
    print(f"  {name:<40} -> {rec}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert (
    recommend_technique(
        has_sensitive_data=True,
        gpu_memory_gb=80,
        model_size_b=7,
        num_tasks=1,
        small_dataset=False,
        need_catastrophic_forgetting_guard=False,
    )
    == "DP-SGD"
), "Sensitive data should route to DP-SGD"
print("✓ Checkpoint 2 passed — decision tree covers 5 scenarios\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: params-vs-use-case landscape
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Visualise the landscape (log scale)")
print("=" * 70)

names = techniques_df["name"].to_list()
pcts = techniques_df["params_pct"].to_list()

# TODO: Horizontal log-scale bar plot of names vs pcts, coloured by band
# (<10 steelblue, <80 darkorange, else crimson). Annotate each bar with
# the percentage. Save to OUTPUT_DIR / "ex2_finetuning_landscape.png".
____
fname = OUTPUT_DIR / "ex2_finetuning_landscape.png"
print(f"  Saved: {fname}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert fname.exists(), "Landscape plot should exist"
print("✓ Checkpoint 3 passed — landscape visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore hospital — differential-privacy SFT
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore hospital wants to fine-tune a 7B model on
# 250,000 de-identified discharge summaries under PDPA. De-identified
# is not enough — LLMs memorise. The Data Protection Officer mandates
# a provable (epsilon, delta)-DP guarantee via DP-SGD.
# Trade-off: ~20% slower training, ~1-3 quality points lost vs plain SFT.
# Benefit: formal bound on single-patient influence on final weights.

print("Singapore hospital — DP-SGD decision:")
nurses = 12
overtime_hours_saved_weekly = nurses * 6
overtime_rate_sgd = 130

# TODO: weekly_saving = overtime_hours_saved_weekly * overtime_rate_sgd
# TODO: annual_saving = weekly_saving * 52
weekly_saving = ____
annual_saving = ____
print(f"  Nurses on night shift:      {nurses}")
print(f"  Overtime hours saved/week:  {overtime_hours_saved_weekly}")
print(f"  Weekly saving:              S${weekly_saving:,}")
print(f"  Annual saving (2 sites):    S${annual_saving * 2:,}")
print(f"  Recommended technique:      DP-SGD (optionally + LoRA)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert annual_saving > 0, "Task 5: hospital should see positive savings"
print("✓ Checkpoint 4 passed — hospital cost/benefit analysed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Surveyed all 10 fine-tuning techniques (LoRA, adapters, prefix,
      prompt, full FT, LLRD, progressive freezing, distillation,
      DP-SGD, EWC)
  [x] Built a decision tree mapping constraints to techniques
  [x] Visualised the parameter-cost landscape on a log scale
  [x] Applied the tree to a Singapore hospital DP-SGD scenario
      (~S$960k/year saving across two sites)

  KEY INSIGHT: there is no "best" fine-tuning technique. Only the
  best technique FOR a specific dataset size, GPU budget, privacy
  constraint, and number of tasks.

  Next: 04_model_merging.py combines fine-tuned models without any
  additional training.
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

# Primary lens: Alignment (KL divergence from base, reward margin).
# Secondary: Output (judge quality on paired completions), Attention
# (layer-wise shift in target modules for LoRA).
if False:  # scaffold — requires trained base + adapter checkpoint
    obs = LLMObservatory(run_id="ex_2_finetune_run")
    # Typical alignment read:
    # for step, metrics in enumerate(training_log):
    #     obs.alignment.log_training_step(step=step, **metrics)
    # obs.alignment.evaluate_pair(base_responses, adapter_responses)
    print("\n── LLM Observatory Report ──")
    findings = obs.report()

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [!] Alignment  (WARNING): KL divergence from base = 0.42 nats
#       Fix: healthy range 0.2-1.0; this is low-end — adapter barely
#            moved. Increase LoRA rank or learning rate.
#   [✓] Output     (HEALTHY): judge win-rate 0.58 vs base (>0.50 = good)
#   [✓] Attention  (HEALTHY): shift concentrated in q_proj/v_proj as
#       expected for LoRA; no drift in frozen layers.
#   [?] Retrieval / Agent / Governance (n/a)
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [ALIGNMENT LENS] KL 0.42 nats is the SIGNATURE of a cautiously-trained
#     LoRA adapter — it diverged from the base distribution but not
#     enough to break it. Above 2.0 nats signals over-fit; below 0.2
#     signals the adapter barely learned. Our value is slightly under the
#     0.5 floor we want for visible task lift.
#     >> Prescription: raise lora_r from 8 -> 16 or train another epoch.
#  [OUTPUT LENS] Win-rate 0.58 > 0.50 confirms the adapter is better
#     than base on held-out prompts — tiny lift but statistically real.
#  [ATTENTION LENS] Shift localised in the target modules = LoRA is
#     doing what it's supposed to do (low-rank delta on attention
#     projections, frozen MLP). If attention shifted everywhere you'd
#     know you accidentally unfroze a module.
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
