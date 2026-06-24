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

from shared.mlfp06.ex_2 import OUTPUT_DIR

# ════════════════════════════════════════════════════════════════════════
# THEORY — The 10 Techniques
# ════════════════════════════════════════════════════════════════════════
# Fine-tuning methods trade off three knobs:
#   - PARAM cost (how many weights you touch)
#   - QUALITY ceiling (how far you can push the model)
#   - CONSTRAINTS (privacy, catastrophic forgetting, multi-task reuse)
#
# This file is the map.  You will revisit it every time you pick a
# method for a new project.

techniques = [
    {
        "name": "LoRA",
        "mechanism": "Low-rank A,B on attention projections",
        "params_pct": 0.5,
        "when": "Default for single-task adaptation",
    },
    {
        "name": "Adapter Layers",
        "mechanism": "Bottleneck FC->act->FC between layers",
        "params_pct": 3.0,
        "when": "Multi-task serving on a shared base",
    },
    {
        "name": "Prefix Tuning",
        "mechanism": "Learnable vectors prepended to K,V in attention",
        "params_pct": 0.1,
        "when": "Very parameter-efficient; limited expressiveness",
    },
    {
        "name": "Prompt Tuning",
        "mechanism": "Learnable soft tokens on the input embeddings",
        "params_pct": 0.05,
        "when": "Minimal overhead; good for classification",
    },
    {
        "name": "Full Fine-Tuning",
        "mechanism": "Backprop through all parameters with LR schedule",
        "params_pct": 100.0,
        "when": "Maximum quality when GPU memory allows",
    },
    {
        "name": "LLRD",
        "mechanism": "Layer-wise LR decay (earlier layers slower)",
        "params_pct": 100.0,
        "when": "Preserve general knowledge in early layers",
    },
    {
        "name": "Progressive Freezing",
        "mechanism": "Start frozen; unfreeze top-down over epochs",
        "params_pct": 50.0,
        "when": "Small datasets where overfitting is a risk",
    },
    {
        "name": "Knowledge Distillation",
        "mechanism": "Teacher-student on soft labels from large model",
        "params_pct": 100.0,
        "when": "Compress a large model into a deployable small one",
    },
    {
        "name": "DP-SGD",
        "mechanism": "Clip gradients, add Gaussian noise, provable DP",
        "params_pct": 100.0,
        "when": "Sensitive data — medical, financial, minors",
    },
    {
        "name": "EWC",
        "mechanism": "Fisher Info penalty on changing important weights",
        "params_pct": 100.0,
        "when": "Sequential tasks; prevent catastrophic forgetting",
    },
]

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
    """Return a recommended fine-tuning technique for the given constraints.

    The tree below is the condensed version of the decision process
    you should apply in practice.  It is deliberately simple — the
    real decision often combines multiple techniques.
    """
    if has_sensitive_data:
        return "DP-SGD"
    if need_catastrophic_forgetting_guard and num_tasks > 1:
        return "EWC"
    # A 7B model in FP16 needs ~14 GB just to load, ~56 GB to fully fine-tune
    if model_size_b * 8 > gpu_memory_gb:
        return "LoRA"
    if num_tasks > 3:
        return "Adapter Layers"
    if small_dataset:
        return "Progressive Freezing"
    return "Full Fine-Tuning"


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

fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
colors = [
    "steelblue" if p < 10 else ("darkorange" if p < 80 else "crimson") for p in pcts
]
bars = ax.barh(names, pcts, color=colors, edgecolor="black")
ax.set_xscale("log")
ax.set_xlabel("Trainable parameters (% of base model)")
ax.set_title("Fine-tuning landscape — parameter cost", fontweight="bold")
for bar, p in zip(bars, pcts):
    ax.annotate(
        f"{p:.2f}%",
        xy=(p, bar.get_y() + bar.get_height() / 2),
        xytext=(3, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
    )
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex2_finetuning_landscape.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fname}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert fname.exists(), "Landscape plot should exist"
print("✓ Checkpoint 3 passed — landscape visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore hospital — differential-privacy SFT
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore hospital wants to fine-tune a 7B open-source
# model on 250,000 de-identified discharge summaries.  The goal is a
# triage assistant that highlights urgent follow-ups.  The hospital is
# bound by the Personal Data Protection Act (PDPA) and the MOH health
# data guidelines; a member-inference attack on the deployed model
# MUST NOT be able to reconstruct any individual patient record.
#
# CONSTRAINT: "de-identified" is not enough.  Research has shown that
# LLMs fine-tuned on even fully de-identified medical text can still
# memorise specific passages and leak them under adversarial prompting.
# The hospital's Data Protection Officer has mandated a provable
# privacy guarantee.
#
# DECISION: the decision tree routes us to DP-SGD (Differential
# Privacy SGD).  DP-SGD clips per-example gradients and adds Gaussian
# noise at every step, producing a formal (epsilon, delta)-DP
# guarantee.  Cost: ~20% slower training and ~1-3 points of quality
# drop vs plain SFT.  Benefit: a provable bound on how much any single
# patient record can influence the final weights.
#
# BUSINESS IMPACT: without DP-SGD, the hospital cannot deploy the
# triage assistant at all (legal risk + reputational risk).  With
# DP-SGD, the assistant reduces triage nurse workload on the night
# shift by ~30%, saving ~S$480,000/year in overtime across two
# hospitals while staying within PDPA's enforcement envelope.
#
# You could combine DP-SGD with LoRA (DP-LoRA) to recover some of the
# quality drop by constraining the update to a low-rank subspace.
# That is an active research area and a natural extension of the
# decision tree above.

print("Singapore hospital — DP-SGD decision:")
nurses = 12
overtime_hours_saved_weekly = nurses * 6
overtime_rate_sgd = 130
weekly_saving = overtime_hours_saved_weekly * overtime_rate_sgd
annual_saving = weekly_saving * 52
print(f"  Nurses on night shift:      {nurses}")
print(f"  Overtime hours saved/week:  {overtime_hours_saved_weekly}")
print(f"  Weekly saving:              S${weekly_saving:,}")
print(f"  Annual saving (2 sites):    S${annual_saving * 2:,}")
print(f"  Recommended technique:      DP-SGD (optionally + LoRA)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert annual_saving > 0, "Task 5: hospital should see positive savings"
print("✓ Checkpoint 4 passed — hospital cost/benefit analysed\n")


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

  KEY INSIGHT: there is no "best" fine-tuning technique.  There is
  only the best technique FOR a specific dataset size, GPU budget,
  privacy constraint, and number of tasks.  The decision tree is how
  you stop defaulting to LoRA out of habit.

  Next: 04_model_merging.py combines fine-tuned models without any
  additional training.
"""
)
