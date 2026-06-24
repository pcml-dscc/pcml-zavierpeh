# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 2.4: Model Merging Without Training
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - THEORY: why merging fine-tuned models works (task vectors)
#   - BUILD: TIES (Trim, Elect Sign, Merge) on synthetic deltas
#   - BUILD: SLERP (spherical linear interpolation) on weight tensors
#   - VISUALISE: weight-norm preservation across merge strategies
#   - APPLY: Singapore fintech — three LoRAs into one deployment
#
# PREREQUISITES: Exercises 2.1 and 2.2 (LoRA + adapters)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. THEORY: task vectors and the merge landscape
#   2. BUILD: TIES merge (trim -> elect sign -> average)
#   3. BUILD: SLERP merge (hypersphere interpolation)
#   4. VISUALISE: norm comparison across merge strategies
#   5. APPLY: Singapore fintech KYC + fraud + explainer merge
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import torch
from dotenv import load_dotenv

from shared.mlfp06.ex_2 import OUTPUT_DIR

load_dotenv()

torch.manual_seed(42)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Task Vectors and the Merge Landscape
# ════════════════════════════════════════════════════════════════════════
# Task vector: tau = W_finetuned - W_base (everything the FT added).
# Task arithmetic: W_merged = W_base + alpha*tau_A + beta*tau_B.
# TIES fixes sign conflict; DARE randomly drops and rescales; SLERP
# walks the hypersphere arc instead of the straight chord (preserves
# weight norms). All are training-free — merging is zero compute.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — BUILD: TIES merge on synthetic 128x128 deltas
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: TIES merge — trim / elect sign / average")
print("=" * 70)

delta_A = torch.randn(128, 128) * 0.1
delta_B = torch.randn(128, 128) * 0.1

# TODO: TRIM step — zero entries with |delta| < 0.05 in copies of delta_A and delta_B
trim_threshold = 0.05
delta_A_trim = ____
delta_B_trim = ____

# TODO: ELECT SIGN — sign_A, sign_B = ... ; elected_sign = (sign_A + sign_B).sign()
sign_A = ____
sign_B = ____
elected_sign = ____

# TODO: MERGE — average entries that agree with elected_sign. Use masks
# (sign == elected_sign).float() and divide by (mask_A + mask_B + 1e-8)
mask_A = ____
mask_B = ____
merged_delta = ____

print(f"Original non-zero params (delta_A): {(delta_A != 0).sum().item():,}")
print(f"After TRIM:                         {(delta_A_trim != 0).sum().item():,}")
print(f"Sign agreement rate:                {(sign_A == sign_B).float().mean():.1%}")
print(f"Merged delta Frobenius norm:        {merged_delta.norm():.4f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert merged_delta.shape == delta_A.shape, "Task 1: TIES should preserve shape"
assert (delta_A_trim != 0).sum() < (delta_A != 0).sum(), "TRIM should zero some values"
print("✓ Checkpoint 1 passed — TIES merge complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: SLERP merge on weight tensors
# ════════════════════════════════════════════════════════════════════════


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two tensors."""
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()
    # TODO: cos_theta = dot(v0_flat, v1_flat) / (norm(v0_flat) * norm(v1_flat) + 1e-8);
    # clamp to [-1,1], take arccos, guard |theta|<1e-6 with linear fallback
    cos_theta = ____
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    if theta.abs() < 1e-6:
        return (1 - t) * v0 + t * v1
    # TODO: w0 = sin((1-t)*theta)/sin(theta); w1 = sin(t*theta)/sin(theta);
    # return w0 * v0 + w1 * v1
    sin_theta = torch.sin(theta)
    w0 = ____
    w1 = ____
    return w0 * v0 + w1 * v1


print("=" * 70)
print("TASK 2: SLERP vs linear interpolation — weight norm comparison")
print("=" * 70)

W_base = torch.randn(128, 128)
W_task_A = W_base + delta_A
W_task_B = W_base + delta_B

slerp_merged = slerp(0.5, W_task_A, W_task_B)
linear_merged = 0.5 * W_task_A + 0.5 * W_task_B

norms = {
    "W_task_A": W_task_A.norm().item(),
    "W_task_B": W_task_B.norm().item(),
    "SLERP midpoint": slerp_merged.norm().item(),
    "Linear midpoint": linear_merged.norm().item(),
}
for k, v in norms.items():
    print(f"  {k:<20} norm = {v:.4f}")

mean_input_norm = 0.5 * (norms["W_task_A"] + norms["W_task_B"])
slerp_gap = abs(norms["SLERP midpoint"] - mean_input_norm)
linear_gap = abs(norms["Linear midpoint"] - mean_input_norm)

print(f"\n  SLERP deviation from mean input norm:  {slerp_gap:.4f}")
print(f"  Linear deviation from mean input norm: {linear_gap:.4f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert slerp_merged.shape == W_task_A.shape, "Task 2: SLERP should preserve shape"
assert slerp_gap <= linear_gap + 1e-6, "SLERP should preserve norms at least as well"
print("✓ Checkpoint 2 passed — SLERP preserves norms\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — VISUALISE: norm comparison across merge strategies
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Visualise merge-strategy norm preservation")
print("=" * 70)

ts = [i / 10 for i in range(11)]
# TODO: slerp_norms = [slerp(t, W_task_A, W_task_B).norm().item() for t in ts]
# TODO: linear_norms = [((1 - t) * W_task_A + t * W_task_B).norm().item() for t in ts]
slerp_norms = ____
linear_norms = ____

# TODO: Plot both curves vs t, axhlines at the input norms. Save to
# OUTPUT_DIR / "ex2_slerp_vs_linear.png"
____
fname = OUTPUT_DIR / "ex2_slerp_vs_linear.png"
print(f"  Saved: {fname}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert fname.exists(), "Task 3: plot should exist"
print("✓ Checkpoint 3 passed — norm comparison visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — APPLY: Singapore fintech — merging three LoRAs
# ════════════════════════════════════════════════════════════════════════
# A Singapore digital bank has three LoRAs (KYC, fraud, explainer) on
# the same 7B base. Three deployments = 3 x 14 GB VRAM + ~780 ms end-
# to-end latency. A TIES merge into a single deployment costs ~2 hours
# of eng time (vs ~6 weeks retraining) and drops latency + VRAM.
# Risk: merged model degrades on fraud eval; fall back to three
# deployments if SLA breaks.

print("Singapore fintech — three-LoRA merge decision:")
rows = [
    ("Option A: three deployments", 3 * 14, 780, 0),
    ("Option B: retrain multi-task", 1 * 14, 320, 18_000),
    ("Option C: TIES merge", 1 * 14, 320, 120),
]
# TODO: Build a polars DataFrame with columns Option, VRAM_GB, Latency_ms, Upfront_cost_SGD
decision_df = ____
print(decision_df)

monthly_saving_sgd = 99 + 4_100
# TODO: annual_saving_sgd = monthly_saving_sgd * 12
annual_saving_sgd = ____
print(f"\n  Monthly saving (option C vs A): S${monthly_saving_sgd:,}")
print(f"  Annual saving:                  S${annual_saving_sgd:,}")
print(f"  Recommended: TIES merge (option C) with per-task eval gate")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert annual_saving_sgd > 0, "Task 4: fintech should see positive savings"
print("✓ Checkpoint 4 passed — fintech merge decision analysed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built TIES merge: trim -> elect sign -> merge
  [x] Built SLERP merge: spherical interpolation on the hypersphere
  [x] Visualised that SLERP preserves weight norms better than linear
  [x] Applied merging to a Singapore fintech three-LoRA scenario
      (~S$50k/year saving vs separate deployments, zero retraining)

  KEY INSIGHT: merging is free compute. TIES handles sign conflict,
  SLERP handles norm drift, and task arithmetic composes them for
  targeted capability addition or removal.

  Next: 05_quantisation.py surveys how we shrink the merged model
  for deployment on smaller hardware.
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
