# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 2.5: Quantisation — Shrinking Models for Deployment
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - THEORY: why quantisation works (weight distributions are compressible)
#   - BUILD: naive INT8 symmetric quantisation from scratch
#   - TRAIN: measure the round-trip error across a weight tensor
#   - VISUALISE: memory / quality / speed trade-off across methods
#   - APPLY: Singapore SME on-device chatbot — CPU deployment with GGUF
#
# PREREQUISITES: Exercise 2.4 (model merging)
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. THEORY: GPTQ / AWQ / GGUF / QLoRA in one table
#   2. BUILD: INT8 symmetric quantiser (from scratch)
#   3. TRAIN: measure round-trip quantisation error
#   4. VISUALISE: memory footprint across precisions
#   5. APPLY: Singapore SME on-device assistant (GGUF on CPU)
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
# THEORY — Why Quantisation Works
# ════════════════════════════════════════════════════════════════════════
# Transformer weights are bell-shaped — most values cluster near 0.
# FP16 spends 16 bits everywhere regardless. Quantisation maps the
# continuous range onto an integer grid: INT8 (256 levels), INT4 (16),
# or NF4 (16 levels laid out to match the normal distribution).
# GPTQ uses the Hessian; AWQ protects salient weights; GGUF is CPU-
# optimised; QLoRA quantises the frozen base and trains LoRA on top.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Print the quantisation landscape table
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Quantisation landscape — GPTQ / AWQ / GGUF / QLoRA")
print("=" * 70)

# TODO: Build a polars DataFrame with columns:
# Method, Precision, "Memory (7B)", "Speed vs FP16", "Best For"
# covering GPTQ, AWQ, GGUF (Q4_K_M), QLoRA, and the FP16 baseline.
quant_table = ____
print(quant_table)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert quant_table.height == 5, "Task 1: quantisation table should have 5 rows"
print("✓ Checkpoint 1 passed — quantisation landscape surveyed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: INT8 symmetric quantiser from scratch
# ════════════════════════════════════════════════════════════════════════


def quantise_int8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Symmetric INT8 quantisation. Returns (int tensor, scale).

    scale = max(|x|) / 127
    q     = round(x / scale).clamp(-127, 127).to(int8)
    """
    # TODO: compute max_abs and scale (guard scale=1.0 when max_abs==0)
    max_abs = ____
    scale = ____
    # TODO: quantise x using round + clamp + cast to int8
    q = ____
    return q, scale


def dequantise_int8(q: torch.Tensor, scale: float) -> torch.Tensor:
    """Reverse the INT8 mapping back to FP32."""
    # TODO: return q.to(float32) * scale
    ____


print("=" * 70)
print("TASK 2: INT8 symmetric quantiser on a 512x512 weight tensor")
print("=" * 70)

W_fp32 = torch.randn(512, 512) * 0.1
q, scale = quantise_int8(W_fp32)
W_roundtrip = dequantise_int8(q, scale)

print(f"Original dtype:   {W_fp32.dtype}  ({W_fp32.element_size() * 8} bits/elem)")
print(f"Quantised dtype:  {q.dtype}       ({q.element_size() * 8} bits/elem)")
print(f"Scale:            {scale:.6f}")
print(f"Memory reduction: {W_fp32.element_size() / q.element_size()}x")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert q.dtype == torch.int8, "Task 2: quantised tensor must be int8"
assert q.shape == W_fp32.shape, "Quantisation should preserve shape"
print("✓ Checkpoint 2 passed — INT8 quantiser built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: measure round-trip error across the weight distribution
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Measure quantisation round-trip error")
print("=" * 70)

# TODO: abs_err = |W_fp32 - W_roundtrip|; compute mae (mean), max_err (max),
# rel_err (mean abs err / mean |W_fp32|)
abs_err = ____
mae = ____
max_err = ____
rel_err = ____

print(f"  Mean absolute error (MAE): {mae:.6f}")
print(f"  Max absolute error:        {max_err:.6f}")
print(f"  Relative error:            {rel_err:.3%}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert mae < scale, "Task 3: MAE should be below the quantisation step"
assert rel_err < 0.05, "INT8 relative error should be under 5%"
print("✓ Checkpoint 3 passed — round-trip error is bounded by 1 LSB\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: memory footprint vs precision
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Visualise memory footprint across precisions (7B model)")
print("=" * 70)

precisions = ["FP32", "FP16 / BF16", "INT8", "INT4 / NF4", "INT2 (Q2_K)"]
bytes_per_param = [4, 2, 1, 0.5, 0.25]
# TODO: memory_gb = [7 * b for b in bytes_per_param]
memory_gb = ____

# TODO: Vertical bar plot with annotations of each bar's GB value.
# Save to OUTPUT_DIR / "ex2_quantisation_memory.png"
____
fname = OUTPUT_DIR / "ex2_quantisation_memory.png"
print(f"  Saved: {fname}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert fname.exists(), "Task 4: memory plot should exist"
print("✓ Checkpoint 4 passed — memory footprint visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore SME on-device assistant (GGUF on CPU)
# ════════════════════════════════════════════════════════════════════════
# A Singapore F&B chain runs 42 outlets with 8 GB ARM tablets (no GPU).
# PDPA requires data stays on-device. FP16 7B (14 GB) impossible; INT8
# 7B (7 GB) leaves no headroom; GGUF Q4_K_M (~4.5 GB) fits with room
# for the POS app. Multilingual (EN/MS/ID) response <1.5s SLA.
# Risk: Q4_K_M hurts Bahasa Indonesia quality — mitigate by training
# a small multilingual LoRA before quantising.

print("Singapore F&B on-device assistant decision:")
outlets = 42
quarterly_onboarding_saving = 5_300
# TODO: annual_onboarding_saving = quarterly_onboarding_saving * 4
annual_onboarding_saving = ____
annual_upsell_uplift = 4_030_000
annual_cloud_avoided = 14_100
# TODO: total_annual_benefit = sum of the three annual values above
total_annual_benefit = ____
print(f"  Outlets:                     {outlets}")
print(f"  Annual onboarding saving:    S${annual_onboarding_saving:,}")
print(f"  Annual upsell uplift:        S${annual_upsell_uplift:,}")
print(f"  Annual cloud cost avoided:   S${annual_cloud_avoided:,}")
print(f"  Total annual benefit:        S${total_annual_benefit:,}")
print(f"  Recommended: GGUF Q4_K_M deployed via llama.cpp on tablet CPU")

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert total_annual_benefit > 0, "Task 5: SME scenario should have positive ROI"
print("✓ Checkpoint 5 passed — SME assistant ROI analysed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Surveyed GPTQ, AWQ, GGUF, and QLoRA in one comparison table
  [x] Built an INT8 symmetric quantiser from scratch (scale only)
  [x] Measured round-trip error on a 512x512 weight tensor
  [x] Visualised memory footprint across FP32 -> INT2
  [x] Applied Q4_K_M GGUF to a Singapore F&B SME scenario
      (~S$4M/year in upsell + cost savings on 42 outlets)

  KEY INSIGHT: quantisation is the single biggest deployment lever.
  FP16 -> INT4 cuts memory 4x at ~2-5% quality drop on most tasks.

  Next: 06_sft_alignment_pipeline.py runs the real kailash-align
  SFT pipeline + AdapterRegistry on the IMDB SFT dataset.
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
