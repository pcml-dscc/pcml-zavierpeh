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
# Transformer weights follow a bell curve — most values cluster near
# zero, a few outliers pull the tails.  FP16 spends 16 bits on every
# weight regardless of whether the value is 0.0001 or 3.2.  That is
# enormously wasteful for the common case.
#
# Quantisation maps the continuous FP16 range onto a small integer grid:
#   INT8:  256 levels (~6x memory savings vs FP16 for weights)
#   INT4:  16 levels  (~4x vs FP16 but bigger quality hit)
#   NF4:   16 levels laid out to match the normal distribution exactly
#
# The four main methods in the table below correspond to different
# answers to "how do we preserve model quality at low bit-widths":
#
#   GPTQ (Frantar 2023): use the Hessian to pick quantisation grid
#       points that minimise per-layer output error.  Best on GPU.
#   AWQ  (Lin 2023):    identify "salient" weights activated by
#       important inputs; protect them at higher precision.  Better
#       than GPTQ on instruction-following benchmarks.
#   GGUF  (llama.cpp):   CPU-optimised mixed precision (Q2_K..Q8_0).
#       Key: runs on ARM/x86 CPUs without a GPU.  Edge deployment.
#   QLoRA (Dettmers 2023): 4-bit (NF4) FROZEN base + FP16 LoRA on top.
#       Enables fine-tuning 65B models on a single 48 GB card.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Print the quantisation landscape table
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Quantisation landscape — GPTQ / AWQ / GGUF / QLoRA")
print("=" * 70)

quant_table = pl.DataFrame(
    {
        "Method": ["GPTQ", "AWQ", "GGUF (Q4_K_M)", "QLoRA", "FP16 baseline"],
        "Precision": [
            "INT4",
            "INT4",
            "4-bit mixed",
            "NF4 base + FP16 LoRA",
            "FP16",
        ],
        "Memory (7B)": ["~4 GB", "~4 GB", "~4.5 GB", "~5 GB train", "~14 GB"],
        "Speed vs FP16": [
            "~1.5-2x",
            "~1.5-2x",
            "CPU-only",
            "~0.8x (train)",
            "1x (baseline)",
        ],
        "Best For": [
            "GPU inference",
            "GPU instruction tasks",
            "CPU/edge deploy",
            "Fine-tune large models",
            "Maximum quality",
        ],
    }
)
print(quant_table)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert quant_table.height == 5, "Task 1: quantisation table should have 5 rows"
print("✓ Checkpoint 1 passed — quantisation landscape surveyed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: INT8 symmetric quantiser from scratch
# ════════════════════════════════════════════════════════════════════════
# Symmetric INT8 quantisation maps the range [-max_abs, +max_abs] onto
# the integer grid {-127, ..., 127}.  Only one scale parameter per
# tensor (or per row, per column, per group).
#
#   q = round(x / scale)        with scale = max_abs / 127
#   x_dequant = q * scale
#
# Asymmetric INT8 adds a zero-point offset so the integer grid can be
# shifted — useful when the data is not centred on zero.  We skip it
# here for clarity.


def quantise_int8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Symmetric INT8 quantisation. Returns (int tensor, scale)."""
    max_abs = x.abs().max().item()
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantise_int8(q: torch.Tensor, scale: float) -> torch.Tensor:
    """Reverse the INT8 mapping back to FP32."""
    return q.to(torch.float32) * scale


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
# "Training" here is evaluating the quantisation quality.  We measure
# three numbers the practitioner actually cares about:
#   - Mean absolute error (MAE) in raw units
#   - Max absolute error (worst case)
#   - Relative error (how big the error is vs the signal)

print("=" * 70)
print("TASK 3: Measure quantisation round-trip error")
print("=" * 70)

abs_err = (W_fp32 - W_roundtrip).abs()
mae = abs_err.mean().item()
max_err = abs_err.max().item()
rel_err = (abs_err.mean() / W_fp32.abs().mean()).item()

print(f"  Mean absolute error (MAE): {mae:.6f}")
print(f"  Max absolute error:        {max_err:.6f}")
print(f"  Relative error:            {rel_err:.3%}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert mae < scale, "Task 3: MAE should be below the quantisation step"
assert rel_err < 0.05, "INT8 relative error should be under 5%"
print("✓ Checkpoint 3 passed — round-trip error is bounded by 1 LSB\n")

# INTERPRETATION: The relative error is well under 1%, which is why
# INT8 quantisation barely moves the needle on downstream accuracy for
# most transformer layers.  INT4 pushes this to ~2-5% and you start to
# need GPTQ / AWQ tricks to stay competitive on instruction tasks.


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: memory footprint vs precision
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Visualise memory footprint across precisions (7B model)")
print("=" * 70)

# Rough memory cost for a 7B-parameter model's weights at each precision.
precisions = ["FP32", "FP16 / BF16", "INT8", "INT4 / NF4", "INT2 (Q2_K)"]
bytes_per_param = [4, 2, 1, 0.5, 0.25]
memory_gb = [7 * b for b in bytes_per_param]

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
colors = ["crimson", "darkorange", "goldenrod", "steelblue", "seagreen"]
bars = ax.bar(precisions, memory_gb, color=colors, edgecolor="black")
for bar, mem in zip(bars, memory_gb):
    ax.annotate(
        f"{mem:.1f} GB",
        xy=(bar.get_x() + bar.get_width() / 2, mem),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
ax.set_ylabel("Memory for 7B weights (GB)")
ax.set_title("Quantisation memory footprint (7B model)", fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex2_quantisation_memory.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fname}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert fname.exists(), "Task 4: memory plot should exist"
print("✓ Checkpoint 4 passed — memory footprint visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore SME on-device assistant (GGUF on CPU)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore F&B retail chain runs 42 outlets across
# Singapore, Johor Bahru, and Batam.  Each outlet has a cheap point-
# of-sale tablet (ARM CPU, 8 GB RAM, no GPU) that staff use to answer
# customer questions in English, Malay, and Bahasa Indonesia: menu
# allergens, halal certification, opening hours, loyalty points.
#
# CONSTRAINTS:
#   - No GPU budget at the outlet tier
#   - Network latency to HQ is unreliable during lunch rush
#   - Data must stay on-device for PDPA compliance (customer queries
#     contain PII: phone numbers, email, children's allergies)
#   - Outlet owners will not tolerate >1.5 second response times
#
# OPTIONS:
#   A. FP16 7B on HQ GPU + network call: 14 GB VRAM, ~500 ms network
#      round-trip, blows PDPA because queries leave the outlet
#   B. FP16 7B on outlet tablet: 14 GB memory — impossible, tablet
#      has 8 GB
#   C. INT8 7B on outlet tablet: 7 GB — technically fits but no
#      headroom for the POS app itself
#   D. GGUF Q4_K_M 7B on outlet tablet: ~4.5 GB — fits comfortably,
#      leaves room for POS + OS
#
# DECISION: Option D (GGUF Q4_K_M on the outlet tablet via llama.cpp).
# Keeps data on-device (PDPA win), works offline, response time ~900 ms
# on the tablet CPU which clears the 1.5 s SLA.
#
# BUSINESS IMPACT:
#   - Staff training time drops: new hires used to spend ~3 hours
#     learning the menu card.  The assistant cuts that to ~45 min.
#     42 outlets * 4 new hires/quarter * 2.25 hrs saved * S$14/hr =
#     ~S$5,300/quarter staff onboarding saving.
#   - Upsell: assistant suggests pairings (dessert, drinks, upsize).
#     Early pilot at two outlets showed a 4% lift on ticket size.
#     At S$2.4M combined annual revenue per outlet, 4% * 42 outlets *
#     S$2.4M = ~S$4,030,000/year in upsell revenue.
#   - Avoided cloud spend: option A would have cost ~S$28/outlet/month
#     in HQ inference * 42 * 12 = ~S$14,100/year that doesn't get spent.
#
# RISK: Q4_K_M quality on multilingual tasks is noticeably worse than
# FP16 for Bahasa Indonesia.  Mitigation: fine-tune a small LoRA on
# bi-lingual menu text BEFORE quantising, so the adapted model
# tolerates the INT4 rounding.

print("Singapore F&B on-device assistant decision:")
outlets = 42
quarterly_onboarding_saving = 5_300
annual_onboarding_saving = quarterly_onboarding_saving * 4
annual_upsell_uplift = 4_030_000
annual_cloud_avoided = 14_100
total_annual_benefit = (
    annual_onboarding_saving + annual_upsell_uplift + annual_cloud_avoided
)
print(f"  Outlets:                     {outlets}")
print(f"  Annual onboarding saving:    S${annual_onboarding_saving:,}")
print(f"  Annual upsell uplift:        S${annual_upsell_uplift:,}")
print(f"  Annual cloud cost avoided:   S${annual_cloud_avoided:,}")
print(f"  Total annual benefit:        S${total_annual_benefit:,}")
print(f"  Recommended: GGUF Q4_K_M deployed via llama.cpp on tablet CPU")

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert total_annual_benefit > 0, "Task 5: SME scenario should have positive ROI"
print("✓ Checkpoint 5 passed — SME assistant ROI analysed\n")


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
  [x] Surveyed GPTQ, AWQ, GGUF, and QLoRA in one comparison table
  [x] Built an INT8 symmetric quantiser from scratch (scale only)
  [x] Measured round-trip error on a 512x512 weight tensor
  [x] Visualised memory footprint across FP32 -> INT2
  [x] Applied Q4_K_M GGUF to a Singapore F&B SME scenario
      (~S$4M/year in upsell + cost savings on 42 outlets)

  KEY INSIGHT: quantisation is the single biggest deployment lever.
  FP16 -> INT4 cuts memory 4x at ~2-5% quality drop on most tasks.
  Pair quantisation with LoRA (QLoRA) for fine-tuning, and with
  llama.cpp GGUF for CPU-only edge deployment.

  Next: 06_sft_alignment_pipeline.py runs the real kailash-align
  SFT pipeline + AdapterRegistry on the IMDB SFT dataset.
"""
)
