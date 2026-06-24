# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 2.1: LoRA From Scratch (Low-Rank Adaptation)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Load an SFT (supervised fine-tuning) dataset of instruction-response pairs
#   - Implement LoRA as an nn.Module: W_new = W + (A @ B) * (alpha / r)
#   - Understand the low-rank intuition: only r directions are updated
#   - Visualise the parameter reduction curve across ranks r = 2..128
#   - Apply LoRA rank selection to a Singapore legal-document assistant
#
# PREREQUISITES: Exercise 1 (LLM fundamentals); M4.3 (SVD, low-rank factorisation)
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Load IMDB SFT dataset (instruction/response pairs)
#   2. THEORY: low-rank factorisation of the weight delta
#   3. BUILD: LoRALayer + LoRALinear (Kaiming A, zero B)
#   4. TRAIN: no-op — LoRA starts as identity, we verify parameter counts
#   5. VISUALISE: params vs rank sweep (log-scale plot saved to outputs/)
#   6. APPLY: Singapore law firm — rank selection cost/benefit
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dotenv import load_dotenv

from shared.mlfp06.ex_2 import (
    D_MODEL,
    LORA_ALPHA,
    LORA_RANK,
    OUTPUT_DIR,
    count_lora_params,
    device,
    full_finetune_params,
    load_imdb_sft,
)

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load SFT dataset
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Load SFT Dataset — IMDB instruction/response pairs")
print("=" * 70)

# TODO: Call load_imdb_sft() and unpack into (sft_data, train_data, eval_data)
sft_data, train_data, eval_data = ____

print(f"Sample instruction:\n  {sft_data['instruction'][0][:200]}...")
print(f"Sample response:\n  {sft_data['response'][0]}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert sft_data.height > 0, "Task 1: SFT dataset should not be empty"
assert "instruction" in sft_data.columns, "Dataset needs 'instruction' column"
assert "response" in sft_data.columns, "Dataset needs 'response' column"
print(f"✓ Checkpoint 1 passed — {sft_data.height} SFT pairs loaded\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Low-Rank Adaptation (LoRA)
# ════════════════════════════════════════════════════════════════════════
# Full fine-tuning: W_new = W + delta_W, where delta_W is d x d.
# LoRA insight: the UPDATE is low-rank. Factorise delta_W = A @ B with
# r << d, so trainable params drop from d^2 to 2*d*r.
# Init trick: A ~ Kaiming, B = 0 so the adapter starts as identity.
# Analogy: LoRA is a capo on a pretrained piano — a small removable
# device that transposes without touching the strings.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: LoRALayer and LoRALinear
# ════════════════════════════════════════════════════════════════════════


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer — from-scratch implementation.

    Computes: output = (x @ A @ B) * (alpha / r)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        # TODO: Set self.scaling to alpha / rank
        self.scaling = ____

        # TODO: Define lora_A as an nn.Parameter of shape (in_features, rank)
        # using torch.empty (we initialise it in reset_parameters)
        self.lora_A = ____
        # TODO: Define lora_B as an nn.Parameter of zeros with shape (rank, out_features)
        self.lora_B = ____

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        """Kaiming init for A; zero init for B (so A @ B = 0 at start)."""
        # TODO: Use nn.init.kaiming_uniform_ on self.lora_A with a=math.sqrt(5)
        ____
        # TODO: Zero-init self.lora_B via nn.init.zeros_
        ____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Apply dropout, multiply by A then B, scale by self.scaling
        ____


class LoRALinear(nn.Module):
    """A frozen linear layer augmented with a trainable LoRA path.

    Computes: y = W_frozen @ x + bias + LoRALayer(x)
    """

    def __init__(
        self,
        pretrained_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = pretrained_linear
        # TODO: Freeze all parameters of self.linear (param.requires_grad = False)
        ____

        # TODO: Attach a LoRALayer with matching in_features/out_features
        self.lora = ____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Return the frozen linear output plus the LoRA output
        ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: verify LoRA identity-at-init + parameter count
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Verify LoRA structure on a synthetic d=512 linear")
print("=" * 70)

pretrained = nn.Linear(D_MODEL, D_MODEL).to(device)
lora_linear = LoRALinear(pretrained, rank=LORA_RANK, alpha=LORA_ALPHA).to(device)

frozen_params = sum(p.numel() for p in lora_linear.linear.parameters())
trainable_params = sum(
    p.numel() for p in lora_linear.lora.parameters() if p.requires_grad
)

x_test = torch.randn(2, 10, D_MODEL, device=device)
with torch.no_grad():
    y_base = lora_linear.linear(x_test)
    y_lora = lora_linear(x_test)
identity_gap = (y_base - y_lora).abs().max().item()

print(f"LoRA layer created: d={D_MODEL}, r={LORA_RANK}")
print(f"  Frozen params:    {frozen_params:,}")
print(f"  Trainable params: {trainable_params:,}")
print(f"  Ratio:            {trainable_params / frozen_params:.4%} of frozen")
print(f"  Identity gap at init: {identity_gap:.2e} (should be ~0)")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert isinstance(lora_linear, LoRALinear), "Task 3: LoRALinear should be created"
assert (
    trainable_params == 2 * D_MODEL * LORA_RANK
), f"LoRA params should be 2*d*r = {2 * D_MODEL * LORA_RANK}"
assert identity_gap < 1e-5, "LoRA should start as identity (B=0)"
print("✓ Checkpoint 2 passed — LoRA structure and identity-init verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: parameter reduction curve across ranks
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Visualise parameter reduction across ranks")
print("=" * 70)

hidden_dim = 2048
num_layers = 22
num_target_modules = 2  # q_proj, v_proj

# TODO: Compute full_total via full_finetune_params(hidden_dim, num_target_modules, num_layers)
full_total = ____

ranks = [2, 4, 8, 16, 32, 64, 128]
# TODO: Build lora_totals as a list of count_lora_params(...) for each rank in ranks
lora_totals = ____
pct_of_full = [p / full_total * 100 for p in lora_totals]

print(f"Model: TinyLlama-scale (d={hidden_dim}, layers={num_layers})")
print(f"Full fine-tuning params: {full_total:,}")
print(f"\n  rank    params         % of full    reduction")
for r, p, pct in zip(ranks, lora_totals, pct_of_full):
    print(f"  r={r:<5} {p:>12,}   {pct:>7.2f}%    {full_total / p:>6.1f}x")

# TODO: Plot a log-log line of ranks vs lora_totals with a crimson axhline at full_total.
# Save to OUTPUT_DIR / "ex2_lora_rank_sweep.png" and close the figure.
____
fname = OUTPUT_DIR / "ex2_lora_rank_sweep.png"

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert all(p < full_total for p in lora_totals), "Task 4: LoRA must use fewer params"
assert pct_of_full[0] < 1.0, "r=2 should be <1% of full"
print("✓ Checkpoint 3 passed — rank sweep visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore law-firm assistant — rank selection
# ════════════════════════════════════════════════════════════════════════
# A Singapore law firm wants to adapt a 7B base model to draft
# first-pass contracts. Full fine-tuning needs ~56 GB for optimiser
# state (impossible on one 24 GB card), so LoRA is the only path.
# Rank guide: r=4 light stylistic / r=8 default / r=16 complex
# domains (legal, medical) / r=32+ diminishing returns.

print("Singapore law-firm rank cost/benefit:")
weekly_hours_before = 6 * 12
weekly_hours_after = 1.5 * 12
hourly_cost_sgd = 120

# TODO: Compute weekly_saving_sgd and annual_saving_sgd (50 working weeks)
weekly_saving_sgd = ____
annual_saving_sgd = ____
print(f"  Hours/week before:  {weekly_hours_before}")
print(f"  Hours/week after:   {weekly_hours_after}")
print(f"  Weekly saving:      S${weekly_saving_sgd:,.0f}")
print(f"  Annual saving:      S${annual_saving_sgd:,.0f}")
print(f"  Training cost:      ~S$80 (one-off GPU run)")
print(
    f"  Recommended rank:   r=16 "
    f"({pct_of_full[ranks.index(16)]:.2f}% of full params)"
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert annual_saving_sgd > 0, "Task 5: law firm should see positive savings"
print("✓ Checkpoint 4 passed — Singapore law firm cost/benefit analysed\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Loaded an SFT dataset of instruction/response pairs (IMDB)
  [x] Built LoRALayer and LoRALinear from scratch (Kaiming A, zero B)
  [x] Verified identity-at-init: LoRA starts as W_new = W
  [x] Visualised the parameter reduction curve across ranks 2..128
  [x] Applied LoRA rank selection to a Singapore law-firm scenario
      (S$335k/year saving at r=16, ~S$80 training cost)

  KEY INSIGHT: LoRA is SVD applied to the UPDATE, not the weight.
  A handful of "directions" in weight space is usually enough to
  adapt a model to a new task, so 2*d*r parameters replace d^2.

  Next: 02_adapter_from_scratch.py builds bottleneck adapters and
  compares them to LoRA across four dimensions.
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
