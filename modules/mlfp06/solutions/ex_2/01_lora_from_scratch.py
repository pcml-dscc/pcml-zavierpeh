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

from shared.mlfp06.ex_2 import (
    ADAPTER_BOTTLENECK,
    D_MODEL,
    LORA_ALPHA,
    LORA_RANK,
    OUTPUT_DIR,
    count_adapter_params,
    count_lora_params,
    device,
    full_finetune_params,
    load_imdb_sft,
)

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load SFT dataset
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Load SFT Dataset — IMDB instruction/response pairs")
print("=" * 70)

sft_data, train_data, eval_data = load_imdb_sft()
print(f"Sample instruction:\n  {sft_data['instruction'][0][:200]}...")
print(f"Sample response:\n  {sft_data['response'][0]}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert sft_data.height > 0, "Task 1: SFT dataset should not be empty"
assert "instruction" in sft_data.columns, "Dataset needs 'instruction' column"
assert "response" in sft_data.columns, "Dataset needs 'response' column"
print(f"✓ Checkpoint 1 passed — {sft_data.height} SFT pairs loaded\n")

# INTERPRETATION: SFT data = (instruction, response) pairs.  The model learns
# to follow instructions by maximising P(response | instruction).  Quality
# matters more than quantity: 500 high-quality domain-specific pairs can
# outperform 10,000 generic pairs.


# ════════════════════════════════════════════════════════════════════════
# THEORY — Low-Rank Adaptation (LoRA)
# ════════════════════════════════════════════════════════════════════════
# Full fine-tuning: W_new = W + delta_W, where delta_W is d x d.
#   That's d^2 trainable parameters per weight matrix. For d=4096 (a 7B
#   model's hidden size), that's 16.7M params PER MATRIX, and a 7B model
#   has hundreds of such matrices.
#
# LoRA insight (Hu et al., 2021): the UPDATE delta_W is low-rank.
#   Factorise it as delta_W = A @ B where A is d x r and B is r x d.
#   With r << d (typically r=8), trainable params drop from d^2 to 2*d*r.
#   For d=4096, r=8:  16.7M -> 65.5K params. A 255x reduction.
#
# Why it works: fine-tuning rarely needs the full expressiveness of
# d^2 directions. A handful of task-specific "directions" in weight
# space is enough.  LoRA IS low-rank factorisation (M4.3 SVD) applied
# to the weight UPDATE, not the weight itself.
#
# Initialisation trick: A ~ Kaiming, B = 0.  At init, A @ B = 0, so
# W_new = W (no change to the pretrained behaviour).  Training then
# gradually learns the delta.
#
# Analogy: think of the pretrained model as a piano.  Full fine-tuning
# rebuilds the whole instrument. LoRA is a capo — a small, cheap, and
# removable device that transposes the entire instrument into a new
# key without touching the strings.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: LoRALayer and LoRALinear
# ════════════════════════════════════════════════════════════════════════


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer — from-scratch implementation.

    Computes: output = (x @ A @ B) * (alpha / r)

    Args:
        in_features:  Input dimension (d_model).
        out_features: Output dimension (d_model for self-attention projections).
        rank:         LoRA rank r.  Lower = fewer params, less expressive.
        alpha:        Scaling factor.  Typical: alpha = 2 * rank.
        dropout:      Dropout on the LoRA path for regularisation.
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
        self.scaling = alpha / rank

        # A: d_in x r  (Kaiming init for stable gradients)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        # B: r x d_out (zero init so LoRA starts as identity)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        """Kaiming init for A; zero init for B.

        Ensures A @ B = 0 at init, so W_new = W (no change at start).
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dropped = self.dropout(x)
        lora_out = dropped @ self.lora_A @ self.lora_B
        return lora_out * self.scaling


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
        # Freeze the pretrained weights
        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            in_features=pretrained_linear.in_features,
            out_features=pretrained_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: verify LoRA identity-at-init + parameter count
# ════════════════════════════════════════════════════════════════════════
# LoRA's training loop runs inside kailash-align's AlignmentPipeline
# (see 05_sft_alignment_pipeline.py).  Here we verify that the
# mathematical structure behaves as claimed: B=0 means zero delta at
# init, and the trainable count matches 2*d*r exactly.

print("=" * 70)
print("TASK 3: Verify LoRA structure on a synthetic d=512 linear")
print("=" * 70)

pretrained = nn.Linear(D_MODEL, D_MODEL).to(device)
lora_linear = LoRALinear(pretrained, rank=LORA_RANK, alpha=LORA_ALPHA).to(device)

frozen_params = sum(p.numel() for p in lora_linear.linear.parameters())
trainable_params = sum(
    p.numel() for p in lora_linear.lora.parameters() if p.requires_grad
)

# Identity-at-init check: W_new(x) should equal W(x) before any training
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

# INTERPRETATION: 2*d*r trainable params vs d^2 frozen.  For d=512, r=8:
# 8,192 vs 262,144 = 3.1%.  The B=0 initialisation means LoRA starts as
# identity, then gradually learns the domain-specific delta during training.


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: parameter reduction curve across ranks
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Visualise parameter reduction across ranks")
print("=" * 70)

# TinyLlama-scale analysis
hidden_dim = 2048
num_layers = 22
num_target_modules = 2  # q_proj, v_proj
full_total = full_finetune_params(hidden_dim, num_target_modules, num_layers)

ranks = [2, 4, 8, 16, 32, 64, 128]
lora_totals = [
    count_lora_params(hidden_dim, r, num_target_modules, num_layers) for r in ranks
]
pct_of_full = [p / full_total * 100 for p in lora_totals]

print(f"Model: TinyLlama-scale (d={hidden_dim}, layers={num_layers})")
print(f"Full fine-tuning params: {full_total:,}")
print(f"\n  rank    params         % of full    reduction")
for r, p, pct in zip(ranks, lora_totals, pct_of_full):
    print(f"  r={r:<5} {p:>12,}   {pct:>7.2f}%    {full_total / p:>6.1f}x")

# Plot: parameter count vs rank (log-log)
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax.plot(ranks, lora_totals, "o-", color="steelblue", linewidth=2, label="LoRA")
ax.axhline(
    full_total, color="crimson", linestyle="--", label=f"Full FT = {full_total:,}"
)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("LoRA rank (r)")
ax.set_ylabel("Trainable parameters")
ax.set_title("LoRA parameter reduction — TinyLlama-scale model", fontweight="bold")
for r, p, pct in zip(ranks, lora_totals, pct_of_full):
    ax.annotate(
        f"{pct:.1f}%", (r, p), textcoords="offset points", xytext=(5, 5), fontsize=8
    )
ax.legend()
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
fname = OUTPUT_DIR / "ex2_lora_rank_sweep.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert all(p < full_total for p in lora_totals), "Task 4: LoRA must use fewer params"
assert pct_of_full[0] < 1.0, "r=2 should be <1% of full"
print("✓ Checkpoint 3 passed — rank sweep visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore law-firm assistant — rank selection
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore law firm wants to adapt a 7B open-source base
# model to draft first-pass contracts in Singlish-aware business English,
# cite local statutes correctly, and handle the firm's preferred clause
# boilerplate.  They have 800 historical contracts for training and a
# single 24 GB GPU.
#
# DECISION: Full fine-tuning of a 7B model needs ~56 GB for gradients
# and optimiser state -> impossible on one 24 GB card.  LoRA is the only
# path.  The question is: what rank?
#
# RANK SELECTION GUIDE:
#   r=4:   <1% params — classification, light stylistic tweaks
#   r=8:   ~1% params — the pragmatic default
#   r=16:  ~2% params — complex domains (legal, medical, financial)
#   r=32+: diminishing returns; consider full FT if budget allows
#
# BUSINESS IMPACT: junior associates at the firm currently spend ~6
# hours/week drafting first-pass contracts at a fully-loaded cost of
# ~S$120/hour (S$720/week per associate, 12 associates = S$8,640/week).
# A LoRA r=16 assistant trained overnight on the 800 contracts reduces
# that to ~1.5 hours/week of review-only work, saving ~S$6,480/week or
# roughly S$335k/year.  Cloud GPU training cost: ~S$80 per run.
#
# RISK: a rank that is too low (r=2) underfits the firm's house style;
# a rank that is too high (r=128) starts to memorise individual contracts
# and leak client data in generation.  r=8 or r=16 is the sweet spot.

# Cost/benefit sweep for the law firm
print("Singapore law-firm rank cost/benefit:")
weekly_hours_before = 6 * 12
weekly_hours_after = 1.5 * 12
hourly_cost_sgd = 120
weekly_saving_sgd = (weekly_hours_before - weekly_hours_after) * hourly_cost_sgd
annual_saving_sgd = weekly_saving_sgd * 50  # 50 working weeks
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
