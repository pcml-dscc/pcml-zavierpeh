# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 2.2: Adapter Layers From Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a bottleneck adapter (LayerNorm -> down -> GELU -> up -> residual)
#   - Understand the zero-init trick that makes the adapter start as identity
#   - Compare LoRA vs adapters across parameter count, complexity, flexibility
#   - Visualise the LoRA-vs-adapter trade-off curve
#   - Apply the choice to a Singapore multi-tenant SaaS scenario
#
# PREREQUISITES: Exercise 2.1 (LoRA from scratch)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. THEORY: why bottleneck adapters exist (Houlsby et al., 2019)
#   2. BUILD: AdapterLayer + AdapterTransformerBlock
#   3. TRAIN: verify identity-at-init and param count
#   4. VISUALISE: params-vs-capacity curve (LoRA vs adapter)
#   5. APPLY: Singapore multi-tenant SaaS — 12 clients, one base model
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn

from shared.mlfp06.ex_2 import (
    ADAPTER_BOTTLENECK,
    D_MODEL,
    OUTPUT_DIR,
    count_adapter_params,
    count_lora_params,
    device,
    full_finetune_params,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Bottleneck Adapters
# ════════════════════════════════════════════════════════════════════════
# Houlsby et al. (2019) proposed inserting a small trainable module
# BETWEEN transformer layers instead of modifying existing weights.
#
#   Input (d) -> LayerNorm -> Down(d -> b) -> GELU -> Up(b -> d) -> + residual
#
# Key properties:
#   - b << d (bottleneck dimension; typical b = 64)
#   - Original transformer weights remain FROZEN
#   - Only the adapter weights (down + up + norms) are trained
#   - Residual connection lets the adapter start as a no-op
#
# Zero-init trick: initialise up_proj weights and bias to zero so that
# at t=0 the adapter output is the residual only -> it behaves as an
# identity function and doesn't perturb the pretrained model until
# training has begun.
#
# Why a NONLINEAR bottleneck (vs LoRA's linear factorisation)?  The
# GELU activation in the middle lets the adapter model nonlinear
# transformations per parameter.  This makes adapters more expressive
# per unit of compute than a pure linear LoRA update, at the cost of
# more parameters overall.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — BUILD: AdapterLayer and AdapterTransformerBlock
# ════════════════════════════════════════════════════════════════════════


class AdapterLayer(nn.Module):
    """Bottleneck adapter module — from-scratch implementation.

    Architecture: x -> LayerNorm -> Down(d->b) -> GELU -> Up(b->d) -> + x
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # Zero-init up-projection so adapter starts as identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.layer_norm(x)
        h = self.down_proj(h)
        h = self.activation(h)
        h = self.up_proj(h)
        h = self.dropout(h)
        return h + residual


class AdapterTransformerBlock(nn.Module):
    """Wraps an existing transformer block with a trainable adapter.

    The original block's weights are frozen; only the adapter trains.
    """

    def __init__(
        self,
        original_block: nn.Module,
        d_model: int,
        bottleneck_dim: int = 64,
    ):
        super().__init__()
        self.block = original_block
        for param in self.block.parameters():
            param.requires_grad = False
        self.adapter = AdapterLayer(d_model, bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        return self.adapter(h)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — TRAIN: verify adapter structure + identity-at-init
# ════════════════════════════════════════════════════════════════════════
# The actual adapter training loop would mirror LoRA's (handled by
# kailash-align in 05_sft_alignment_pipeline.py when target modules are
# configured for adapter-style injection).  Here we verify the module
# contract and count parameters.

print("\n" + "=" * 70)
print("TASK 2: Verify adapter structure on d=512 synthetic input")
print("=" * 70)

adapter = AdapterLayer(d_model=D_MODEL, bottleneck_dim=ADAPTER_BOTTLENECK).to(device)
adapter_params = sum(p.numel() for p in adapter.parameters())

x_test = torch.randn(2, 10, D_MODEL, device=device)
with torch.no_grad():
    y_adapter = adapter(x_test)
identity_gap = (x_test - y_adapter).abs().max().item()

print(f"Adapter layer: d={D_MODEL}, bottleneck={ADAPTER_BOTTLENECK}")
print(f"  Adapter params:       {adapter_params:,}")
print(f"  Output shape:         {tuple(y_adapter.shape)}")
print(f"  Identity gap at init: {identity_gap:.2e} (not exactly 0 due to LayerNorm)")

expected = count_adapter_params(D_MODEL, ADAPTER_BOTTLENECK, num_layers=1)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert isinstance(adapter, AdapterLayer), "Task 2: AdapterLayer should be created"
assert y_adapter.shape == x_test.shape, "Adapter output shape should match input"
assert (
    adapter_params == expected
), f"Adapter param count mismatch: got {adapter_params}, expected {expected}"
print("✓ Checkpoint 1 passed — adapter structure verified\n")

# INTERPRETATION: Adapter params = 2*d*b + 2*b + 2*d (down, up, norms).
# For d=512, b=64: ~66K.  More than LoRA r=8 (~8K) but the GELU in the
# middle lets it model nonlinear transformations, which LoRA cannot.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — COMPARE: LoRA vs Adapter across 4 dimensions
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: LoRA vs Adapter — 4-dimensional comparison")
print("=" * 70)

comparison = pl.DataFrame(
    {
        "Dimension": [
            "Parameter Update Mechanism",
            "Parameter Efficiency",
            "Implementation Complexity",
            "Flexibility & Modularity",
        ],
        "LoRA": [
            "Low-rank A,B added in parallel to attention projections",
            "~2*d*r params (linear only)",
            "Simple: 2 matrices per target, no architecture change",
            "Merge A,B into W at inference -> zero overhead",
        ],
        "Adapter": [
            "Bottleneck FC -> GELU -> FC inserted between layers",
            "~2*d*b + 2*d params (nonlinear)",
            "Moderate: new layers, residual path, zero-init strategy",
            "Stack multiple adapters per base model; cannot merge into W",
        ],
    }
)
print(comparison)

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert comparison.height == 4, "Task 3: comparison should cover 4 dimensions"
print("✓ Checkpoint 2 passed — LoRA vs Adapter comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: params-vs-capacity trade-off curve
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Visualise LoRA vs Adapter trade-off")
print("=" * 70)

# Per-target-module params for a single d=512 linear
lora_ranks = [4, 8, 16, 32, 64]
adapter_bs = [16, 32, 64, 128, 256]

lora_counts = [count_lora_params(D_MODEL, r) for r in lora_ranks]
adapter_counts = [count_adapter_params(D_MODEL, b) for b in adapter_bs]

full_per_module = full_finetune_params(D_MODEL)

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax.plot(
    lora_ranks, lora_counts, "o-", color="steelblue", linewidth=2, label="LoRA (rank)"
)
ax.plot(
    adapter_bs,
    adapter_counts,
    "s-",
    color="darkorange",
    linewidth=2,
    label="Adapter (bottleneck)",
)
ax.axhline(
    full_per_module,
    color="crimson",
    linestyle="--",
    label=f"Full FT ({full_per_module:,})",
)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Rank (LoRA) / Bottleneck (Adapter)")
ax.set_ylabel("Trainable parameters per target linear")
ax.set_title("LoRA vs Adapter — parameter cost (d=512)", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
fname = OUTPUT_DIR / "ex2_lora_vs_adapter.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fname}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert all(c < full_per_module for c in lora_counts), "LoRA should be < full"
print("✓ Checkpoint 3 passed — trade-off curve saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore multi-tenant SaaS (12 clients, one base)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore HR-tech SaaS serves 12 enterprise clients.
# Each client wants the shared LLM to speak "in their voice": their
# own job-ad tone, their own policy-explanation phrasing, their own
# preferred Singlish/formal balance.  Legally, one client's fine-tuned
# adaptation MUST NOT leak into another client's inference path
# (tenant isolation — see rules/tenant-isolation.md).
#
# CHOICE: LoRA or adapters?
#
#   LoRA pros:
#     - Tiny adapters (~65K params each at r=16 for a 7B base)
#     - Can be MERGED into the base for each tenant's dedicated
#       inference pod, zero inference overhead
#     - 12 LoRA checkpoints total ~10 MB on disk
#
#   Adapter pros:
#     - STACKABLE at inference time: the same shared base model can
#       load several adapters at once (rare, but useful for A/B testing
#       two of a tenant's adaptations side-by-side)
#     - Nonlinear capacity per parameter helps with very different
#       house styles (e.g. a tenant in the construction industry vs a
#       tenant in private banking)
#
# DECISION: LoRA is the right default.  For a 12-tenant SaaS running
# on a single shared GPU pool, LoRA merges into per-tenant inference
# pods with zero latency penalty, and the 10 MB of adapters fits
# trivially in a per-tenant secrets store.  Adapters would add
# ~3 ms/request of overhead per adapter block at inference time — at
# 2M requests/day that is ~100 minutes of extra compute.
#
# BUSINESS IMPACT: previously the SaaS fine-tuned a full 7B model per
# tenant (12 x S$600 = S$7,200/month in GPU cost, plus 12 x 14 GB of
# VRAM).  LoRA drops that to one shared base (14 GB) plus 12 LoRAs at
# ~S$25 each per re-train cycle = S$300/month — a S$6,900/month saving
# or S$82,800/year.

print("Singapore multi-tenant SaaS decision:")
tenants = 12
ft_cost_per_tenant = 600
lora_cost_per_tenant = 25
monthly_saving = tenants * (ft_cost_per_tenant - lora_cost_per_tenant)
print(f"  Tenants:                    {tenants}")
print(f"  Full FT cost / tenant/mo:   S${ft_cost_per_tenant}")
print(f"  LoRA cost / tenant/mo:      S${lora_cost_per_tenant}")
print(f"  Monthly saving:             S${monthly_saving:,}")
print(f"  Annual saving:              S${monthly_saving * 12:,}")
print(f"  Recommended technique:      LoRA r=16 per tenant")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert monthly_saving > 0, "Task 5: SaaS should see positive savings"
print("✓ Checkpoint 4 passed — SaaS cost/benefit analysed\n")


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
  [x] Built AdapterLayer and AdapterTransformerBlock from scratch
  [x] Zero-initialised up-projection for identity-at-init behaviour
  [x] Compared LoRA vs adapters across 4 dimensions
  [x] Visualised the params-vs-capacity trade-off curve
  [x] Applied the choice to a Singapore 12-tenant SaaS
      (S$82,800/year saving by switching to LoRA r=16)

  KEY INSIGHT: LoRA dominates single-task adaptation by merging into
  the base at inference.  Adapters shine when you need to STACK
  multiple adaptations on a shared base at runtime (rare in production).

  Next: 03_finetuning_landscape.py surveys all 10 techniques across
  the full fine-tuning landscape.
"""
)
