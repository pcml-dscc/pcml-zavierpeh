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
from dotenv import load_dotenv

from shared.mlfp06.ex_2 import (
    ADAPTER_BOTTLENECK,
    D_MODEL,
    OUTPUT_DIR,
    count_adapter_params,
    count_lora_params,
    device,
    full_finetune_params,
)

load_dotenv()

# ════════════════════════════════════════════════════════════════════════
# THEORY — Bottleneck Adapters (Houlsby et al., 2019)
# ════════════════════════════════════════════════════════════════════════
# Insert a small trainable module between transformer layers:
#   x -> LayerNorm -> Down(d -> b) -> GELU -> Up(b -> d) -> + residual
# Zero-init the up projection so the adapter starts as identity.
# The GELU gives adapters nonlinear capacity that LoRA does not have.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — BUILD: AdapterLayer and AdapterTransformerBlock
# ════════════════════════════════════════════════════════════════════════


class AdapterLayer(nn.Module):
    """Bottleneck adapter module — from-scratch implementation."""

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        # TODO: LayerNorm over d_model, down_proj Linear(d_model -> bottleneck_dim),
        # GELU activation, up_proj Linear(bottleneck_dim -> d_model), Dropout(p=dropout)
        self.layer_norm = ____
        self.down_proj = ____
        self.activation = ____
        self.up_proj = ____
        self.dropout = ____

        # TODO: Zero-init up_proj.weight and up_proj.bias so adapter starts as identity
        ____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: residual = x; pass through norm -> down -> activation -> up -> dropout;
        # return adapter output + residual
        ____


class AdapterTransformerBlock(nn.Module):
    """Wraps an existing transformer block with a trainable adapter."""

    def __init__(
        self,
        original_block: nn.Module,
        d_model: int,
        bottleneck_dim: int = 64,
    ):
        super().__init__()
        self.block = original_block
        # TODO: Freeze original_block's parameters (requires_grad = False)
        ____
        # TODO: Attach an AdapterLayer(d_model, bottleneck_dim)
        self.adapter = ____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Run the frozen block then the adapter on its output
        ____


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — TRAIN: verify adapter structure + parameter count
# ════════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — COMPARE: LoRA vs Adapter across 4 dimensions
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: LoRA vs Adapter — 4-dimensional comparison")
print("=" * 70)

# TODO: Build a polars DataFrame `comparison` with columns
# "Dimension", "LoRA", "Adapter" covering:
#   - Parameter Update Mechanism
#   - Parameter Efficiency
#   - Implementation Complexity
#   - Flexibility & Modularity
comparison = ____
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

lora_ranks = [4, 8, 16, 32, 64]
adapter_bs = [16, 32, 64, 128, 256]

# TODO: lora_counts = [count_lora_params(D_MODEL, r) for r in lora_ranks]
# TODO: adapter_counts = [count_adapter_params(D_MODEL, b) for b in adapter_bs]
lora_counts = ____
adapter_counts = ____

full_per_module = full_finetune_params(D_MODEL)

# TODO: Log-log matplotlib plot with two series (LoRA, Adapter) and an axhline at full_per_module.
# Save to OUTPUT_DIR / "ex2_lora_vs_adapter.png".
____
fname = OUTPUT_DIR / "ex2_lora_vs_adapter.png"
print(f"  Saved: {fname}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert all(c < full_per_module for c in lora_counts), "LoRA should be < full"
print("✓ Checkpoint 3 passed — trade-off curve saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore multi-tenant SaaS (12 clients, one base)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore HR-tech SaaS serves 12 enterprise clients.
# Each client wants the shared LLM to speak "in their voice" while
# staying tenant-isolated (rules/tenant-isolation.md).
# LoRA pros: tiny (~65K params per tenant at r=16) and mergeable into
# the base for zero inference overhead.
# Adapter pros: stackable at runtime, nonlinear capacity per parameter.
# Decision: LoRA r=16 per tenant. 12 LoRAs total ~10 MB on disk,
# drops VRAM from 12 x 14 GB to 1 x 14 GB shared base.

print("Singapore multi-tenant SaaS decision:")
tenants = 12
ft_cost_per_tenant = 600
lora_cost_per_tenant = 25

# TODO: monthly_saving = tenants * (ft_cost_per_tenant - lora_cost_per_tenant)
monthly_saving = ____
print(f"  Tenants:                    {tenants}")
print(f"  Full FT cost / tenant/mo:   S${ft_cost_per_tenant}")
print(f"  LoRA cost / tenant/mo:      S${lora_cost_per_tenant}")
print(f"  Monthly saving:             S${monthly_saving:,}")
print(f"  Annual saving:              S${monthly_saving * 12:,}")
print(f"  Recommended technique:      LoRA r=16 per tenant")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert monthly_saving > 0, "Task 5: SaaS should see positive savings"
print("✓ Checkpoint 4 passed — SaaS cost/benefit analysed\n")


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
