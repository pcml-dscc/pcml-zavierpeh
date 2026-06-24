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
# A task vector is the *difference* between a fine-tuned model and its
# base:  tau = W_finetuned - W_base.  It captures everything the
# fine-tuning added, and nothing else.
#
# Task arithmetic (Ilharco et al., 2023):
#   - Add:      W_merged = W_base + alpha * tau_A + beta * tau_B
#   - Subtract: W_merged = W_base - alpha * tau_A    (remove a skill)
#   - Negate:   reverses the adaptation entirely
#
# But raw addition of task vectors causes *sign conflict*: two fine-tunes
# may update the same parameter in opposite directions, and averaging
# cancels both out into noise.  TIES and DARE solve this.
#
# TIES (Yadav et al., 2023): TRIM small deltas, ELECT the majority sign,
# then MERGE only the deltas that agree.  Reduces noise and conflict.
#
# DARE (Yu et al., 2023): randomly DROP a fraction of delta parameters
# then RESCALE the survivors by 1/(1-drop).  Like dropout for merging.
#
# SLERP (Spherical Linear Interpolation): walks the shortest path on the
# hypersphere between two weight tensors rather than the straight chord.
# Preserves weight norms, which matters because transformer weights are
# calibrated to a specific norm at training time.
#
# Crucially: merging is FREE — zero extra training compute.  This makes
# it the cheapest way to combine capabilities from multiple fine-tunes.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — BUILD: TIES merge on synthetic 128x128 deltas
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: TIES merge — trim / elect sign / average")
print("=" * 70)

delta_A = torch.randn(128, 128) * 0.1  # Task A fine-tuned delta
delta_B = torch.randn(128, 128) * 0.1  # Task B fine-tuned delta

# Step 1 — TRIM: zero out small-magnitude updates (treat as noise)
trim_threshold = 0.05
delta_A_trim = delta_A.clone()
delta_A_trim[delta_A_trim.abs() < trim_threshold] = 0
delta_B_trim = delta_B.clone()
delta_B_trim[delta_B_trim.abs() < trim_threshold] = 0

# Step 2 — ELECT SIGN: majority vote per parameter
sign_A = delta_A_trim.sign()
sign_B = delta_B_trim.sign()
elected_sign = (sign_A + sign_B).sign()

# Step 3 — MERGE: average the deltas that agree with the elected sign
mask_A = (sign_A == elected_sign).float()
mask_B = (sign_B == elected_sign).float()
merged_delta = (delta_A_trim * mask_A + delta_B_trim * mask_B) / (
    mask_A + mask_B + 1e-8
)

print(f"Original non-zero params (delta_A): {(delta_A != 0).sum().item():,}")
print(f"After TRIM:                         {(delta_A_trim != 0).sum().item():,}")
print(f"Sign agreement rate:                {(sign_A == sign_B).float().mean():.1%}")
print(f"Merged delta Frobenius norm:        {merged_delta.norm():.4f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert merged_delta.shape == delta_A.shape, "Task 1: TIES should preserve shape"
assert (delta_A_trim != 0).sum() < (delta_A != 0).sum(), "TRIM should zero some values"
print("✓ Checkpoint 1 passed — TIES merge complete\n")

# INTERPRETATION: TIES prevents sign cancellation during merging.
# Without it, two good task vectors that disagree on a parameter
# produce a weaker merged result than either input alone.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: SLERP merge on weight tensors
# ════════════════════════════════════════════════════════════════════════


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two tensors.

    t=0 -> v0; t=1 -> v1; t=0.5 -> midpoint on the hypersphere.
    Falls back to linear interpolation when the vectors are nearly
    parallel (theta < 1e-6) to avoid divide-by-zero.
    """
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()
    cos_theta = torch.dot(v0_flat, v1_flat) / (v0_flat.norm() * v1_flat.norm() + 1e-8)
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    if theta.abs() < 1e-6:
        return (1 - t) * v0 + t * v1
    sin_theta = torch.sin(theta)
    w0 = torch.sin((1 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
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

# INTERPRETATION: Linear merging shrinks toward the origin whenever the
# two inputs are not perfectly aligned.  SLERP walks the arc on the
# hypersphere and keeps the interpolated norm close to the inputs.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — VISUALISE: norm comparison across merge strategies
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Visualise merge-strategy norm preservation")
print("=" * 70)

# Sweep the interpolation parameter t from 0 -> 1 and record norms
ts = [i / 10 for i in range(11)]
slerp_norms = [slerp(t, W_task_A, W_task_B).norm().item() for t in ts]
linear_norms = [((1 - t) * W_task_A + t * W_task_B).norm().item() for t in ts]

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax.plot(ts, slerp_norms, "o-", color="steelblue", linewidth=2, label="SLERP")
ax.plot(ts, linear_norms, "s-", color="darkorange", linewidth=2, label="Linear")
ax.axhline(
    norms["W_task_A"], color="gray", linestyle=":", alpha=0.6, label="input norms"
)
ax.axhline(norms["W_task_B"], color="gray", linestyle=":", alpha=0.6)
ax.set_xlabel("Interpolation parameter t (0 = task A, 1 = task B)")
ax.set_ylabel("Frobenius norm of merged weights")
ax.set_title("SLERP vs Linear — weight norm preservation", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex2_slerp_vs_linear.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fname}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert fname.exists(), "Task 3: plot should exist"
print("✓ Checkpoint 3 passed — norm comparison visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — APPLY: Singapore fintech — merging three LoRAs into one
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore digital bank has trained three independent LoRA
# adapters on top of the same 7B base:
#   tau_kyc     — KYC document extraction (NRIC, addresses, phone formats)
#   tau_fraud   — Transaction-narrative fraud flagging
#   tau_explain — Plain-English explanations for rejected applications
#
# Each adapter was trained by a different team over several weeks.  The
# platform team wants to ship a SINGLE deployed model that handles all
# three tasks instead of three separate inference pipelines.
#
# OPTIONS:
#   A. Three deployments (status quo): 3 * 14 GB GPU VRAM per region.
#      High latency for multi-step journeys (KYC -> fraud -> explain).
#   B. Train a multi-task adapter: ~2 weeks of additional GPU time to
#      re-train on pooled data, new evals needed for each capability.
#   C. TIES merge the three LoRAs into one: zero training, ~2 hours of
#      engineering to run the merge + eval.  Some quality drop on each
#      task but typically <2 percentage points.
#
# DECISION: TIES merge (option C).  The $/capability trade-off favours
# free merging unless the merged model drops >5 points on any eval.
#
# BUSINESS IMPACT:
#   - Infra saving: drop from 3 * 14 GB to 1 * 14 GB per region (3
#     regions = 28 GB freed).  At ~S$1.20/GB-month on managed GPU,
#     that is ~S$33/region/month * 3 = S$99/month of inference VRAM.
#   - Latency: end-to-end KYC -> fraud -> explain journey drops from
#     ~780 ms (three hops) to ~320 ms (single inference).  On the
#     bank's 180,000 onboarding journeys/month, that saves ~23 agent-
#     hours of wait-time monitoring, or ~S$4,100/month in operations
#     cost.
#   - Avoided retraining: option B would have consumed ~S$18,000 of
#     GPU time plus ~6 weeks of engineering effort.
#
# RISK: merged model degrades on one of the tasks (most likely fraud,
# which is the most narrowly-scoped).  Mitigation: re-run the three
# task evaluation suites and fall back to option A if any eval drops
# below the SLA threshold.

print("Singapore fintech — three-LoRA merge decision:")
rows = [
    ("Option A: three deployments", 3 * 14, 780, 0),
    ("Option B: retrain multi-task", 1 * 14, 320, 18_000),
    ("Option C: TIES merge", 1 * 14, 320, 120),
]
decision_df = pl.DataFrame(
    {
        "Option": [r[0] for r in rows],
        "VRAM_GB": [r[1] for r in rows],
        "Latency_ms": [r[2] for r in rows],
        "Upfront_cost_SGD": [r[3] for r in rows],
    }
)
print(decision_df)
monthly_saving_sgd = 99 + 4_100
annual_saving_sgd = monthly_saving_sgd * 12
print(f"\n  Monthly saving (option C vs A): S${monthly_saving_sgd:,}")
print(f"  Annual saving:                  S${annual_saving_sgd:,}")
print(f"  Recommended: TIES merge (option C) with per-task eval gate")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert annual_saving_sgd > 0, "Task 4: fintech should see positive savings"
print("✓ Checkpoint 4 passed — fintech merge decision analysed\n")


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
  [x] Built TIES merge: trim -> elect sign -> merge
  [x] Built SLERP merge: spherical interpolation on the hypersphere
  [x] Visualised that SLERP preserves weight norms better than linear
  [x] Applied merging to a Singapore fintech three-LoRA scenario
      (~S$50k/year saving vs separate deployments, zero retraining)

  KEY INSIGHT: merging is free compute.  TIES handles sign conflict,
  SLERP handles norm drift, and task arithmetic composes them for
  targeted capability addition or removal.

  Next: 05_quantisation.py surveys how we shrink the merged model for
  deployment on smaller hardware.
"""
)
