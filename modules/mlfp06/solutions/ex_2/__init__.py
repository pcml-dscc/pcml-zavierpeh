# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""MLFP06 — Exercise 2: LLM Fine-Tuning (R10 directory).

Six independently runnable technique files. Each file follows the 5-phase
R10 structure: Theory -> Build -> Train -> Visualise -> Apply.

  01_lora_from_scratch.py          — LoRA as nn.Module (A @ B low-rank)
  02_adapter_from_scratch.py       — Bottleneck adapter (down->GELU->up)
  03_finetuning_landscape.py       — 10-technique decision tree
  04_model_merging.py              — TIES / DARE / SLERP / task arithmetic
  05_quantisation.py               — GPTQ / AWQ / GGUF / QLoRA survey
  06_sft_alignment_pipeline.py     — kailash-align SFT + AdapterRegistry
"""
