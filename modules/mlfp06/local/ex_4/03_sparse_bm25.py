# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 4.3: Sparse Retrieval with BM25 From Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement BM25 from scratch (tokenisation, IDF, term frequency)
#   - Understand why keyword matching beats embeddings for exact terms
#   - Tune BM25's k1 and b parameters
#   - Apply BM25 to a Singapore HDB regulation lookup
#
# PREREQUISITES: Exercise 4.1 (chunking), M4.6 (keyword search background)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load and chunk the corpus
#   2. Implement the BM25 class
#   3. Run BM25 on a real eval question
#   4. Visualise the score distribution
#   5. Apply: HDB regulation lookup
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import re
from collections import Counter

from shared.mlfp06.ex_4 import (
    load_rag_corpus,
    plot_score_distribution,
    split_corpus,
)


def chunk_sentence(text: str, max_chunk_chars: int = 500) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chunk_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load corpus and chunk
# ════════════════════════════════════════════════════════════════════════

corpus = load_rag_corpus(sample_size=1000)
doc_texts, eval_questions, eval_answers = split_corpus(corpus, n_eval=20)

all_chunks = []
for text in doc_texts:
    all_chunks.extend(chunk_sentence(text, 500))
chunk_subset = all_chunks[:200]
print(f"BM25 index: {len(chunk_subset)} chunks")


# ════════════════════════════════════════════════════════════════════════
# THEORY — BM25 as IDF-weighted term frequency
# ════════════════════════════════════════════════════════════════════════
#     score(q, d) = sum_t  IDF(t) * (tf(t,d) * (k1 + 1))
#                         / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
# k1 controls tf saturation; b controls length normalisation.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Implement BM25 from scratch
# ════════════════════════════════════════════════════════════════════════


class BM25:
    """BM25 sparse retrieval — from-scratch implementation."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.N = len(documents)
        # TODO: Tokenise every document (call self._tokenise) and store:
        #   self.doc_tokens  — list[list[str]]
        #   self.doc_lengths — list[int]
        #   self.avgdl       — float (average document length in tokens)
        ____

        # TODO: Compute self.df (Counter mapping term -> #docs containing it)
        #       and self.tf (list of Counter objects, one per document).
        ____

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        # Hint: return re.findall(r"\w+", text.lower())
        ____

    def _idf(self, term: str) -> float:
        # TODO: Return log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        ____

    def score(self, query: str, doc_idx: int) -> float:
        # TODO: For each query term, compute IDF and the BM25 tf component,
        # then sum them up. See the formula in the theory block above.
        ____

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        # TODO: Score every document, sort descending, return top_k as
        # [{"text", "score", "doc_idx"}, ...]
        ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Run BM25 on a real evaluation query
# ════════════════════════════════════════════════════════════════════════

bm25 = BM25(chunk_subset)
test_query = eval_questions[0]
bm25_results = bm25.search(test_query, top_k=5)
for i, r in enumerate(bm25_results[:3]):
    print(f"  {i+1}. score={r['score']:.3f}: {r['text'][:90]}...")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(bm25_results) == 5
assert all(r["score"] >= 0 for r in bm25_results)
assert bm25.avgdl > 0
print("\n--- Checkpoint passed ---\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the score distribution
# ════════════════════════════════════════════════════════════════════════

all_results = bm25.search(test_query, top_k=len(chunk_subset))
score_values = [r["score"] for r in all_results]
# TODO: Call plot_score_distribution(...)
____


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore HDB regulation lookup
# ════════════════════════════════════════════════════════════════════════
# Domain acronyms (BTO, MOP, EIP, SPR) are out-of-distribution for
# generic embeddings. BM25 matches them exactly.
#
# BUSINESS IMPACT: HDB handles ~1M customer interactions/year. BM25-based
# lookup cuts officer time by 75s × 1M = 20,833 hours/year ≈ S$729K
# saved annually.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print(
    """
  [x] Implemented BM25 from scratch
  [x] Visualised the score distribution

  Next: 04_hybrid_rrf.py combines dense and sparse with RRF fusion...
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

# Primary lens: Retrieval (recall@k, context utilisation, faithfulness).
# Secondary: Output (judge on final answers). Classic RAG failures —
# over-narrow chunks, stale index, judge flags fabrication.
if False:  # scaffold — requires an evaluated RAG pipeline
    obs = LLMObservatory(run_id="ex_4_rag_run")
    # obs.retrieval.evaluate(
    #     queries=eval_queries,
    #     retrieved_contexts=per_query_chunks,
    #     answers=generator_answers,
    #     ground_truth_ids=per_query_relevant_ids,
    #     k=5,
    # )
    print("\n── LLM Observatory Report ──")
    findings = obs.report()

# ══════ EXPECTED OUTPUT (synthesised reference) ══════
# ════════════════════════════════════════════════════════════════
#   LLM Observatory — composite Prescription Pad
# ════════════════════════════════════════════════════════════════
#   [!] Retrieval  (WARNING): recall@5 = 0.62 — chunks too narrow
#       Fix: increase chunk_size from 256 to 512 tokens, OR add
#            HyDE query rewriting before dense retrieval.
#   [✓] Output     (HEALTHY): faithfulness 0.87 (answers grounded in
#       retrieved chunks even when recall is imperfect).
#   [?] Attention / Agent / Alignment / Governance (n/a)
# ════════════════════════════════════════════════════════════════
#
# STUDENT INTERPRETATION GUIDE — reading the Prescription Pad:
#
#  [RETRIEVAL LENS] recall@5 = 0.62 is the SIGNATURE of over-narrow
#     chunks — the index contains the right passage but the retriever
#     returns a neighbour that misses the key entity. This is the
#     failure the chunking exercise (ex_4.1) prepared you to diagnose.
#     >> Prescription: (a) increase chunk_size, (b) add overlap, (c)
#        switch to hybrid BM25+dense (ex_4.4), or (d) rerank (ex_4.5).
#  [OUTPUT LENS] Faithfulness 0.87 on a recall of 0.62 means the
#     generator is honest — when it doesn't have the right chunk it
#     says so instead of fabricating. That's the GOOD failure mode.
#     The bad failure mode would be high recall + low faithfulness
#     (retrieval works but the LLM still hallucinates).
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
