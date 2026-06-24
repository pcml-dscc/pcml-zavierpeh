# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 4.4: Hybrid Retrieval with Reciprocal Rank Fusion
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Combine dense and sparse retrieval with Reciprocal Rank Fusion (RRF)
#   - Understand why rank-based fusion is more robust than score fusion
#   - Implement RRF with k=60
#   - Apply hybrid retrieval to a Singapore fintech compliance search
#
# PREREQUISITES: Exercises 4.2 (dense), 4.3 (sparse)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Build a dense store and a BM25 index over the same chunks
#   2. Implement reciprocal_rank_fusion
#   3. Run hybrid retrieval on a real query
#   4. Visualise dense / sparse / hybrid rankings
#   5. Apply: MAS fintech compliance search
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import re
from collections import Counter

from shared.mlfp06.ex_4 import (
    DenseVectorStore,
    EMBED_DIM,
    embed_many,
    generate_embedding,
    load_rag_corpus,
    make_delegate,
    plot_strategy_comparison,
    run_async,
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


class BM25:
    """Carry-over BM25 from Exercise 4.3."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.documents, self.N = k1, b, documents, len(documents)
        self.doc_tokens = [re.findall(r"\w+", d.lower()) for d in documents]
        self.doc_lengths = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(self.N, 1)
        self.df: dict[str, int] = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] += 1
        self.tf = [Counter(tokens) for tokens in self.doc_tokens]

    def _idf(self, term: str) -> float:
        df_t = self.df.get(term, 0)
        return math.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = re.findall(r"\w+", query.lower())
        doc_len = self.doc_lengths[doc_idx]
        tf_doc = self.tf[doc_idx]
        total = 0.0
        for term in query_tokens:
            tf_val = tf_doc.get(term, 0)
            num = tf_val * (self.k1 + 1)
            den = tf_val + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            total += self._idf(term) * num / den
        return total

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"text": self.documents[idx], "score": s, "doc_idx": idx}
            for idx, s in scores[:top_k]
        ]


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build the dense store AND BM25 index
# ════════════════════════════════════════════════════════════════════════

corpus = load_rag_corpus(sample_size=1000)
doc_texts, eval_questions, eval_answers = split_corpus(corpus, n_eval=20)

all_chunks = []
for text in doc_texts:
    all_chunks.extend(chunk_sentence(text, 500))
chunk_subset = all_chunks[:30]

embeddings = run_async(embed_many(chunk_subset, budget_usd=3.0))
dense_store = DenseVectorStore()
for i, (text, emb) in enumerate(zip(chunk_subset, embeddings)):
    dense_store.add(text, emb, {"chunk_idx": i})
bm25 = BM25(chunk_subset)
print(f"Indexed {len(chunk_subset)} chunks into both dense + BM25")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Reciprocal Rank Fusion
# ════════════════════════════════════════════════════════════════════════
#     RRF_score(d) = sum over rankers r:  1 / (k + rank_r(d))
# RRF ignores raw scores and uses only ranks, so fusion is stable even
# when the two rankers produce numbers on very different scales.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Implement reciprocal_rank_fusion
# ════════════════════════════════════════════════════════════════════════


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Combine multiple ranked lists using RRF."""
    # TODO: For each ranked list, for each (rank, item) enumerate from 1,
    # accumulate 1 / (k + rank) into a dict keyed by hash(item["text"]).
    # Sort the resulting dict by score descending and return
    # [{"text", "rrf_score"}, ...].
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Run hybrid retrieval on a real query
# ════════════════════════════════════════════════════════════════════════


async def hybrid_search(query: str, top_k: int = 5) -> dict:
    delegate = make_delegate(budget_usd=0.5)
    # TODO: Generate an embedding for the query, call dense_store.search
    # and bm25.search for top_k*2 each, then fuse via reciprocal_rank_fusion.
    # Return {"dense": ..., "sparse": ..., "fused": ...}.
    ____


test_query = eval_questions[0]
results = run_async(hybrid_search(test_query, top_k=5))

print("\nHybrid (RRF) top-3:")
for i, r in enumerate(results["fused"][:3]):
    print(f"  {i+1}. rrf={r['rrf_score']:.4f}: {r['text'][:80]}...")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(results["fused"]) > 0
assert all("rrf_score" in r for r in results["fused"])
print("\n--- Checkpoint passed ---\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise dense / sparse / hybrid rankings
# ════════════════════════════════════════════════════════════════════════

dense_scores = [r["score"] for r in results["dense"]]
sparse_scores = [
    r["score"] / max(r["score"] for r in results["sparse"]) for r in results["sparse"]
]
fused_scores = [r["rrf_score"] * 100 for r in results["fused"]]

# TODO: Call plot_strategy_comparison with a dict of 3 strategies.
____


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore fintech compliance search
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: MAS-licensed fintech runs compliance search over 5,000
# regulation documents. Officers ask questions mixing acronyms ("PEP",
# "AML") and semantic phrases ("cross-border wire transfer").
#
# BUSINESS IMPACT: ~S$960K/year saved on compliance lookups at a
# 50-officer firm. Missed lookups carry regulatory fines up to S$1M
# each — hybrid retrieval is risk management, not optimisation.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print(
    """
  [x] Built dense + BM25 indexes over the same chunks
  [x] Implemented Reciprocal Rank Fusion
  [x] Visualised the three retrievers

  Next: 05_rerank_rag_pipeline.py adds reranking, RAGAS, HyDE, and the
  full RAG pipeline...
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
