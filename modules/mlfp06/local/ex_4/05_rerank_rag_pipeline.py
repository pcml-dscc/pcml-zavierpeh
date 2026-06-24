# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 4.5: Cross-Encoder Reranking + RAGAS + HyDE + Pipeline
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement an LLM-based cross-encoder reranker
#   - Evaluate RAG quality with RAGAS (4 LLM-as-judge metrics)
#   - Implement HyDE query expansion
#   - Wire a full retrieve -> rerank -> generate RAG pipeline
#   - Apply end-to-end RAG to a Singapore insurance claims assistant
#
# PREREQUISITES: Exercises 4.2, 4.3, 4.4
# ESTIMATED TIME: ~60 min
#
# TASKS:
#   1. Build the retrieval substrate (dense + BM25 + hybrid)
#   2. Implement cross_encoder_rerank
#   3. Implement compute_ragas_metrics
#   4. Implement hyde_retrieve
#   5. Assemble the full RAG pipeline
#   6. Visualise RAGAS scores
#   7. Apply: Singapore insurance claims assistant
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
    delegate_text,
    embed_many,
    generate_embedding,
    load_rag_corpus,
    make_delegate,
    plot_ragas_metrics,
    rag_answer,
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


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    doc_scores: dict[int, float] = {}
    doc_texts: dict[int, str] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            doc_id = hash(item["text"])
            doc_texts[doc_id] = item["text"]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"text": doc_texts[d], "rrf_score": s} for d, s in fused]


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build the retrieval substrate
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


async def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    delegate = make_delegate(budget_usd=0.5)
    q_emb = await generate_embedding(query, delegate)
    dense_results = dense_store.search(q_emb, top_k=top_k * 2)
    sparse_results = bm25.search(query, top_k=top_k * 2)
    return reciprocal_rank_fusion([dense_results, sparse_results])[:top_k]


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Cross-encoder reranker
# ════════════════════════════════════════════════════════════════════════


async def cross_encoder_rerank(
    query: str, candidates: list[dict], top_k: int = 3
) -> list[dict]:
    """Re-rank candidates by asking the LLM to score relevance 0-10."""
    delegate = make_delegate(budget_usd=1.0)
    scored = []
    for candidate in candidates[:10]:
        # TODO: Build a prompt asking for a 0-10 relevance score between
        # query and candidate["text"], call delegate_text, parse a float.
        # Attach it as "rerank_score" on a copy of the candidate dict.
        ____
    # TODO: Sort by rerank_score descending and return top_k.
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — RAGAS metrics
# ════════════════════════════════════════════════════════════════════════


async def _judge_score(delegate, prompt: str) -> float:
    response = await delegate_text(delegate, prompt)
    try:
        return min(max(float(re.search(r"[\d.]+", response).group()), 0), 1)
    except (AttributeError, ValueError):
        return 0.5


async def compute_ragas_metrics(
    question: str, answer: str, context: str, ground_truth: str
) -> dict:
    """Four RAGAS metrics via LLM-as-judge."""
    delegate = make_delegate(budget_usd=1.0)
    # TODO: For each of (faithfulness, answer_relevance, context_relevance,
    # context_recall), build a judge prompt and call _judge_score.
    # Return them in a dict.
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — HyDE
# ════════════════════════════════════════════════════════════════════════


async def hyde_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """HyDE: generate a hypothetical answer, embed IT, retrieve against it."""
    delegate = make_delegate(budget_usd=1.0)
    # TODO: Prompt the LLM for a short paragraph that *would* answer the
    # question, then generate_embedding on that paragraph and call
    # dense_store.search with it.
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Full RAG pipeline
# ════════════════════════════════════════════════════════════════════════


async def full_rag_pipeline(query: str) -> dict:
    """Retrieve (hybrid) -> rerank (cross-encoder) -> generate."""
    # TODO: 1. candidates = await hybrid_search(query, top_k=10)
    #       2. reranked = await cross_encoder_rerank(query, candidates, top_k=3)
    #       3. context = "\n\n---\n\n".join(r["text"] for r in reranked)
    #       4. answer = await rag_answer(query, context)
    #       5. Return {"query", "answer", "context", "n_retrieved", "n_reranked"}
    ____


async def run_pipeline_and_eval() -> tuple[list[dict], dict]:
    pipeline_results = []
    ragas_accum = {
        "faithfulness": [],
        "answer_relevance": [],
        "context_relevance": [],
        "context_recall": [],
    }
    for i in range(3):
        q, gt = eval_questions[i], eval_answers[i]
        result = await full_rag_pipeline(q)
        pipeline_results.append(result)
        metrics = await compute_ragas_metrics(
            q, result["answer"], result["context"], gt
        )
        for k, v in metrics.items():
            ragas_accum[k].append(v)
        print(f"  Q{i+1}: {q[:60]}... -> faith={metrics['faithfulness']:.2f}")
    avg = {k: sum(v) / len(v) for k, v in ragas_accum.items()}
    return pipeline_results, avg


pipeline_results, avg_ragas = run_async(run_pipeline_and_eval())

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(pipeline_results) >= 3
assert all("answer" in r for r in pipeline_results)
assert all(0 <= v <= 1 for v in avg_ragas.values())
print("\n--- Checkpoint passed ---\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise RAGAS scores
# ════════════════════════════════════════════════════════════════════════

# TODO: Call plot_ragas_metrics(avg_ragas, title=..., filename="ex4_05_ragas_metrics.png")
____


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore insurance claims assistant
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: AIA Singapore runs a claims-processing assistant. Agents ask
# questions like "Is physiotherapy covered under Essential Classic after
# a motor vehicle accident in Malaysia if the policy was issued in 2021?".
# The full RAG pipeline (hybrid + rerank + generate + RAGAS gate) is
# what separates a system that answers vs one that hallucinates and
# triggers a MAS fine.
#
# BUSINESS IMPACT: ~S$1.6M/year saved on agent lookup time for 200K
# claims/year. The RAGAS faithfulness gate is the compliance control —
# MAS fines up to S$1M per finding for inconsistent claims handling.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print(
    """
  [x] Built a cross-encoder reranker
  [x] Implemented RAGAS (4 metrics) as LLM-as-judge
  [x] Implemented HyDE query expansion
  [x] Assembled the full retrieve -> rerank -> generate pipeline

  NEXT: Exercise 5 moves from RAG (the LLM READS documents) to AGENTS
  (the LLM takes ACTIONS — calls tools, observes results, iterates).
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
