# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 4.5: Cross-Encoder Reranking + RAGAS + HyDE + Pipeline
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement an LLM-based cross-encoder reranker
#   - Evaluate RAG quality with RAGAS (4 metrics as LLM-as-judge)
#   - Implement HyDE (Hypothetical Document Embeddings) query expansion
#   - Wire a full retrieve -> rerank -> generate RAG pipeline
#   - Apply end-to-end RAG to a Singapore insurance claims assistant
#
# PREREQUISITES: Exercises 4.2, 4.3, 4.4
# ESTIMATED TIME: ~60 min
#
# TASKS:
#   1. Build the retrieval substrate (dense + BM25 + hybrid)
#   2. Implement cross_encoder_rerank
#   3. Implement compute_ragas_metrics (faithfulness, answer relevance,
#      context relevance, context recall)
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

import matplotlib.pyplot as plt
import numpy as np

from shared.mlfp06.ex_4 import (
    DenseVectorStore,
    EMBED_DIM,
    OUTPUT_DIR,
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
print(f"Indexing {len(chunk_subset)} chunks...")

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
    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    return fused[:top_k]


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why reranking, RAGAS, and HyDE all exist
# ════════════════════════════════════════════════════════════════════════
# Retrieval gets you CANDIDATE documents. Everything after retrieval is
# about precision: "of the top-20 candidates, which 3 should I actually
# inject into the prompt?".
#
# RERANKING: bi-encoders (the dense embedding model) compare query and
# document INDEPENDENTLY, which is fast but lossy. A cross-encoder feeds
# query AND document together through the same transformer, which is
# accurate but O(N) per query. The production pattern: bi-encoder
# retrieves top-100, cross-encoder reranks to top-5.
#
# RAGAS: four decomposed metrics that explain RAG failure modes.
#   - Low faithfulness + high answer relevance = hallucination (the
#     model made up a plausible answer unsupported by context)
#   - High faithfulness + low answer relevance = wrong question
#     (the answer is grounded but doesn't address the query)
#   - Low context relevance = retrieval failed
#   - Low context recall = the corpus doesn't contain the answer
#
# HyDE: query embeddings and document embeddings live in DIFFERENT
# semantic regions. "What causes inflation?" is short and
# interrogative; a paragraph about inflation causes is long and
# declarative. HyDE generates a hypothetical answer first, embeds the
# hypothetical (which is close to real answers in embedding space), and
# retrieves against that. One extra LLM call per query; bigger recall
# gain on abstract questions.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Cross-encoder reranker
# ════════════════════════════════════════════════════════════════════════


async def cross_encoder_rerank(
    query: str, candidates: list[dict], top_k: int = 3
) -> list[dict]:
    """Re-rank candidates using LLM-based cross-encoding.

    Production would use a dedicated cross-encoder model
    (ms-marco-MiniLM-L-6-v2) — here we use Delegate as a pedagogical stand-in.
    """
    delegate = make_delegate(budget_usd=1.0)
    scored = []
    for candidate in candidates[:10]:
        prompt = (
            "Rate the relevance of this passage to the query on a scale of 0-10.\n\n"
            f"Query: {query[:300]}\n"
            f"Passage: {candidate['text'][:500]}\n\n"
            "Output ONLY a single number (0-10):"
        )
        response = await delegate_text(delegate, prompt)
        try:
            score = float(re.search(r"[\d.]+", response).group())
            score = min(max(score, 0), 10)
        except (AttributeError, ValueError):
            score = 5.0
        scored.append({**candidate, "rerank_score": score})
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — RAGAS metrics (LLM-as-judge)
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
    """RAGAS-style decomposition via LLM-as-judge."""
    delegate = make_delegate(budget_usd=1.0)
    faithfulness = await _judge_score(
        delegate,
        f"Given the context and answer, rate how well the answer is "
        f"supported by the context (0.0=not supported, 1.0=fully supported).\n\n"
        f"Context: {context[:500]}\nAnswer: {answer[:300]}\n\n"
        f"Output ONLY a number between 0.0 and 1.0:",
    )
    answer_relevance = await _judge_score(
        delegate,
        f"Rate how well this answer addresses the question "
        f"(0.0=irrelevant, 1.0=perfectly relevant).\n\n"
        f"Question: {question[:300]}\nAnswer: {answer[:300]}\n\n"
        f"Output ONLY a number between 0.0 and 1.0:",
    )
    context_relevance = await _judge_score(
        delegate,
        f"Rate how relevant this context is to the question "
        f"(0.0=irrelevant, 1.0=highly relevant).\n\n"
        f"Question: {question[:300]}\nContext: {context[:500]}\n\n"
        f"Output ONLY a number between 0.0 and 1.0:",
    )
    context_recall = await _judge_score(
        delegate,
        f"Rate how much of the ground-truth answer can be found in the "
        f"retrieved context (0.0=none, 1.0=all information present).\n\n"
        f"Ground truth: {ground_truth[:300]}\nContext: {context[:500]}\n\n"
        f"Output ONLY a number between 0.0 and 1.0:",
    )
    return {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_relevance": context_relevance,
        "context_recall": context_recall,
    }


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — HyDE
# ════════════════════════════════════════════════════════════════════════


async def hyde_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """HyDE: generate a hypothetical answer, embed it, retrieve similar docs."""
    delegate = make_delegate(budget_usd=1.0)
    hyde_prompt = (
        "Write a short paragraph (3-5 sentences) that would be the ideal "
        "answer to this question. It does not need to be factually correct "
        "— it should contain the key concepts and vocabulary that a real "
        "answer would use.\n\n"
        f"Question: {query}\n\nHypothetical answer:"
    )
    hypo_doc = await delegate_text(delegate, hyde_prompt)
    print(f"  HyDE hypothetical: {hypo_doc[:150]}...")
    hypo_emb = await generate_embedding(hypo_doc, delegate)
    return dense_store.search(hypo_emb, top_k=top_k)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Full RAG pipeline
# ════════════════════════════════════════════════════════════════════════


async def full_rag_pipeline(query: str) -> dict:
    """Retrieve (hybrid) -> rerank (cross-encoder) -> generate."""
    candidates = await hybrid_search(query, top_k=10)
    reranked = await cross_encoder_rerank(query, candidates, top_k=3)
    context = "\n\n---\n\n".join(r["text"] for r in reranked)
    answer = await rag_answer(query, context, budget_usd=0.5)
    return {
        "query": query,
        "answer": answer,
        "context": context,
        "n_retrieved": len(candidates),
        "n_reranked": len(reranked),
    }


async def run_pipeline_and_eval() -> tuple[list[dict], dict]:
    pipeline_results = []
    ragas_accum = {
        "faithfulness": [],
        "answer_relevance": [],
        "context_relevance": [],
        "context_recall": [],
    }
    for i in range(3):
        q = eval_questions[i]
        gt = eval_answers[i]
        print(f"\n  Q{i+1}: {q[:80]}...")
        result = await full_rag_pipeline(q)
        print(f"  A: {result['answer'][:180]}...")
        pipeline_results.append(result)

        metrics = await compute_ragas_metrics(
            q, result["answer"], result["context"], gt
        )
        for k, v in metrics.items():
            ragas_accum[k].append(v)
        print(
            f"    RAGAS: faith={metrics['faithfulness']:.2f} "
            f"rel={metrics['answer_relevance']:.2f} "
            f"ctx={metrics['context_relevance']:.2f} "
            f"recall={metrics['context_recall']:.2f}"
        )

    avg_metrics = {k: sum(v) / len(v) for k, v in ragas_accum.items()}
    return pipeline_results, avg_metrics


pipeline_results, avg_ragas = run_async(run_pipeline_and_eval())

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(pipeline_results) >= 3, "Pipeline should process at least 3 questions"
assert all("answer" in r for r in pipeline_results), "Each result should have an answer"
assert all(0 <= v <= 1 for v in avg_ragas.values()), "RAGAS metrics must be in [0,1]"
print("\n--- Checkpoint passed --- full RAG pipeline with RAGAS evaluation\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — HyDE comparison
# ════════════════════════════════════════════════════════════════════════


async def compare_hyde():
    delegate = make_delegate(budget_usd=0.5)
    q = eval_questions[0]
    print(f"\nComparing direct vs HyDE retrieval for: {q[:80]}...")
    q_emb = await generate_embedding(q, delegate)
    direct = dense_store.search(q_emb, top_k=3)
    hyde = await hyde_retrieve(q, top_k=3)
    print(f"  Direct top-1 score: {direct[0]['score']:.3f}")
    print(f"  HyDE   top-1 score: {hyde[0]['score']:.3f}")
    return direct, hyde


direct_results, hyde_results = run_async(compare_hyde())
assert len(hyde_results) > 0, "HyDE should return results"
print("\n--- HyDE comparison complete ---\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Visualise RAGAS scores
# ════════════════════════════════════════════════════════════════════════

print("\nAverage RAGAS metrics across 3 questions:")
for k, v in avg_ragas.items():
    print(f"  {k}: {v:.2f}")

plot_ragas_metrics(
    avg_ragas,
    title="RAGAS Evaluation — Full Pipeline (hybrid -> rerank -> generate)",
    filename="ex4_05_ragas_metrics.png",
)

# R9A: RAGAS radar chart — shows the "shape" of RAG quality at a glance.
# A perfect system is a full diamond; a hallucinating system has high
# answer_relevance but low faithfulness (top-left dip).
labels = list(avg_ragas.keys())
values = [avg_ragas[l] for l in labels]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
values_closed = values + [values[0]]
angles_closed = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles_closed, values_closed, "o-", linewidth=2, color="steelblue")
ax.fill(angles_closed, values_closed, alpha=0.25, color="steelblue")
ax.set_xticks(angles)
ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
ax.set_title(
    "RAGAS Radar — Pipeline Quality Shape",
    fontsize=13,
    fontweight="bold",
    pad=20,
)
# Draw the 0.70 target ring
target_ring = [0.70] * (len(angles) + 1)
ax.plot(
    angles_closed,
    target_ring,
    "--",
    color="grey",
    alpha=0.5,
    linewidth=1,
    label="target=0.70",
)
ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05), fontsize=8)
plt.tight_layout()
fname = OUTPUT_DIR / "ex4_05_ragas_radar.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname}")

# R9A: retrieval precision at k — how many of the top-k reranked chunks
# were actually relevant (judged by context_relevance > 0.5)?
# This approximates precision@k for the full pipeline.
fig, ax = plt.subplots(figsize=(7, 4))
k_values = list(range(1, len(pipeline_results[0].get("context", "").split("---")) + 1))
# Use reranked chunk count per query as a proxy
precisions = []
for result in pipeline_results:
    chunks = [c.strip() for c in result["context"].split("---") if c.strip()]
    cumulative = []
    for k in range(1, len(chunks) + 1):
        # Heuristic: chunks that appear in the answer are "relevant"
        relevant = sum(
            1
            for c in chunks[:k]
            if any(word in result["answer"].lower() for word in c.lower().split()[:3])
        )
        cumulative.append(relevant / k)
    precisions.append(cumulative)

if precisions:
    max_k = max(len(p) for p in precisions)
    avg_precision = []
    for k in range(max_k):
        vals = [p[k] for p in precisions if k < len(p)]
        avg_precision.append(sum(vals) / len(vals) if vals else 0)
    ax.plot(
        range(1, max_k + 1),
        avg_precision,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("k (number of retrieved chunks)")
    ax.set_ylabel("Precision@k (approx)")
    ax.set_title(
        "Retrieval Precision@k — Reranked Pipeline", fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = OUTPUT_DIR / "ex4_05_precision_at_k.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")

# INTERPRETATION: The radar chart reveals your pipeline's "personality":
# - Balanced diamond = solid RAG system
# - Low faithfulness + high relevance = hallucination (the model answers
#   the right question but invents facts)
# - Low context_recall = your corpus doesn't contain the answer
# The precision@k curve shows reranking value: if precision drops sharply
# after k=1, the reranker is concentrating relevance at the top.


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore insurance claims assistant
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: AIA Singapore runs a claims-processing assistant for its
# agents. When a customer files an accident claim, the agent needs to
# answer questions like "Is physiotherapy covered under the Essential
# Classic plan after a motor vehicle accident if the policy was issued
# in 2021 and the accident happened in Malaysia?".
#
# Answering this requires pulling from:
#   - The customer's policy document (coverage, exclusions, effective dates)
#   - The claims manual (territorial scope, third-party rules)
#   - The medical schedule (what counts as physiotherapy, max visits)
#
# WHY THE FULL PIPELINE MATTERS:
#   - Hybrid retrieval finds both the policy clause ("Essential Classic")
#     via BM25 AND the medical schedule ("physiotherapy covered up to 30
#     sessions post-accident") via dense embeddings.
#   - Cross-encoder reranking pushes the MOST relevant 3 clauses to the
#     top — critical because the wrong clause means a wrong payout
#     decision.
#   - RAGAS metrics run in shadow mode: every claim decision is scored,
#     and any answer with faithfulness < 0.8 is flagged for human review
#     BEFORE it goes to the customer.
#   - HyDE helps when the agent types a conversational query instead of
#     insurance jargon ("can we pay for his back treatment after the
#     car crash?").
#
# BUSINESS IMPACT: AIA Singapore processes ~200,000 claims per year.
# Agents currently spend ~15 minutes per claim looking up policy
# clauses; the assistant cuts this to ~3 minutes. At a loaded agent
# cost of ~S$40/hour, the saving is 12 min × 200K / 60 × S$40 = S$1.6M/year
# in agent time. The RAGAS faithfulness gate is what prevents the bigger
# cost: a regulatory fine from MAS for inconsistent claims handling
# (historically up to S$1M per finding). That's why reranking +
# evaluation isn't polish — it's the compliance control.


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
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built an LLM-based cross-encoder reranker
  [x] Implemented RAGAS (faithfulness, answer relevance, context
      relevance, context recall) as LLM-as-judge
  [x] Implemented HyDE query expansion (generate hypothetical, embed it)
  [x] Wired a full retrieve -> rerank -> generate pipeline
  [x] Visualised RAGAS metrics against a 0.70 target
  [x] Mapped the full pipeline to a Singapore insurance claims use case

  PRODUCTION RECIPE:
    1. Chunk documents (sentence chunking is a good default)
    2. Dense index in a vector DB + BM25 index in SQLite FTS5 / Elasticsearch
    3. Hybrid retrieve top-20 with RRF(k=60)
    4. Cross-encoder rerank to top-3
    5. Inject top-3 into the answer prompt
    6. Run RAGAS in shadow mode on every production query
    7. Alert on faithfulness < 0.8 (hallucination signal)
    8. Add HyDE for abstract-query corners (long-tail improvements)

  RAG VS FINE-TUNING:
    Use RAG when documents change frequently, need citations, audit trail
    Use fine-tuning when: style adaptation, domain vocabulary, latency
    Combine: fine-tune on domain + RAG for up-to-date facts

  NEXT: Exercise 5 moves from RAG (the LLM READS documents) to AGENTS
  (the LLM takes ACTIONS — calls tools, observes results, iterates).
"""
)
