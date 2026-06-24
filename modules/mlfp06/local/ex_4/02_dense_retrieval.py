# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 4.2: Dense Retrieval with Embeddings + Cosine
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a dense vector store with cosine similarity ranking
#   - Embed a corpus and a query into the same semantic space
#   - Inspect retrieved chunks and their similarity scores
#   - Apply dense retrieval to a Singapore healthcare FAQ
#
# PREREQUISITES: Exercise 4.1 (chunking), Exercise 1 (Delegate)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load and chunk the corpus
#   2. Build a DenseVectorStore with 30 embedded chunks
#   3. Embed a real eval question and search the store
#   4. Visualise the similarity score distribution
#   5. Apply: Singapore polyclinic FAQ bot
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import re

from shared.mlfp06.ex_4 import (
    DenseVectorStore,
    EMBED_DIM,
    embed_many,
    generate_embedding,
    load_rag_corpus,
    make_delegate,
    plot_score_distribution,
    run_async,
    split_corpus,
)


def chunk_sentence(text: str, max_chunk_chars: int = 500) -> list[str]:
    """Sentence-boundary chunking from Exercise 4.1."""
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
# TASK 1 — Load corpus and build a chunk subset
# ════════════════════════════════════════════════════════════════════════

corpus = load_rag_corpus(sample_size=1000)
doc_texts, eval_questions, eval_answers = split_corpus(corpus, n_eval=20)

all_chunks = []
for i, text in enumerate(doc_texts):
    for j, chunk in enumerate(chunk_sentence(text, 500)):
        all_chunks.append({"doc_idx": i, "chunk_idx": j, "text": chunk})

chunk_subset = [c["text"] for c in all_chunks[:30]]
print(f"Embedding {len(chunk_subset)} chunks...")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Dense retrieval as semantic search
# ════════════════════════════════════════════════════════════════════════
# A dense retriever represents every document and every query as a point
# in a high-dimensional vector space. "Similar meaning" becomes "nearby
# points". The similarity metric is cosine similarity — the angle
# between two vectors, ignoring their magnitudes.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the DenseVectorStore
# ════════════════════════════════════════════════════════════════════════

# TODO: Call run_async(embed_many(chunk_subset, budget_usd=3.0))
embeddings = ____

# TODO: Instantiate DenseVectorStore and call .add() for each (text, embedding)
dense_store = ____
____

print(f"Dense store: {len(dense_store.documents)} chunks, dim={EMBED_DIM}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Embed a query and search
# ════════════════════════════════════════════════════════════════════════


async def dense_search(query: str, top_k: int = 5) -> list[dict]:
    delegate = make_delegate(budget_usd=0.5)
    # TODO: Generate an embedding for query via generate_embedding,
    # then call dense_store.search with top_k.
    ____


test_query = eval_questions[0]
dense_results = run_async(dense_search(test_query, top_k=5))
for i, r in enumerate(dense_results):
    print(f"  {i+1}. score={r['score']:.3f}: {r['text'][:80]}...")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(dense_store.documents) == len(chunk_subset)
assert len(embeddings[0]) == EMBED_DIM
assert len(dense_results) == 5
print("\n--- Checkpoint passed ---\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the similarity distribution
# ════════════════════════════════════════════════════════════════════════

all_results = dense_store.search(embeddings[0], top_k=len(chunk_subset))
score_values = [r["score"] for r in all_results]

# TODO: Call plot_score_distribution with score_values, a title, xlabel,
# and filename "ex4_02_dense_score_dist.png".
____


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore polyclinic FAQ bot
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: SingHealth handles ~30,000 FAQ phone calls per month asking
# variations of 200 canonical questions. Dense retrieval over an FAQ
# corpus handles paraphrases that BM25 would miss.
#
# BUSINESS IMPACT: At S$8/call × 30K calls/month = S$240K/month on phone
# triage. A 60% deflection saves S$144K/month = S$1.7M/year.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print(
    """
  [x] Built a dense vector store with cosine similarity
  [x] Retrieved top-k chunks for a real query
  [x] Inspected similarity distribution

  Next: 03_sparse_bm25.py shows why BM25 is still in the mix...
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
