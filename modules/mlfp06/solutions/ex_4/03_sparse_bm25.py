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
#   2. Implement the BM25 class (tokenise, _idf, score, search)
#   3. Run BM25 on a real eval question
#   4. Visualise the score distribution
#   5. Apply: HDB (Housing Board) regulation lookup
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
    OUTPUT_DIR,
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

all_chunks: list[dict] = []
for i, text in enumerate(doc_texts):
    for j, chunk in enumerate(chunk_sentence(text, 500)):
        all_chunks.append({"doc_idx": i, "chunk_idx": j, "text": chunk})

# BM25 is cheap — index the first 200 chunks (still fast on laptop).
chunk_subset = [c["text"] for c in all_chunks[:200]]
print(f"BM25 index: {len(chunk_subset)} chunks")


# ════════════════════════════════════════════════════════════════════════
# THEORY — BM25 as IDF-weighted term frequency
# ════════════════════════════════════════════════════════════════════════
# BM25 is the production search ranker that powered Google in the 2000s,
# and still powers Elasticsearch, Lucene, and SQLite FTS5 today. It
# scores a document by summing over the query terms:
#
#     score(q, d) = sum_t  IDF(t) * (tf(t,d) * (k1 + 1))
#                         / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
#
# Components:
#   - IDF(t):  rare terms are informative; common terms ("the") are not
#   - tf(t,d): how often t appears in d (saturates — ten "dog"s is
#              barely more informative than five "dog"s, handled by k1)
#   - |d|/avgdl: length normalisation so long documents don't dominate
#   - k1, b: tunable knobs (k1=1.5, b=0.75 are the textbook defaults)
#
# Analogy: Imagine a librarian finding books that mention a phrase.
# They weight rare words ("supraventricular tachycardia") much higher
# than common words ("the heart"). A book that repeats "tachycardia"
# three times scores higher than one that mentions it once, but not
# linearly — diminishing returns.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Implement BM25 from scratch
# ════════════════════════════════════════════════════════════════════════


class BM25:
    """BM25 sparse retrieval — from-scratch implementation.

    Args:
        documents: corpus texts, one per "document" (here: per chunk).
        k1: term-frequency saturation (default 1.5 — higher means tf
            matters more).
        b: length-normalisation strength (0 = ignore length, 1 = fully
            normalise, default 0.75).
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.N = len(documents)

        # Tokenise every document once
        self.doc_tokens = [self._tokenise(d) for d in documents]
        self.doc_lengths = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(self.N, 1)

        # Document frequency: how many documents contain each term
        self.df: dict[str, int] = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] += 1

        # Term frequencies per document (Counter is a dict with .get)
        self.tf: list[Counter] = [Counter(tokens) for tokens in self.doc_tokens]

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Simple lowercased word tokeniser. Production would use a
        language-aware tokeniser (spaCy, tantivy) and stemming."""
        return re.findall(r"\w+", text.lower())

    def _idf(self, term: str) -> float:
        """Smoothed inverse document frequency (adds 0.5 to avoid
        log(0) and +1 outside to keep IDF non-negative)."""
        df_t = self.df.get(term, 0)
        return math.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score for a single query-document pair."""
        query_tokens = self._tokenise(query)
        doc_len = self.doc_lengths[doc_idx]
        tf_doc = self.tf[doc_idx]
        total = 0.0
        for term in query_tokens:
            tf_val = tf_doc.get(term, 0)
            idf = self._idf(term)
            numerator = tf_val * (self.k1 + 1)
            denominator = tf_val + self.k1 * (
                1 - self.b + self.b * doc_len / self.avgdl
            )
            total += idf * numerator / denominator
        return total

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k documents by BM25 score."""
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"text": self.documents[idx], "score": score, "doc_idx": idx}
            for idx, score in scores[:top_k]
        ]


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Run BM25 on a real evaluation query
# ════════════════════════════════════════════════════════════════════════

bm25 = BM25(chunk_subset)
print(f"Average document length: {bm25.avgdl:.1f} tokens")
print(f"Vocabulary size: {len(bm25.df)} unique terms")

test_query = eval_questions[0]
bm25_results = bm25.search(test_query, top_k=5)
print(f"\nQuery: {test_query[:100]}...")
for i, r in enumerate(bm25_results[:3]):
    print(f"  {i+1}. score={r['score']:.3f}: {r['text'][:100]}...")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(bm25_results) == 5, "BM25 should return top-5"
assert all(r["score"] >= 0 for r in bm25_results), "BM25 scores must be non-negative"
assert bm25.avgdl > 0, "BM25 should compute avg document length"
print("\n--- Checkpoint passed --- BM25 sparse retrieval from scratch\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the score distribution
# ════════════════════════════════════════════════════════════════════════

all_results = bm25.search(test_query, top_k=len(chunk_subset))
score_values = [r["score"] for r in all_results]
plot_score_distribution(
    score_values,
    title=f"BM25 Score Distribution — {test_query[:50]}...",
    xlabel="BM25 score",
    filename="ex4_03_bm25_score_dist.png",
)

# R9A: term frequency heatmap — shows which query terms contribute to
# the BM25 score of the top-5 documents. Reveals WHY a document ranks
# high (exact term overlap) and which terms are "wasted" (zero TF).
query_tokens = list(set(re.findall(r"\w+", test_query.lower())))[:10]
top_docs = bm25_results[:5]
tf_matrix = np.zeros((len(top_docs), len(query_tokens)))
for i, doc in enumerate(top_docs):
    doc_idx = doc["doc_idx"]
    for j, term in enumerate(query_tokens):
        tf_matrix[i, j] = bm25.tf[doc_idx].get(term, 0)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(tf_matrix, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(query_tokens)))
ax.set_xticklabels(query_tokens, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(top_docs)))
ax.set_yticklabels([f"Doc {d['doc_idx']}" for d in top_docs], fontsize=9)
ax.set_title(
    "Term Frequency Heatmap — Query Terms in Top-5 BM25 Docs",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Query term")
ax.set_ylabel("Top-k document")
for i in range(len(top_docs)):
    for j in range(len(query_tokens)):
        ax.text(
            j,
            i,
            f"{int(tf_matrix[i, j])}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if tf_matrix[i, j] > tf_matrix.max() / 2 else "black",
        )
fig.colorbar(im, ax=ax, label="Term frequency")
plt.tight_layout()
fname = OUTPUT_DIR / "ex4_03_bm25_tf_heatmap.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname}")

# INTERPRETATION: The heatmap shows exactly which words drive each
# document's ranking. A document that matches all query terms is a
# strong hit; one that matches a single rare term may still rank high
# because BM25's IDF weighting amplifies rare terms.


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore HDB regulation lookup
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: The Housing & Development Board (HDB) publishes hundreds of
# regulation documents covering BTO eligibility, resale procedures,
# renovation permits, and fine schedules. HDB's customer-service officers
# need to look up exact clauses when residents call.
#
# WHY BM25 SHINES HERE:
#   - "BTO", "DBSS", "HDB", "MOP", "EIP", "SPR" are domain-specific
#     acronyms that generic embedding models rarely saw during training.
#     Dense retrieval maps "MOP" to "floor" rather than "Minimum
#     Occupation Period" because the acronym is out-of-distribution.
#   - BM25 matches "MOP" exactly — if a regulation document contains
#     "MOP", BM25 ranks it first regardless of what an embedding model
#     thinks the word means.
#   - Resident queries often use the acronym verbatim: "What is my MOP
#     for a BTO flat bought in 2019?"
#
# TUNING: For regulation lookup, raise k1 to 2.0 (emphasise exact-term
# repetition — a clause that mentions "MOP" five times is more on-topic
# than one that mentions it once) and lower b to 0.5 (don't penalise
# long regulation sections that may legitimately be long).
#
# BUSINESS IMPACT: HDB handles ~1M customer-service interactions per
# year. If officers spend an average of 90 seconds per lookup and BM25
# cuts that to 15 seconds, the saving is 75 seconds * 1M = 20,833 hours
# per year. At a loaded cost of S$35/hr per officer, that's S$729K/year
# in freed-up officer time — and shorter wait times mean higher
# resident satisfaction scores (the real currency of a public agency).


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
  [x] Implemented BM25 from scratch: tokenise, IDF, tf saturation, length norm
  [x] Understood k1 (tf saturation) and b (length normalisation)
  [x] Ran BM25 on real eval queries and inspected top-k
  [x] Visualised the BM25 score distribution — a long sparse tail
  [x] Mapped BM25 to an HDB regulation lookup use case (exact acronym match)

  KEY INSIGHT: Dense retrieval answers "what is this about?"; BM25
  answers "does this contain these exact words?". Production systems
  need both — and the next exercise shows how to fuse them.

  Next: 04_hybrid_rrf.py combines dense and sparse with RRF fusion...
"""
)
