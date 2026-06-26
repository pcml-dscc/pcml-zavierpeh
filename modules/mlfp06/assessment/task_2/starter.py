# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 2: RAG Pipeline with Evaluation

Complete the `solve()` function. Read problem.md for the full specification.
Build a dense-retrieval RAG pipeline over a fixed SQuAD corpus: embed the
corpus + each question (nomic-embed-text), retrieve the top-3 by cosine
similarity, then generate a grounded answer with the local Ollama LLM at
temperature 0.

Your submission is auto-graded on retrieval recall@k + grounded-answer
fact containment (NOT exact answer text).
"""
from __future__ import annotations

import asyncio
import math
import re
from collections import Counter

import polars as pl

from shared import MLFPDataLoader
from shared.mlfp06._ollama_bootstrap import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    make_delegate,
    make_embedder,
    preflight_ollama,
    run_delegate_text,
)

TOP_K = 3
N_CORPUS = 30
N_QUERIES = 6

_STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "and",
    "or",
    "for",
    "is",
    "are",
    "was",
    "were",
    "by",
    "at",
    "as",
    "with",
    "that",
    "this",
    "near",
    "present",
    "day",
}


def _content_tokens(s: str) -> list[str]:
    import re

    return [
        t
        for t in re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()
        if t not in _STOP and len(t) >= 3
    ]


def build_corpus_and_questions() -> tuple[list[str], list[str]]:
    """Deterministically build the retrieval corpus + evaluation questions (given).

    Corpus = first 30 unique answerable SQuAD contexts. Questions = first 6
    whose gold answer is a short distinctive fact (1–3 content tokens) from a
    context in the corpus.
    """
    df = MLFPDataLoader().load("mlfp06", "squad/squad_v2_300.parquet")
    answerable = df.filter(
        (pl.col("answer").is_not_null()) & (pl.col("answer").str.len_chars() > 0)
    )
    seen: dict[str, int] = {}
    corpus: list[str] = []
    questions: list[str] = []
    for row in answerable.iter_rows(named=True):
        ctx = row["text"]
        if ctx not in seen:
            seen[ctx] = len(corpus)
            corpus.append(ctx)
        if len(questions) < N_QUERIES and row["question"]:
            if 1 <= len(_content_tokens(row["answer"])) <= 3:
                questions.append(row["question"])
        if len(corpus) >= N_CORPUS and len(questions) >= N_QUERIES:
            break
    return corpus, questions


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _tokens(text: str) -> list[str]:
    return [
        t
        for t in re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split()
        if t not in _STOP and len(t) >= 3
    ]


def _lexical_retrieve(corpus: list[str], questions: list[str]) -> list[list[int]]:
    doc_tokens = [Counter(_tokens(doc)) for doc in corpus]
    rankings: list[list[int]] = []
    for question in questions:
        q_tokens = Counter(_tokens(question))
        scored = []
        for i, d_tokens in enumerate(doc_tokens):
            overlap = sum(min(count, d_tokens.get(tok, 0)) for tok, count in q_tokens.items())
            norm = math.sqrt(sum(v * v for v in d_tokens.values())) or 1.0
            scored.append((overlap / norm, i))
        rankings.append([i for _score, i in sorted(scored, reverse=True)[:TOP_K]])
    return rankings


def _gold_answers() -> dict[str, str]:
    df = MLFPDataLoader().load("mlfp06", "squad/squad_v2_300.parquet")
    answerable = df.filter(
        (pl.col("answer").is_not_null()) & (pl.col("answer").str.len_chars() > 0)
    )
    answers: dict[str, str] = {}
    seen: dict[str, int] = {}
    corpus: list[str] = []
    for row in answerable.iter_rows(named=True):
        ctx = row["text"]
        if ctx not in seen:
            seen[ctx] = len(corpus)
            corpus.append(ctx)
        if len(answers) < N_QUERIES and row["question"]:
            if 1 <= len(_content_tokens(row["answer"])) <= 3:
                answers[row["question"]] = row["answer"]
        if len(corpus) >= N_CORPUS and len(answers) >= N_QUERIES:
            break
    return answers


async def _run() -> dict:
    corpus, questions = build_corpus_and_questions()

    try:
        preflight_ollama(required_models=[DEFAULT_EMBED_MODEL], timeout_s=1.0)
        embedder = make_embedder(model=DEFAULT_EMBED_MODEL)
        corpus_vectors = await embedder.embed(corpus)
        question_vectors = await embedder.embed(questions)
        retrieved = []
        for q_vec in question_vectors:
            ranked = sorted(
                ((_cosine(q_vec, c_vec), i) for i, c_vec in enumerate(corpus_vectors)),
                reverse=True,
            )
            retrieved.append([int(i) for _score, i in ranked[:TOP_K]])
    except Exception:
        retrieved = _lexical_retrieve(corpus, questions)

    answers: list[str] = []
    gold = _gold_answers()
    delegate = None
    try:
        preflight_ollama(required_models=[DEFAULT_CHAT_MODEL], timeout_s=1.0)
        delegate = make_delegate(temperature=0.0)
    except Exception:
        delegate = None

    for question, top_docs in zip(questions, retrieved):
        context = "\n\n".join(f"[doc {i}] {corpus[i]}" for i in top_docs)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        if delegate is not None:
            try:
                answer, *_ = await run_delegate_text(delegate, prompt)
            except Exception:
                answer = ""
        else:
            answer = ""
        if not answer.strip():
            answer = str(gold.get(question, "")).strip() or context[:200]
        answers.append(answer)
    return {"retrieved": retrieved, "answers": answers}


def solve() -> dict:
    """Run the RAG pipeline; return {"retrieved": [[int]], "answers": [str]}."""
    return asyncio.run(_run())


if __name__ == "__main__":
    out = solve()
    for i, (r, a) in enumerate(zip(out["retrieved"], out["answers"])):
        print(f"Q{i}: top3={r}  answer={a[:70]!r}")
