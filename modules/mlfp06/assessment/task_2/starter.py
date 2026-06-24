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

import polars as pl

from shared import MLFPDataLoader
from shared.mlfp06._ollama_bootstrap import (
    make_delegate,
    make_embedder,
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


async def _run() -> dict:
    corpus, questions = build_corpus_and_questions()

    embedder = make_embedder(model="nomic-embed-text")

    # TODO 1: Embed the corpus and the questions with embedder.embed(list_of_text).
    #         (embed is async — await it.)

    # TODO 2: For each question, rank corpus docs by _cosine() and take the
    #         top-3 indices (highest similarity first).

    # TODO 3: For each question, build a context from its top-3 docs and
    #         generate a grounded answer via make_delegate(temperature=0.0) +
    #         run_delegate_text. Instruct the model to answer ONLY from context.

    retrieved: list[list[int]] = []
    answers: list[str] = []
    return {"retrieved": retrieved, "answers": answers}


def solve() -> dict:
    """Run the RAG pipeline; return {"retrieved": [[int]], "answers": [str]}."""
    return asyncio.run(_run())


if __name__ == "__main__":
    out = solve()
    for i, (r, a) in enumerate(zip(out["retrieved"], out["answers"])):
        print(f"Q{i}: top3={r}  answer={a[:70]!r}")
