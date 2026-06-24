# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 6 — NLP & Topic Modelling.

Contains: corpus loading, TF-IDF/CountVectorizer construction, NPMI
coherence scoring, sentiment lexicons, Singapore/APAC text-analytics
scenario helpers, and plot output directory management.

Technique-specific code (TF-IDF/BM25 scoring, NMF decomposition, LDA
fitting, BERTopic pipeline, Word2Vec sentiment classifier) does NOT
belong here — it lives in the per-technique files under
modules/mlfp04/solutions/ex_6/.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

OUTPUT_DIR = Path("outputs") / "ex6_nlp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# TOY CORPUS — small Singapore-flavoured corpus for teaching derivations
# ════════════════════════════════════════════════════════════════════════

TOY_CORPUS: list[str] = [
    "Singapore economy grew strongly in 2024",
    "Singapore property market shows resilience",
    "MAS tightens monetary policy amid global uncertainty",
    "Property developers report strong demand",
    "Singapore government announces new housing measures",
    "Global markets react to central bank decisions",
    "Technology sector leads Singapore stock exchange",
    "Housing prices continue upward trend in Singapore",
]


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore/APAC document corpus
# ════════════════════════════════════════════════════════════════════════


def load_corpus(min_chars: int = 100) -> pl.DataFrame:
    """Load the MLFP document corpus (title + content + category).

    Returns a polars DataFrame with a derived 'text' column that concatenates
    title and content, filtered to rows with content longer than ``min_chars``.
    Categories include singapore_economy, singapore_housing, asean_economy,
    ml_fundamentals, and several other topical clusters — well-suited to
    unsupervised topic discovery.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp03", "documents.parquet")
    return df.with_columns(
        (pl.col("title") + ". " + pl.col("content")).alias("text"),
    ).filter(pl.col("content").str.len_chars() > min_chars)


def corpus_as_lists(
    df: pl.DataFrame,
) -> tuple[list[str], list[str]]:
    """Split a corpus frame into parallel (documents, categories) lists."""
    return df["text"].to_list(), df["category"].to_list()


# ════════════════════════════════════════════════════════════════════════
# NPMI TOPIC COHERENCE
# ════════════════════════════════════════════════════════════════════════


def compute_npmi(
    documents: list[str],
    topic_words: list[list[str]],
    max_docs: int = 3000,
) -> list[float]:
    """Compute Normalised Pointwise Mutual Information coherence per topic.

    NPMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i) P(w_j))) / (-log P(w_i, w_j))

    Range is [-1, 1]. Higher = more coherent topic. We approximate
    probabilities from a document-level co-occurrence count over the first
    ``max_docs`` documents (enough for a stable signal).
    """
    docs = documents[:max_docs]
    word_doc_count: Counter[str] = Counter()
    pair_doc_count: Counter[tuple[str, str]] = Counter()
    n_docs = len(docs)

    for doc in docs:
        words = set(doc.lower().split())
        for w in words:
            word_doc_count[w] += 1
        word_list = list(words)
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                pair = tuple(sorted([word_list[i], word_list[j]]))
                pair_doc_count[pair] += 1

    coherences: list[float] = []
    for topic in topic_words:
        npmi_sum = 0.0
        n_pairs = 0
        for i in range(len(topic)):
            for j in range(i + 1, len(topic)):
                w_i, w_j = topic[i].lower(), topic[j].lower()
                pair = tuple(sorted([w_i, w_j]))
                p_i = word_doc_count.get(w_i, 0) / max(n_docs, 1)
                p_j = word_doc_count.get(w_j, 0) / max(n_docs, 1)
                p_ij = pair_doc_count.get(pair, 0) / max(n_docs, 1)
                if p_ij > 0 and p_i > 0 and p_j > 0:
                    pmi = np.log(p_ij / (p_i * p_j))
                    npmi = pmi / (-np.log(p_ij))
                    npmi_sum += npmi
                    n_pairs += 1
        coherences.append(npmi_sum / max(n_pairs, 1))
    return coherences


# ════════════════════════════════════════════════════════════════════════
# SENTIMENT LEXICONS — baseline for review triage
# ════════════════════════════════════════════════════════════════════════

POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        "good",
        "great",
        "excellent",
        "best",
        "strong",
        "growth",
        "resilience",
        "success",
        "positive",
        "improved",
        "innovative",
        "robust",
        "outperformed",
        "gains",
        "upbeat",
    }
)

NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "bad",
        "poor",
        "decline",
        "fall",
        "weak",
        "crisis",
        "risk",
        "loss",
        "negative",
        "failed",
        "uncertainty",
        "downturn",
        "slump",
        "recession",
        "worry",
    }
)


def lexicon_sentiment(docs: Iterable[str]) -> np.ndarray:
    """Score each document in ``docs`` as a scalar in [-1, 1] via a naive lexicon.

    (pos - neg) / (pos + neg), zero when neither list matches.
    """
    scores: list[float] = []
    for doc in docs:
        words = set(doc.lower().split())
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        total = pos + neg
        scores.append((pos - neg) / total if total > 0 else 0.0)
    return np.asarray(scores, dtype=np.float64)


# ════════════════════════════════════════════════════════════════════════
# SCENARIO HELPERS — Singapore / APAC text analytics context
# ════════════════════════════════════════════════════════════════════════

SCENARIOS: dict[str, str] = {
    "tfidf_bm25": (
        "CASE: ST Engineering internal document search. 180K engineering "
        "reports across aerospace, marine, and electronics business units. "
        "TF-IDF vs BM25 determines which report a manager sees first when "
        "searching 'turbine blade fatigue.' A missed report costs ~S$450K "
        "in duplicated R&D work. BM25's length normalisation (b=0.75) "
        "ensures short executive summaries compete fairly with 80-page "
        "technical manuals, improving first-hit accuracy by ~14%."
    ),
    "nmf_topics": (
        "CASE: Singapore Press Holdings (SPH) newsroom content tagging. "
        "~2,400 articles/day across The Straits Times, Business Times, "
        "zaobao.com. NMF runs nightly on the 24-hour TF-IDF matrix and "
        "produces 20 interpretable topics (housing, MAS monetary policy, "
        "SEA Games, etc.) for the recommendation engine. Non-negativity "
        "means the topic-keyword report is directly auditable by the "
        "editorial desk. Ad yield on tagged articles is +11% vs untagged."
    ),
    "lda_topics": (
        "CASE: Monetary Authority of Singapore (MAS) enforcement scanning. "
        "~60K complaint emails/year across retail banking, insurance, "
        "fintech. LDA's mixed-membership model matters — a single complaint "
        "about a crypto rug pull mentioning SGD withdrawals touches BOTH "
        "a 'digital asset fraud' topic AND a 'cross-border payments' topic. "
        "Soft topic assignments route the complaint to both enforcement "
        "teams, cutting median resolution time from 18 days to 11."
    ),
    "bertopic": (
        "CASE: Grab customer-support ticket clustering across Singapore, "
        "Indonesia, Thailand, Vietnam. ~35K tickets/week, mixed English + "
        "Bahasa Indonesia + Thai + Vietnamese. BERTopic's multilingual "
        "sentence-transformer embeddings discover topics that TF-IDF "
        "misses ('driver helmet policy' clusters across languages because "
        "the embeddings are language-agnostic). NPMI coherence of 0.18 vs "
        "LDA's 0.08; each well-clustered ticket saves ~S$3.20 in routing."
    ),
    "sentiment_word2vec": (
        "CASE: DBS Bank app-store review triage. 40K reviews/month across "
        "iOS, Google Play, and Huawei AppGallery in English, Mandarin, "
        "Malay, Tamil. A Word2Vec + logistic-regression sentiment classifier "
        "trained on labelled English reviews transfers to the other three "
        "languages via shared subword tokens. Each negative review routed "
        "to CX within 10 min prevents ~S$8K of avoided viral complaints — "
        "catching 20 extra/month is S$160K vs S$120 in compute."
    ),
}


def print_scenario(name: str) -> None:
    """Print a named Singapore/APAC scenario block for the APPLY phase."""
    body = SCENARIOS.get(name, "")
    if not body:
        return
    print("\n" + "=" * 70)
    print(f"  APPLY — {name}")
    print("=" * 70)
    print(body)
    print("=" * 70 + "\n")
