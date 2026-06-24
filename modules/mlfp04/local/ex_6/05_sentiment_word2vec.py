# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6.5: Word2Vec Features + Sentiment Analysis
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain Word2Vec as a shallow prediction network whose hidden
#     layer weights become the word embeddings
#   - Average word vectors into a document vector
#   - Train a sentiment classifier on document vectors
#   - Apply the technique to DBS Bank multilingual review triage
#
# PREREQUISITES: Ex 6.1 (TF-IDF), basic classification.
# ESTIMATED TIME: ~35 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_6 import (
    NEGATIVE_WORDS,
    OUTPUT_DIR,
    POSITIVE_WORDS,
    corpus_as_lists,
    lexicon_sentiment,
    load_corpus,
    print_scenario,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: document vectors + lexicon baseline
# ════════════════════════════════════════════════════════════════════════

corpus_df = load_corpus()
documents, categories = corpus_as_lists(corpus_df)
print(f"Corpus: {len(documents):,} documents")

# TODO: Score every document with the shared lexicon_sentiment() helper
lex_scores = ____

print("\nLexicon sentiment baseline:")
print(f"  Positive (> 0.3): {(lex_scores > 0.3).mean():.1%}")
print(f"  Negative (< -0.3):{(lex_scores < -0.3).mean():.1%}")
print(f"  Mean sentiment:   {lex_scores.mean():+.4f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: pseudo-Word2Vec + logistic regression
# ════════════════════════════════════════════════════════════════════════

EMBED_DIM = 64
vocab_cache: dict[str, np.ndarray] = {}


def pseudo_word2vec(token: str) -> np.ndarray:
    """Deterministic pseudo-Word2Vec — stands in for a real pretrained
    embedding table so the exercise stays self-contained."""
    if token in vocab_cache:
        return vocab_cache[token]
    seed = abs(hash(token)) % (2**32)
    vec = np.random.default_rng(seed).standard_normal(EMBED_DIM).astype(np.float64)
    vocab_cache[token] = vec
    return vec


def document_vector(text: str) -> np.ndarray:
    """Return the mean of the pseudo-Word2Vec vectors of the tokens in ``text``."""
    # TODO: split text on whitespace, keep only alphabetic tokens, look each
    # up with pseudo_word2vec(), and return the mean along axis 0. Return a
    # zero vector when there are no tokens.
    ____


doc_vectors = np.stack([document_vector(doc) for doc in documents])
labels = (lex_scores > 0).astype(int)

split = int(0.8 * len(doc_vectors))
X_train, X_test = doc_vectors[:split], doc_vectors[split:]
y_train, y_test = labels[:split], labels[split:]

# TODO: Fit a LogisticRegression(max_iter=500, random_state=42) on (X_train,
# y_train) and score it on both train and test splits.
clf = ____
clf.fit(X_train, y_train)
train_acc = ____
test_acc = ____

print(f"\nLogistic regression on document vectors:")
print(f"  Train accuracy: {train_acc:.3f}")
print(f"  Test  accuracy: {test_acc:.3f}")


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert doc_vectors.shape == (
    len(documents),
    EMBED_DIM,
), "Task 3: document vector shape mismatch"
assert 0.0 <= test_acc <= 1.0, "Task 3: test accuracy must be in [0, 1]"
assert len(POSITIVE_WORDS) > 5 and len(NEGATIVE_WORDS) > 5, "Task 3: lexicons non-empty"
print("\n[ok] Checkpoint 1 passed — Word2Vec features + classifier\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: sentiment by category
# ════════════════════════════════════════════════════════════════════════

news_with_sentiment = corpus_df.with_columns(pl.Series("sentiment", lex_scores))

cat_sentiment: dict[str, float] = {}
print("\nMean lexicon sentiment by category:")
for cat in sorted(news_with_sentiment["category"].unique().to_list()):
    mean_s = float(
        news_with_sentiment.filter(pl.col("category") == cat)["sentiment"].mean()
    )
    cat_sentiment[cat] = mean_s
    print(f"  {cat:<22} {mean_s:+.4f}")

viz = ModelVisualizer()
cat_data = {cat: {"mean_sentiment": s} for cat, s in cat_sentiment.items()}
fig_cat = viz.metric_comparison(cat_data)
fig_cat.update_layout(title="Mean Sentiment by Category")
fig_cat.write_html(str(OUTPUT_DIR / "ex6_5_sentiment_by_category.html"))


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (
    len(cat_sentiment) > 0
), "Task 4: must compute sentiment for at least one category"
assert all(
    -1.0 <= s <= 1.0 for s in cat_sentiment.values()
), "Task 4: sentiment in [-1, 1]"
print("\n[ok] Checkpoint 2 passed — visualisations written\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Bank Multilingual Review Triage
# ════════════════════════════════════════════════════════════════════════

print_scenario("sentiment_word2vec")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Explained Word2Vec as a shallow network whose hidden-layer
      weights become the embeddings
  [x] Averaged word vectors into a document vector
  [x] Trained a sentiment classifier on document vectors
  [x] Mapped the technique to DBS Bank multilingual review triage

  EXERCISE 6 COMPLETE. Five text tools: TF-IDF/BM25, NMF, LDA,
  BERTopic, Word2Vec + LR. Next: Exercise 7 — matrix factorisation
  for recommender systems.
"""
)
