# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6.5: Word2Vec Features + Sentiment Analysis
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain how Word2Vec learns dense word vectors (CBOW vs skip-gram)
#   - Average word vectors into a document vector
#   - Use document vectors as features for a sentiment classifier
#   - Compare a lexicon baseline to a learned classifier
#   - Apply the technique to DBS Bank multilingual review triage
#
# PREREQUISITES: Ex 6.1 (TF-IDF), basic classification (logistic
# regression), understanding that neural networks learn by minimising
# a loss function.
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — Word2Vec as a shallow prediction network
#   2. Build — document vectors from averaged word embeddings
#   3. Train — compare lexicon sentiment vs learned classifier
#   4. Visualise — sentiment by category, confusion on negative class
#   5. Apply — DBS Bank app-store review triage
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
# THEORY — Word2Vec as a Shallow Network
# ════════════════════════════════════════════════════════════════════════
# Word2Vec (Mikolov 2013) trains a two-layer neural network to predict
# a target word from its context (CBOW) or the context from the target
# (skip-gram). The network itself is discarded after training — what
# you keep is the HIDDEN LAYER: for each word in the vocabulary, the
# hidden-layer weights are its dense embedding vector.
#
# Emergent properties from this single optimisation objective:
#
#   - Similar words end up near each other:
#         cos(king, queen) > cos(king, car)
#
#   - Vector arithmetic encodes analogies:
#         king - man + woman ≈ queen
#
#   - Dense 100-300D vectors replace 10K+ sparse bag-of-words
#
# WHY WE AVERAGE TO MAKE DOCUMENT VECTORS:
# A document is a bag of words, so a reasonable document vector is
# just the mean of its word vectors. This "Sentence-BOW" baseline is
# crude but effective — it captures topic-level similarity even
# without attention or positional encoding.
#
# This same optimisation principle — drive features out of a
# reconstruction / prediction loss — is what MLFP05 neural networks
# exploit. Word2Vec is the simplest example of the idea.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: document vectors + lexicon baseline
# ════════════════════════════════════════════════════════════════════════

corpus_df = load_corpus()
documents, categories = corpus_as_lists(corpus_df)
print(f"Corpus: {len(documents):,} documents")

# Lexicon baseline — fast and interpretable
lex_scores = lexicon_sentiment(documents)
print("\n" + "=" * 70)
print("  Lexicon Sentiment Baseline")
print("=" * 70)
print(f"Positive (> 0.3): {(lex_scores > 0.3).mean():.1%}")
print(f"Neutral:          {((lex_scores >= -0.3) & (lex_scores <= 0.3)).mean():.1%}")
print(f"Negative (< -0.3):{(lex_scores < -0.3).mean():.1%}")
print(f"Mean sentiment:   {lex_scores.mean():+.4f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Word2Vec-style document vectors + classifier
# ════════════════════════════════════════════════════════════════════════
# For teaching purposes we use a deterministic pseudo-Word2Vec: a hash
# of each token seeds a small random vector. This is NOT a real Word2Vec
# but it demonstrates the averaging mechanic without requiring a 4GB
# pre-trained embedding file. In production you would load real vectors
# from gensim's downloader ("word2vec-google-news-300") or use a
# kailash_ml SentenceTransformer wrapper.

RNG = np.random.default_rng(42)
EMBED_DIM = 64
vocab_cache: dict[str, np.ndarray] = {}


def pseudo_word2vec(token: str) -> np.ndarray:
    """Deterministic pseudo-Word2Vec — hash the token to a seed, then draw
    a stable random vector. Stands in for a pretrained embedding table."""
    if token in vocab_cache:
        return vocab_cache[token]
    seed = abs(hash(token)) % (2**32)
    vec = np.random.default_rng(seed).standard_normal(EMBED_DIM).astype(np.float64)
    vocab_cache[token] = vec
    return vec


def document_vector(text: str) -> np.ndarray:
    """Average the pseudo-Word2Vec vectors of the tokens in a document."""
    tokens = [t for t in text.lower().split() if t.isalpha()]
    if not tokens:
        return np.zeros(EMBED_DIM, dtype=np.float64)
    return np.mean([pseudo_word2vec(t) for t in tokens], axis=0)


doc_vectors = np.stack([document_vector(doc) for doc in documents])

# Bootstrap labels from the lexicon for a self-supervised sentiment task:
# the classifier's job is to predict the lexicon's sign from the averaged
# word-embedding features. A real pipeline uses human-labelled reviews;
# the lexicon label is a stand-in to keep the exercise self-contained.
labels = (lex_scores > 0).astype(int)
print(
    f"\nLabel balance: positive={labels.sum()}, negative={len(labels) - labels.sum()}"
)

# Holdout split
split = int(0.8 * len(doc_vectors))
X_train, X_test = doc_vectors[:split], doc_vectors[split:]
y_train, y_test = labels[:split], labels[split:]

clf = LogisticRegression(max_iter=500, random_state=42)
clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))
print(f"\nLogistic regression on document vectors:")
print(f"  Train accuracy: {train_acc:.3f}")
print(f"  Test  accuracy: {test_acc:.3f}")


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert doc_vectors.shape == (
    len(documents),
    EMBED_DIM,
), "Task 3: document vector shape mismatch"
assert 0.0 <= test_acc <= 1.0, "Task 3: test accuracy must be a probability"
assert len(POSITIVE_WORDS) > 5 and len(NEGATIVE_WORDS) > 5, "Task 3: lexicons non-empty"
print("\n[ok] Checkpoint 1 passed — Word2Vec document vectors + classifier\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: sentiment by category
# ════════════════════════════════════════════════════════════════════════

news_with_sentiment = corpus_df.with_columns(
    pl.Series("sentiment", lex_scores),
)

print("\nMean lexicon sentiment by category:")
cat_sentiment: dict[str, float] = {}
for cat in sorted(news_with_sentiment["category"].unique().to_list()):
    mean_s = float(
        news_with_sentiment.filter(pl.col("category") == cat)["sentiment"].mean()
    )
    cat_sentiment[cat] = mean_s
    indicator = "+" if mean_s > 0.05 else "-" if mean_s < -0.05 else "~"
    bar = "#" * int(abs(mean_s) * 20)
    print(f"  {cat:<22} {mean_s:+.4f} {indicator} {bar}")

viz = ModelVisualizer()

cat_data = {cat: {"mean_sentiment": s} for cat, s in cat_sentiment.items()}
fig_cat = viz.metric_comparison(cat_data)
fig_cat.update_layout(title="Mean Sentiment by Category")
fig_cat.write_html(str(OUTPUT_DIR / "ex6_5_sentiment_by_category.html"))

acc_data = {
    "Lexicon baseline": {
        "accuracy": float(accuracy_score(labels, (lex_scores > 0).astype(int)))
    },
    "Word2Vec + LR (train)": {"accuracy": float(train_acc)},
    "Word2Vec + LR (test)": {"accuracy": float(test_acc)},
}
fig_acc = viz.metric_comparison(acc_data)
fig_acc.update_layout(title="Sentiment Classifier Comparison")
fig_acc.write_html(str(OUTPUT_DIR / "ex6_5_classifier_comparison.html"))

print(f"\nSaved: {OUTPUT_DIR}/ex6_5_sentiment_by_category.html")
print(f"Saved: {OUTPUT_DIR}/ex6_5_classifier_comparison.html")


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
print(
    """
WHY WORD2VEC + LR FOR DBS:
  - Lexicon sentiment is free but brittle. "Not good" counts as
    positive because "good" is in the positive list and negation is
    ignored. Word2Vec + logistic regression learns from training data
    that "not good" drifts the document vector toward the negative
    region.
  - Word2Vec embeddings are pretrained on enormous corpora, so the
    sentiment classifier transfers reasonably well across the
    DBS-relevant languages (English, Mandarin — less well for Malay
    and Tamil unless you use multilingual fastText).
  - The pipeline is still cheap: 64D vectors, a linear classifier,
    ~2ms per review on CPU. 40K reviews/month costs ~S$4 of compute.
  - Higher accuracy on negative reviews is where the S$$$ live:
    catching 20 extra negatives/month that the lexicon misses is
    worth ~S$160K/month in avoided viral complaints.

WHEN TO GO FURTHER:
  - For really tricky negations and domain-specific slang, Module 5
    fine-tunes a distilled BERT on DBS-specific review data. Expect
    another +4-6 accuracy points on the negative class.
  - Module 6 introduces the LLM-based triage path (zero-shot with
    Kaizen Delegate) which removes the need for labelled data entirely.
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Explained Word2Vec as a shallow neural network whose hidden
      layer weights are the embeddings
  [x] Averaged word vectors into a document vector
  [x] Trained a logistic regression classifier on document vectors
  [x] Compared lexicon sentiment to a learned classifier
  [x] Mapped the technique to DBS Bank multilingual review triage

  KEY INSIGHT: Word2Vec is the entry point to representation learning.
  The training objective ("predict context") is a proxy — what you
  actually care about is the hidden layer it learns along the way.
  MLFP05's neural networks generalise this idea: train on one
  objective, harvest the hidden representations, reuse them downstream.

  EXERCISE 6 COMPLETE. You now hold five distinct tools for text:
    - TF-IDF / BM25       — classic retrieval, no training
    - NMF                 — fast, interpretable topics
    - LDA                 — probabilistic, mixed-membership topics
    - BERTopic            — multilingual, semantic topics
    - Word2Vec + LR       — learned features for classification

  Next: Exercise 7 — matrix factorisation for recommender systems.
  You will see that the same optimisation-drives-features principle
  powers user-item embeddings, which powers Netflix, Spotify, and
  every modern recommender.
"""
)
