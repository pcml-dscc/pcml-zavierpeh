# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6.4: BERTopic — Neural Topic Modelling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a BERTopic pipeline: sentence embeddings -> UMAP -> HDBSCAN
#     -> c-TF-IDF topic extraction
#   - Compare NPMI coherence against NMF/LDA
#   - Fall back to NMF when sentence-transformers is not installed
#   - Apply BERTopic to Grab multilingual support-ticket clustering
#
# PREREQUISITES: Ex 6.2 (NMF), Ex 6.3 (LDA), Ex 3 (UMAP), Ex 1 (HDBSCAN).
# ESTIMATED TIME: ~35 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_6 import (
    OUTPUT_DIR,
    compute_npmi,
    corpus_as_lists,
    load_corpus,
    print_scenario,
)

try:
    from bertopic import BERTopic  # type: ignore[import-untyped]
except ImportError:
    BERTopic = None


# ════════════════════════════════════════════════════════════════════════
# TASK 2 + 3 — BUILD and TRAIN
# ════════════════════════════════════════════════════════════════════════

corpus_df = load_corpus()
documents, categories = corpus_as_lists(corpus_df)
print(f"Corpus: {len(documents):,} documents")

if BERTopic is not None:
    print("\nBERTopic pipeline (embeddings -> UMAP -> HDBSCAN -> c-TF-IDF)")

    # TODO: Construct a BERTopic model with embedding_model="all-MiniLM-L6-v2",
    # min_topic_size=15, nr_topics="auto", verbose=False.
    topic_model = ____

    # TODO: Call fit_transform(documents) and unpack into (topics, probs).
    topics, probs = ____

    topic_info = topic_model.get_topic_info()
    n_topics = int((topic_info["Topic"] >= 0).sum())

    topic_words: list[list[str]] = []
    for topic_id in range(min(n_topics, 10)):
        words = [w for w, _ in topic_model.get_topic(topic_id)[:10]]
        topic_words.append(words)

    print(f"\nTopics discovered: {n_topics}")
    for _, row in topic_info[topic_info["Topic"] >= 0].head(10).iterrows():
        print(f"  Topic {row['Topic']}: {str(row['Name'])[:60]} (n={row['Count']})")

    method_label = "BERTopic"
    doc_topics_vec = np.asarray(topics)
else:
    # Fallback: NMF-over-TF-IDF approximation with a loud, actionable message
    print("\n" + "!" * 70)
    print(
        "  bertopic / sentence-transformers not installed — run "
        "`uv sync --extra topic-models` for the real pipeline"
    )
    print("  Running NMF-over-TF-IDF approximation for this session")
    print("!" * 70 + "\n")

    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", max_df=0.95, min_df=5
    )
    tfidf = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    n_topics = 10
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=400, init="nndsvd")
    W = nmf.fit_transform(tfidf)
    H = nmf.components_
    doc_topics_vec = W.argmax(axis=1)

    topic_words = []
    for t in range(n_topics):
        top_idx = H[t].argsort()[-10:][::-1]
        words = [feature_names[i] for i in top_idx]
        topic_words.append(words)
        count = int((doc_topics_vec == t).sum())
        print(f"  Topic {t}: {', '.join(words[:8])} (n={count})")

    method_label = "NMF-fallback"


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(topic_words) > 0, "Task 3: should discover at least one topic"
assert all(len(w) >= 5 for w in topic_words), "Task 3: each topic needs 5+ words"
print(
    f"\n[ok] Checkpoint 1 passed — {method_label} produced {len(topic_words)} topics\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════

# TODO: compute NPMI coherence for topic_words over the documents
coherences = ____
mean_npmi = float(np.mean(coherences))
print(f"{method_label} mean NPMI coherence: {mean_npmi:+.4f}")

viz = ModelVisualizer()
coherence_data = {f"Topic_{i}": {"NPMI": float(c)} for i, c in enumerate(coherences)}
fig_coh = viz.metric_comparison(coherence_data)
fig_coh.update_layout(title=f"{method_label} Topic Coherence (NPMI)")
fig_coh.write_html(str(OUTPUT_DIR / "ex6_4_bertopic_coherence.html"))


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(coherences) == len(topic_words), "Task 4: one NPMI per topic"
assert mean_npmi > -0.5, f"Task 4: mean NPMI > -0.5, got {mean_npmi:.4f}"
print("\n[ok] Checkpoint 2 passed — coherence computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Grab Multilingual Ticket Clustering
# ════════════════════════════════════════════════════════════════════════

print_scenario("bertopic")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Assembled BERTopic's embed/reduce/cluster/describe pipeline
  [x] Handled the optional extra with a loud, actionable fallback
  [x] Measured NPMI coherence and compared it to NMF/LDA
  [x] Mapped BERTopic to Grab's multilingual routing problem

  Next: 05_sentiment_word2vec.py — word embeddings as features.
"""
)
