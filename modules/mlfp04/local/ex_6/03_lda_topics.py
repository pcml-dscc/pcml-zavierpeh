# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6.3: LDA — Probabilistic Topic Modelling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Fit LDA and read topic-word / document-topic distributions
#   - Measure perplexity across K to pick an elbow
#   - Find a document with mixed membership across 2+ topics
#   - Apply LDA to MAS enforcement complaint routing
#
# PREREQUISITES: Exercise 6.2 (NMF), basic probability.
# ESTIMATED TIME: ~35 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_6 import (
    OUTPUT_DIR,
    compute_npmi,
    corpus_as_lists,
    load_corpus,
    print_scenario,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD
# ════════════════════════════════════════════════════════════════════════

corpus_df = load_corpus()
documents, categories = corpus_as_lists(corpus_df)
print(f"Corpus: {len(documents):,} documents")

vectorizer = TfidfVectorizer(
    max_features=3000, stop_words="english", max_df=0.95, min_df=3
)
X = vectorizer.fit_transform(documents)
vocab = vectorizer.get_feature_names_out()


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: perplexity sweep
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  LDA Perplexity Sweep")
print("=" * 70)

k_grid = [5, 8, 10, 12, 15]
results = {}
for k in k_grid:
    # TODO: Construct a LatentDirichletAllocation with n_components=k,
    # random_state=42, max_iter=30, learning_method="online", batch_size=128.
    # Fit on X, compute perplexity(X), store model+perplexity in results[k].
    lda = ____
    lda.fit(X)
    perp = ____
    results[k] = {"model": lda, "perplexity": float(perp)}
    print(f"  K={k:>2}: perplexity = {perp:,.0f}")

K = 10
lda_model = results[K]["model"]

# TODO: Project documents to a topic distribution via lda_model.transform(X)
doc_topics = ____
dominant = doc_topics.argmax(axis=1)

topic_words: list[list[str]] = []
for t in range(K):
    top_idx = lda_model.components_[t].argsort()[-8:][::-1]
    words = [vocab[i] for i in top_idx]
    topic_words.append(words)
    print(f"  Topic {t}: {', '.join(words)}")


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert doc_topics.shape == (len(documents), K), "Task 3: document-topic shape mismatch"
row_sums = doc_topics.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-2), "Task 3: LDA rows must sum to 1"
print("\n[ok] Checkpoint 1 passed — LDA perplexity sweep + topics\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════

# TODO: Find the first document whose top topic probability is < 0.5
# and which has at least 2 topics with probability > 0.1. Store its index
# in mixed_idx (-1 if none found).
mixed_idx = ____

if mixed_idx >= 0:
    print(f"\nMixed-membership example — document #{mixed_idx}")
    print(f"Text (first 200 chars): {documents[mixed_idx][:200]}...")
    for t in np.argsort(-doc_topics[mixed_idx])[:5]:
        prob = doc_topics[mixed_idx, t]
        if prob < 0.02:
            break
        print(f"  Topic {t} ({prob:5.1%}) — {', '.join(topic_words[t][:5])}")

coherences = compute_npmi(documents, topic_words)
mean_npmi = float(np.mean(coherences))
print(f"\nLDA mean NPMI coherence: {mean_npmi:+.4f}")

viz = ModelVisualizer()
perp_data = {f"K={k}": {"perplexity": results[k]["perplexity"]} for k in k_grid}
fig_perp = viz.metric_comparison(perp_data)
fig_perp.update_layout(title="LDA Perplexity vs Number of Topics")
fig_perp.write_html(str(OUTPUT_DIR / "ex6_3_lda_perplexity.html"))


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (
    results[5]["perplexity"] != results[15]["perplexity"]
), "Task 4: perplexity must vary"
assert len(coherences) == K, "Task 4: one NPMI per topic"
print("\n[ok] Checkpoint 2 passed — mixed-membership example + visualisations\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS Enforcement Complaint Routing
# ════════════════════════════════════════════════════════════════════════

print_scenario("lda_topics")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Fit LDA across multiple K and read a perplexity curve
  [x] Found a document with mixed membership across 2+ topics
  [x] Measured NPMI coherence for LDA topics
  [x] Mapped LDA to MAS enforcement complaint routing

  Next: 04_bertopic.py — neural embeddings + UMAP + HDBSCAN + c-TF-IDF.
"""
)
