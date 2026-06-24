# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6.2: NMF Topic Modelling
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Factorise a TF-IDF matrix X ≈ W @ H with NMF
#   - Read W as document-topic weights and H as topic-word weights
#   - Measure NPMI topic coherence
#   - Apply NMF to SPH newsroom content tagging
#
# PREREQUISITES: Exercise 6.1 (TF-IDF), linear algebra.
# ESTIMATED TIME: ~30 min
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


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: TF-IDF -> NMF pipeline
# ════════════════════════════════════════════════════════════════════════

corpus_df = load_corpus()
documents, categories = corpus_as_lists(corpus_df)
print(f"Corpus: {len(documents):,} documents")

# TODO: Build a TfidfVectorizer with max_features=3000, stop_words="english",
# max_df=0.95, min_df=3. Fit_transform the documents and grab the vocab.
vectorizer = ____
X = ____
vocab = vectorizer.get_feature_names_out()
print(f"TF-IDF matrix: {X.shape}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit NMF
# ════════════════════════════════════════════════════════════════════════

n_topics = 10

# TODO: Instantiate NMF with n_components=n_topics, random_state=42,
# max_iter=400, init="nndsvd"; fit_transform X to get W; read H from
# nmf.components_
nmf = ____
W = ____
H = ____

recon_error = np.linalg.norm(X.toarray() - W @ H) / np.linalg.norm(X.toarray())
print(f"Relative reconstruction error: {recon_error:.4f}")

topic_words: list[list[str]] = []
for t in range(n_topics):
    # TODO: Extract the top 8 word indices for topic t from H[t]
    # Hint: H[t].argsort()[-8:][::-1]
    top_idx = ____
    words = [vocab[i] for i in top_idx]
    topic_words.append(words)
    print(f"  Topic {t}: {', '.join(words)}")

doc_topic = W.argmax(axis=1)


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert W.shape == (len(documents), n_topics), "Task 3: W shape mismatch"
assert H.shape == (n_topics, len(vocab)), "Task 3: H shape mismatch"
assert W.min() >= -1e-10 and H.min() >= -1e-10, "Task 3: NMF must be non-negative"
print("\n[ok] Checkpoint 1 passed — NMF factorisation valid\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: NPMI coherence
# ════════════════════════════════════════════════════════════════════════

# TODO: Call compute_npmi(documents, topic_words) and compute the mean
coherences = ____
mean_npmi = float(np.mean(coherences))
print(f"NMF mean NPMI: {mean_npmi:+.4f}")

viz = ModelVisualizer()
coherence_data = {f"Topic_{i}": {"NPMI": float(c)} for i, c in enumerate(coherences)}
fig_coh = viz.metric_comparison(coherence_data)
fig_coh.update_layout(title="NMF Topic Coherence (NPMI)")
fig_coh.write_html(str(OUTPUT_DIR / "ex6_2_nmf_coherence.html"))


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(coherences) == n_topics, "Task 4: one NPMI per topic"
assert mean_npmi > -0.5, f"Task 4: mean NPMI should be > -0.5, got {mean_npmi:.4f}"
print("\n[ok] Checkpoint 2 passed — NPMI computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: SPH Newsroom Content Tagging
# ════════════════════════════════════════════════════════════════════════

print_scenario("nmf_topics")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Factorised a TF-IDF matrix with NMF (X ≈ W @ H)
  [x] Read W/H as document-topic / topic-word weights
  [x] Measured NPMI coherence
  [x] Mapped NMF to SPH newsroom auto-tagging

  Next: 03_lda_topics.py — probabilistic topic modelling with LDA.
"""
)
