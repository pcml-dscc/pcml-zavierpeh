# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6.2: NMF Topic Modelling — Non-Negative Factorisation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Factorise a TF-IDF matrix X ≈ W @ H with NMF
#   - Read W as document-topic weights and H as topic-word weights
#   - Explain why non-negativity makes topics interpretable
#   - Measure NPMI topic coherence on a real corpus
#   - Apply NMF to SPH newsroom content tagging at scale
#
# PREREQUISITES: Exercise 6.1 (TF-IDF), linear algebra (matrix factorisation).
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — non-negativity and additive parts
#   2. Build — NMF on a TF-IDF matrix
#   3. Train — fit NMF and inspect reconstruction quality
#   4. Visualise — topic keyword bars, NPMI coherence per topic
#   5. Apply — SPH newsroom auto-tagging scenario
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
# THEORY — Non-Negative Matrix Factorisation
# ════════════════════════════════════════════════════════════════════════
# NMF decomposes the TF-IDF matrix X (n_docs x n_vocab) as:
#
#     X ≈ W @ H     with W >= 0 and H >= 0
#
#     W[doc, topic]  = document's weight for each topic
#     H[topic, word] = topic's weight for each word
#
# Compare with PCA: PCA allows NEGATIVE loadings, so a topic might be
# "+housing -technology", which is hard to read. NMF forbids negatives,
# so every topic only ADDS word mass — you can write each document as a
# literal sum of topic contributions. This is called "parts-based
# representation" and it's the reason NMF topics read like human topics.
#
# Algorithm: alternating non-negative least squares. No probabilistic
# interpretation, no Dirichlet priors — just convex optimisation with
# non-negativity constraints.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: TF-IDF -> NMF pipeline
# ════════════════════════════════════════════════════════════════════════

corpus_df = load_corpus()
documents, categories = corpus_as_lists(corpus_df)
print(f"Corpus: {len(documents):,} documents across {len(set(categories))} categories")

vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english",
    max_df=0.95,
    min_df=3,
)
X = vectorizer.fit_transform(documents)
vocab = vectorizer.get_feature_names_out()
print(f"TF-IDF matrix: {X.shape} (docs x vocab)")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit NMF with a few topic counts
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  NMF Topic Discovery")
print("=" * 70)

n_topics = 10
nmf = NMF(n_components=n_topics, random_state=42, max_iter=400, init="nndsvd")
W = nmf.fit_transform(X)
H = nmf.components_

recon_error = np.linalg.norm(X.toarray() - W @ H) / np.linalg.norm(X.toarray())
print(f"Relative reconstruction error: {recon_error:.4f}")
print(f"Lower is better — 0 = perfect reconstruction, 1 = useless")

print(f"\nTop words per topic (K={n_topics}):")
topic_words: list[list[str]] = []
for t in range(n_topics):
    top_idx = H[t].argsort()[-8:][::-1]
    words = [vocab[i] for i in top_idx]
    topic_words.append(words)
    print(f"  Topic {t}: {', '.join(words)}")

# Hard-assign each document to its top topic
doc_topic = W.argmax(axis=1)
print(f"\nDocuments per topic:")
for t in range(n_topics):
    count = int((doc_topic == t).sum())
    print(f"  Topic {t}: {count} docs")


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert W.shape == (len(documents), n_topics), "Task 3: W should be (n_docs, n_topics)"
assert H.shape == (n_topics, len(vocab)), "Task 3: H should be (n_topics, n_vocab)"
assert W.min() >= -1e-10, "Task 3: NMF W must be non-negative"
assert H.min() >= -1e-10, "Task 3: NMF H must be non-negative"
assert recon_error < 1.0, "Task 3: reconstruction should be better than zero matrix"
print("\n[ok] Checkpoint 1 passed — NMF factorisation valid and non-negative\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: topic coherence via NPMI
# ════════════════════════════════════════════════════════════════════════

coherences = compute_npmi(documents, topic_words)
mean_npmi = float(np.mean(coherences))
print(f"NPMI coherence — mean: {mean_npmi:+.4f}")
print("(Higher is better. NPMI > 0.1 = topics cohere above chance.)")
for i, c in enumerate(coherences):
    bar = "#" * max(0, int((c + 0.3) * 30))
    print(f"  Topic {i}: {c:+.4f} {bar}")

viz = ModelVisualizer()

# Coherence bar chart
coherence_data = {f"Topic_{i}": {"NPMI": float(c)} for i, c in enumerate(coherences)}
fig_coh = viz.metric_comparison(coherence_data)
fig_coh.update_layout(title="NMF Topic Coherence (NPMI)")
fig_coh.write_html(str(OUTPUT_DIR / "ex6_2_nmf_coherence.html"))

# Topic size distribution
size_data = {
    f"Topic_{t}": {"docs": int((doc_topic == t).sum())} for t in range(n_topics)
}
fig_size = viz.metric_comparison(size_data)
fig_size.update_layout(title="NMF Topic Size Distribution")
fig_size.write_html(str(OUTPUT_DIR / "ex6_2_nmf_topic_sizes.html"))

print(f"\nSaved: {OUTPUT_DIR}/ex6_2_nmf_coherence.html")
print(f"Saved: {OUTPUT_DIR}/ex6_2_nmf_topic_sizes.html")


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(coherences) == n_topics, "Task 4: one NPMI per topic"
assert mean_npmi > -0.5, f"Task 4: mean NPMI should be > -0.5, got {mean_npmi:.4f}"
print("\n[ok] Checkpoint 2 passed — NPMI computed and visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: SPH Newsroom Content Tagging
# ════════════════════════════════════════════════════════════════════════

print_scenario("nmf_topics")
print(
    """
WHY NMF IS THE RIGHT TOOL FOR SPH:
  - The editorial desk needs INTERPRETABLE topics, not black-box
    embeddings. A journalist must be able to read "Topic 3: housing,
    HDB, BTO, resale, Orchard" and immediately understand it means
    the Singapore property beat.
  - NMF is deterministic and fast (seconds on 24 hours of articles),
    so the nightly tagging job fits in the production pipeline window.
  - Non-negativity makes the topic-keyword report AUDITABLE. The
    editorial standards team can review the top-20 words per topic
    and flag any that mix semantically unrelated terms.

NUMBERS TO REMEMBER:
  - ~2,400 articles/day at SPH across ST, BT, zaobao.com
  - NMF with K=20 fits in ~6 seconds on a modern laptop
  - Ad yield on auto-tagged articles is +11% vs untagged (more
    relevant programmatic matches) = ~S$4.2M/year uplift
  - Compare to BERTopic (next exercise) which is ~8x slower but
    produces more fine-grained topics — worth it for some use cases
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
  [x] Factorised a TF-IDF matrix with NMF (X ≈ W @ H)
  [x] Read W as document-topic weights, H as topic-word weights
  [x] Explained why non-negativity makes topics interpretable
  [x] Measured NPMI topic coherence without human annotation
  [x] Mapped the technique to SPH newsroom auto-tagging

  KEY INSIGHT: NMF is the Pareto-optimal choice when you need
  interpretable topics FAST. It is not the most accurate, it is not
  the most semantic, but every topic it finds is readable, every
  run is deterministic, and every fit finishes before your coffee.

  Next: 03_lda_topics.py — probabilistic topic modelling with
  mixed-membership via Latent Dirichlet Allocation.
"""
)
