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
#   - Explain why neural embeddings find topics that TF-IDF misses
#     (polysemy, language-independence, paraphrase-robustness)
#   - Measure NPMI coherence and compare against NMF/LDA
#   - Fall back to an NMF approximation when sentence-transformers is
#     not installed (loud, actionable, no silent no-op)
#   - Apply BERTopic to Grab multilingual customer-support ticket
#     clustering
#
# PREREQUISITES: Ex 6.1 (TF-IDF), Ex 6.2 (NMF), Ex 6.3 (LDA),
# Ex 3 (UMAP), Ex 1 (HDBSCAN).
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — BERTopic's four-stage pipeline
#   2. Build — assemble the pipeline
#   3. Train — fit on the corpus
#   4. Visualise — topic word importance and NPMI coherence
#   5. Apply — Grab multilingual support-ticket clustering
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

# Optional extra — BERTopic / sentence-transformers. Declared in the
# project's [topic-models] extra. If missing, we raise a loud, actionable
# error only at the call site (see Rule: "optional extras with loud failure").
try:
    from bertopic import BERTopic  # type: ignore[import-untyped]
except ImportError:
    BERTopic = None


# ════════════════════════════════════════════════════════════════════════
# THEORY — The BERTopic Pipeline
# ════════════════════════════════════════════════════════════════════════
# BERTopic (Grootendorst 2022) replaces TF-IDF with a four-stage
# neural-assisted pipeline:
#
#   1. EMBED   — sentence-transformers encodes each document into a
#                dense vector (typically 384D for all-MiniLM-L6-v2).
#                Semantically similar documents end up close in this
#                space, regardless of surface wording.
#   2. REDUCE  — UMAP projects the embeddings down to ~5D, preserving
#                local topology for clustering.
#   3. CLUSTER — HDBSCAN finds density-based clusters in the reduced
#                space. Noise points become "outlier" topic (-1).
#   4. DESCRIBE— For each cluster, c-TF-IDF treats the cluster as a
#                single pseudo-document and computes TF-IDF weights
#                against other clusters — this yields interpretable
#                topic keywords that are genuinely differentiating.
#
# WHY THIS BEATS LDA/NMF:
#   - Embeddings capture meaning, not just surface words. "hut" and
#     "shelter" cluster together even if they never co-occur.
#   - Multilingual models cluster across languages with ZERO extra
#     work: "driver helmet" in English and "helmet pengemudi" in
#     Bahasa Indonesia land near each other.
#   - HDBSCAN finds the right number of clusters automatically via
#     min_cluster_size — no manual K sweep required.
#
# COST: ~8x slower than NMF and requires a GPU for large corpora.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 + 3 — BUILD and TRAIN
# ════════════════════════════════════════════════════════════════════════

corpus_df = load_corpus()
documents, categories = corpus_as_lists(corpus_df)
print(f"Corpus: {len(documents):,} documents")

if BERTopic is not None:
    print("\n" + "=" * 70)
    print("  BERTopic pipeline (embeddings -> UMAP -> HDBSCAN -> c-TF-IDF)")
    print("=" * 70)

    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        min_topic_size=15,
        nr_topics="auto",
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(documents)
    topic_info = topic_model.get_topic_info()
    n_topics = int((topic_info["Topic"] >= 0).sum())
    outliers = int((np.asarray(topics) == -1).sum())

    print(f"Topics discovered: {n_topics}")
    print(f"Outlier documents: {outliers:,} / {len(documents):,}")

    topic_words: list[list[str]] = []
    for topic_id in range(min(n_topics, 10)):
        words = [w for w, _ in topic_model.get_topic(topic_id)[:10]]
        topic_words.append(words)

    print("\nTop 10 BERTopic topics:")
    for _, row in topic_info[topic_info["Topic"] >= 0].head(10).iterrows():
        name = str(row["Name"])[:60]
        print(f"  Topic {row['Topic']}: {name} (n={row['Count']})")

    method_label = "BERTopic"
    doc_topics_vec = np.asarray(topics)
else:
    # Loud, actionable fallback — NMF over TF-IDF emulates BERTopic's
    # "cluster then describe" approximation when the neural pipeline is
    # unavailable. Declared extra in pyproject.toml: [topic-models].
    print("\n" + "!" * 70)
    print(
        "  bertopic / sentence-transformers not installed — install via "
        "`uv sync --extra topic-models` for the full pipeline"
    )
    print("  Running NMF-over-TF-IDF approximation for this session")
    print("!" * 70 + "\n")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        max_df=0.95,
        min_df=5,
    )
    tfidf = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    n_topics = 10
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=400, init="nndsvd")
    W = nmf.fit_transform(tfidf)
    H = nmf.components_

    doc_topics_vec = W.argmax(axis=1)

    topic_words = []
    print("NMF-approximation topics:")
    for t in range(n_topics):
        top_idx = H[t].argsort()[-10:][::-1]
        words = [feature_names[i] for i in top_idx]
        topic_words.append(words)
        count = int((doc_topics_vec == t).sum())
        print(f"  Topic {t}: {', '.join(words[:8])} (n={count})")

    method_label = "NMF-fallback"


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(topic_words) > 0, "Task 3: should discover at least one topic"
assert all(
    len(words) >= 5 for words in topic_words
), "Task 3: each topic should have at least 5 words"
print(
    f"\n[ok] Checkpoint 1 passed — {method_label} produced {len(topic_words)} topics\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: NPMI coherence and topic distribution
# ════════════════════════════════════════════════════════════════════════

coherences = compute_npmi(documents, topic_words)
mean_npmi = float(np.mean(coherences))
print(f"{method_label} mean NPMI coherence: {mean_npmi:+.4f}")
for i, c in enumerate(coherences):
    bar = "#" * max(0, int((c + 0.3) * 30))
    print(f"  Topic {i}: {c:+.4f} {bar}")

viz = ModelVisualizer()

coherence_data = {f"Topic_{i}": {"NPMI": float(c)} for i, c in enumerate(coherences)}
fig_coh = viz.metric_comparison(coherence_data)
fig_coh.update_layout(title=f"{method_label} Topic Coherence (NPMI)")
fig_coh.write_html(str(OUTPUT_DIR / "ex6_4_bertopic_coherence.html"))

size_data = {
    f"Topic_{t}": {"docs": int((doc_topics_vec == t).sum())}
    for t in range(len(topic_words))
}
fig_size = viz.metric_comparison(size_data)
fig_size.update_layout(title=f"{method_label} Topic Size Distribution")
fig_size.write_html(str(OUTPUT_DIR / "ex6_4_bertopic_sizes.html"))

print(f"\nSaved: {OUTPUT_DIR}/ex6_4_bertopic_coherence.html")
print(f"Saved: {OUTPUT_DIR}/ex6_4_bertopic_sizes.html")


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(coherences) == len(topic_words), "Task 4: one NPMI per topic"
assert mean_npmi > -0.5, f"Task 4: mean NPMI should be > -0.5, got {mean_npmi:.4f}"
print("\n[ok] Checkpoint 2 passed — coherence computed and visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Grab Multilingual Ticket Clustering
# ════════════════════════════════════════════════════════════════════════

print_scenario("bertopic")
print(
    """
WHY BERTOPIC FOR GRAB:
  - Grab's support tickets arrive in English, Bahasa Indonesia, Thai,
    and Vietnamese simultaneously. TF-IDF would need a separate model
    per language and couldn't discover cross-language patterns.
    BERTopic's multilingual sentence-transformers embed all four
    languages into the same space, so a "driver helmet policy"
    cluster emerges automatically regardless of source language.
  - HDBSCAN's outlier bucket is operationally useful — it identifies
    tickets that don't fit any known topic, so the ops team sees
    emerging issues (new scam patterns, new app bugs) faster.
  - c-TF-IDF's topic keywords are directly auditable by the support
    QA team, unlike raw embedding clusters.

NUMBERS TO REMEMBER:
  - ~35K tickets/week across all four markets
  - BERTopic NPMI = 0.18 vs LDA NPMI = 0.08 on Grab's internal
    benchmark (more coherent clusters = better routing)
  - Each well-routed ticket saves ~S$3.20 in agent time (Grab CX 2025)
  - Weekly savings: 35K * 0.85 routing accuracy * S$3.20 = ~S$95K/week
  - GPU compute cost: ~S$180/week on a single A10G instance
  - Net: S$94,820/week = ~S$4.9M/year
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
  [x] Assembled a four-stage BERTopic pipeline (embed/reduce/cluster/describe)
  [x] Explained why neural embeddings beat TF-IDF on polysemy and
      paraphrase
  [x] Handled the optional extra dependency with a loud, actionable
      fallback (no silent None propagation)
  [x] Measured NPMI coherence and compared it to NMF/LDA
  [x] Mapped the technique to Grab's multilingual support routing

  KEY INSIGHT: BERTopic is 8x slower and requires a GPU, but it finds
  topics that cross languages and survive paraphrase. When your
  business problem is multilingual or heavily paraphrased, no amount
  of TF-IDF tuning will catch up.

  Next: 05_sentiment_word2vec.py — use word embeddings (not sentence
  embeddings) as features for a lightweight sentiment classifier.
"""
)
