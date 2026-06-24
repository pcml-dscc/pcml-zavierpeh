# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6.1: TF-IDF and BM25 — Classic Text Retrieval
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Derive TF-IDF from a bag-of-words matrix
#   - Implement BM25 with saturation (k1) and length normalisation (b)
#   - Compare TF-IDF and BM25 on the same corpus
#   - Apply the technique to ST Engineering internal document search
#
# PREREQUISITES: Linear algebra (sparse matrices), basic probability.
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why IDF exists and what BM25 fixes
#   2. Build — CountVectorizer -> TfidfVectorizer -> manual BM25
#   3. Train — score terms across the corpus
#   4. Visualise — top-term rankings, BM25 saturation curve
#   5. Apply — ST Engineering scenario
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_6 import OUTPUT_DIR, TOY_CORPUS, print_scenario


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: bag-of-words -> TF-IDF
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  Bag-of-Words Warmup")
print("=" * 70)

# TODO: Build a CountVectorizer with English stop words and fit_transform
# the TOY_CORPUS. Store the fitted vectorizer and the matrix.
# Hint: CountVectorizer(stop_words="english"), .fit_transform(TOY_CORPUS)
bow_vectorizer = ____
X_bow = ____
bow_vocab = bow_vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(bow_vocab)}")
print(f"Matrix shape:    {X_bow.shape}")


print("\n" + "=" * 70)
print("  TF-IDF Weights")
print("=" * 70)

# TODO: Build a TfidfVectorizer (stop_words="english", norm="l2") and
# fit_transform the TOY_CORPUS. Expose get_feature_names_out() and idf_.
tfidf_vectorizer = ____
X_tfidf = ____
tfidf_vocab = tfidf_vectorizer.get_feature_names_out()
idf_values = tfidf_vectorizer.idf_


# ── Checkpoint 1 ─────────────────────────────────────────────────────
idf_dict = dict(zip(tfidf_vocab, idf_values))
assert (
    idf_dict["singapore"] < idf_dict["monetary"]
), "Task 2: 'singapore' (common) should have lower IDF than 'monetary' (rare)"
row_norms = np.sqrt(np.asarray(X_tfidf.multiply(X_tfidf).sum(axis=1)).flatten())
assert all(
    abs(n - 1.0) < 0.01 for n in row_norms if n > 0
), "Task 2: TF-IDF rows should be L2-normalised"
print("\n[ok] Checkpoint 1 passed — TF-IDF IDF correctly ranks rare vs common terms\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN-FREE SCORING: manual BM25
# ════════════════════════════════════════════════════════════════════════
# BM25(t, d) = IDF(t) * (tf * (k1 + 1)) /
#              (tf + k1 * (1 - b + b * |d| / avgdl))


def bm25_score(
    tf: float,
    df: int,
    N: int,
    dl: int,
    avgdl: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """Return BM25 score for a single (term, document) pair."""
    # TODO: Compute the Robertson/Sparck-Jones IDF:
    # idf = log((N - df + 0.5) / (df + 0.5) + 1)
    idf = ____

    # TODO: Compute the saturated TF component using tf, k1, b, dl, avgdl
    # Hint: numerator = tf * (k1 + 1); denominator = tf + k1 * (1 - b + b * dl / avgdl)
    tf_component = ____

    return idf * tf_component


doc_lengths = [len(doc.split()) for doc in TOY_CORPUS]
avgdl = float(np.mean(doc_lengths))
N_docs = len(TOY_CORPUS)
sg_df = sum(1 for doc in TOY_CORPUS if "singapore" in doc.lower())

tf_grid = np.arange(1, 21)
saturation_scores = [
    bm25_score(float(tf), df=2, N=100, dl=50, avgdl=40.0) for tf in tf_grid
]


# ── Checkpoint 2 ─────────────────────────────────────────────────────
test_bm25 = bm25_score(tf=3, df=2, N=10, dl=50, avgdl=40)
assert test_bm25 > 0, "Task 3: BM25 should be positive for present terms"
score_tf1 = bm25_score(tf=1, df=2, N=10, dl=50, avgdl=40)
score_tf10 = bm25_score(tf=10, df=2, N=10, dl=50, avgdl=40)
assert score_tf10 < 10 * score_tf1, "Task 3: BM25 should saturate"
print("[ok] Checkpoint 2 passed — BM25 with saturation + length normalisation\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: top-term ranking + BM25 saturation curve
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# TODO: Compute the mean TF-IDF weight per column and surface the top 10.
# Hint: np.asarray(X_tfidf.mean(axis=0)).flatten().argsort()[-10:][::-1]
mean_tfidf = ____
top_idx = ____
top_terms = {tfidf_vocab[i]: float(mean_tfidf[i]) for i in top_idx}

term_ranking = {term: {"mean_tfidf": w} for term, w in top_terms.items()}
fig_terms = viz.metric_comparison(term_ranking)
fig_terms.update_layout(title="Top Terms by Mean TF-IDF")
fig_terms.write_html(str(OUTPUT_DIR / "ex6_1_tfidf_top_terms.html"))

sat_curve = {
    f"tf={tf}": {"BM25": float(score)} for tf, score in zip(tf_grid, saturation_scores)
}
fig_sat = viz.metric_comparison(sat_curve)
fig_sat.update_layout(title="BM25 Saturation Curve (k1=1.2, b=0.75)")
fig_sat.write_html(str(OUTPUT_DIR / "ex6_1_bm25_saturation.html"))


# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(top_terms) == 10, "Task 4: should surface top 10 terms"
assert saturation_scores[0] < saturation_scores[-1], "Task 4: BM25 monotonic in tf"
print("[ok] Checkpoint 3 passed — visualisations written\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: ST Engineering Internal Search
# ════════════════════════════════════════════════════════════════════════

print_scenario("tfidf_bm25")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a bag-of-words and TF-IDF matrix with sklearn
  [x] Implemented BM25 with saturation + length normalisation
  [x] Visualised top terms and the BM25 saturation curve
  [x] Mapped the technique to ST Engineering internal search

  Next: 02_nmf_topics.py — factorise the TF-IDF matrix into topics.
"""
)
