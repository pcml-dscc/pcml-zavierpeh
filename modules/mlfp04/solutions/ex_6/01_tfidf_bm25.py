# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6.1: TF-IDF and BM25 — Classic Text Retrieval
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Derive TF-IDF from a bag-of-words matrix and explain why IDF
#     down-weights common words
#   - Implement BM25 scoring with term-frequency saturation (k1) and
#     document length normalisation (b)
#   - Compare TF-IDF and BM25 ranking on the same corpus
#   - Read a TF-IDF/BM25 score as a retrieval signal, not a probability
#   - Apply the technique to ST Engineering internal document search
#
# PREREQUISITES: Linear algebra (sparse matrices), basic probability.
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why IDF exists and what BM25 fixes
#   2. Build — CountVectorizer -> TfidfVectorizer -> manual BM25
#   3. Train — score terms across the corpus (no gradient descent)
#   4. Visualise — top-term rankings, BM25 saturation curve
#   5. Apply — ST Engineering internal search scenario
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_6 import OUTPUT_DIR, TOY_CORPUS, print_scenario


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why TF-IDF, Why BM25
# ════════════════════════════════════════════════════════════════════════
# A bag-of-words vector counts every term equally, so "the" dominates
# "monetary" even though "the" carries no meaning. IDF fixes this:
#
#   TF(t, d) = count(t in d) / length(d)
#   IDF(t)   = log(N / df(t))
#   TF-IDF   = TF * IDF
#
# Rare terms get high IDF and dominate the score; common terms ("the",
# "singapore" in a Singapore corpus) get shrunk toward zero.
#
# BM25 is the next refinement, used by Elasticsearch and every modern
# search engine:
#
#   BM25(t, d) = IDF(t) * (tf * (k1 + 1)) /
#                (tf + k1 * (1 - b + b * |d| / avgdl))
#
# Two improvements over TF-IDF:
#   1. TF saturates (k1 = 1.2): the 10th mention of a word adds less
#      than the 1st — matching the intuition that repeating "turbine"
#      100 times doesn't make a document 100x more about turbines.
#   2. Length normalisation (b = 0.75): long documents don't
#      automatically accumulate higher scores.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: bag-of-words -> TF-IDF -> BM25
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  Bag-of-Words Warmup")
print("=" * 70)

bow_vectorizer = CountVectorizer(stop_words="english")
X_bow = bow_vectorizer.fit_transform(TOY_CORPUS)
bow_vocab = bow_vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(bow_vocab)}")
print(f"Matrix shape:    {X_bow.shape} (docs x vocab)")
print("\nDoc 0 non-zero terms:")
doc0 = X_bow[0].toarray()[0]
for term, count in zip(bow_vocab, doc0):
    if count > 0:
        print(f"  '{term}': {int(count)}")


print("\n" + "=" * 70)
print("  TF-IDF Weights (Docs 0 & 1)")
print("=" * 70)

tfidf_vectorizer = TfidfVectorizer(stop_words="english", norm="l2")
X_tfidf = tfidf_vectorizer.fit_transform(TOY_CORPUS)
tfidf_vocab = tfidf_vectorizer.get_feature_names_out()
idf_values = tfidf_vectorizer.idf_
doc0_tfidf = X_tfidf[0].toarray()[0]
doc1_tfidf = X_tfidf[1].toarray()[0]

print(f"{'Term':<20} {'Doc0 TF-IDF':>14} {'Doc1 TF-IDF':>14} {'IDF':>10}")
print("-" * 62)
for term, idf, t0, t1 in sorted(
    zip(tfidf_vocab, idf_values, doc0_tfidf, doc1_tfidf),
    key=lambda x: -abs(x[2] + x[3]),
)[:12]:
    print(f"  {term:<20} {t0:>14.4f} {t1:>14.4f} {idf:>10.4f}")

print("\nInterpretation:")
print("  'singapore' appears in 4/8 docs -> low IDF -> penalised")
print("  'monetary'  appears in 1/8 docs -> high IDF -> rewarded")


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


def bm25_score(
    tf: float,
    df: int,
    N: int,
    dl: int,
    avgdl: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """Compute BM25 for a single (term, document) pair.

    IDF uses the Robertson/Sparck-Jones variant: log((N - df + 0.5) / (df + 0.5) + 1).
    """
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
    tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return idf * tf_component


doc_lengths = [len(doc.split()) for doc in TOY_CORPUS]
avgdl = float(np.mean(doc_lengths))
N_docs = len(TOY_CORPUS)

print("=" * 70)
print("  BM25 Scoring for 'singapore'")
print("=" * 70)
print(f"Document lengths: {doc_lengths}")
print(f"Average length:   {avgdl:.1f}")

sg_df = sum(1 for doc in TOY_CORPUS if "singapore" in doc.lower())

print(f"\n{'Doc':>4} {'TF':>4} {'Len':>5} {'TF-IDF':>10} {'BM25':>10}")
print("-" * 38)
sg_idx = list(tfidf_vocab).index("singapore")
for i, doc in enumerate(TOY_CORPUS):
    words = doc.lower().split()
    tf = words.count("singapore")
    if tf > 0:
        tfidf_val = float(X_tfidf[i].toarray()[0, sg_idx])
        bm25_val = bm25_score(tf, sg_df, N_docs, len(words), avgdl)
        print(f"{i:>4} {tf:>4} {len(words):>5} {tfidf_val:>10.4f} {bm25_val:>10.4f}")

# Saturation demonstration: TF grows 1 -> 20, score grows sub-linearly
tf_grid = np.arange(1, 21)
saturation_scores = [
    bm25_score(float(tf), df=2, N=100, dl=50, avgdl=40.0) for tf in tf_grid
]


# ── Checkpoint 2 ─────────────────────────────────────────────────────
test_bm25 = bm25_score(tf=3, df=2, N=10, dl=50, avgdl=40)
assert test_bm25 > 0, "Task 3: BM25 should be positive for present terms"
score_tf1 = bm25_score(tf=1, df=2, N=10, dl=50, avgdl=40)
score_tf10 = bm25_score(tf=10, df=2, N=10, dl=50, avgdl=40)
assert score_tf10 < 10 * score_tf1, "Task 3: BM25 should saturate (tf=10 < 10 * tf=1)"
print("\n[ok] Checkpoint 2 passed — BM25 with saturation + length normalisation\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: top-term ranking + BM25 saturation curve
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Top terms by mean TF-IDF across the corpus
mean_tfidf = np.asarray(X_tfidf.mean(axis=0)).flatten()
top_idx = mean_tfidf.argsort()[-10:][::-1]
top_terms = {tfidf_vocab[i]: float(mean_tfidf[i]) for i in top_idx}

print("Top 10 terms by mean TF-IDF:")
for term, weight in top_terms.items():
    bar = "#" * int(weight * 40)
    print(f"  {term:<14} {weight:.4f} {bar}")

term_ranking = {term: {"mean_tfidf": weight} for term, weight in top_terms.items()}
fig_terms = viz.metric_comparison(term_ranking)
fig_terms.update_layout(title="Top Terms by Mean TF-IDF")
fig_terms.write_html(str(OUTPUT_DIR / "ex6_1_tfidf_top_terms.html"))

# BM25 saturation curve
sat_curve = {
    f"tf={tf}": {"BM25": float(score)} for tf, score in zip(tf_grid, saturation_scores)
}
fig_sat = viz.metric_comparison(sat_curve)
fig_sat.update_layout(title="BM25 Saturation Curve (k1=1.2, b=0.75)")
fig_sat.write_html(str(OUTPUT_DIR / "ex6_1_bm25_saturation.html"))

print(f"\nSaved: {OUTPUT_DIR}/ex6_1_tfidf_top_terms.html")
print(f"Saved: {OUTPUT_DIR}/ex6_1_bm25_saturation.html")


# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(top_terms) == 10, "Task 4: should surface top 10 terms"
assert saturation_scores[0] < saturation_scores[-1], "Task 4: BM25 monotonic in tf"
assert (
    saturation_scores[-1] < 10 * saturation_scores[0]
), "Task 4: saturation must bend the curve"
print("\n[ok] Checkpoint 3 passed — visualisations written\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: ST Engineering Internal Search
# ════════════════════════════════════════════════════════════════════════

print_scenario("tfidf_bm25")
print(
    """
WHY BM25 WINS HERE:
  - ST Engineering reports vary from 2-page memos to 80-page aerospace
    manuals. Pure TF-IDF rewards the longer manuals even when the memo
    is more relevant (longer doc = more accumulated weight).
  - BM25's b=0.75 normalises for document length, so a 2-page memo with
    three mentions of "turbine blade fatigue" outranks an 80-page manual
    with 12 mentions buried in supply-chain minutiae.
  - TF saturation (k1=1.2) prevents the aerospace manual from dominating
    just because "turbine" appears 90 times in its table of contents.

NUMBERS TO REMEMBER:
  - BM25 is ~14% more accurate at first-hit retrieval than TF-IDF on the
    Lemur/TREC test collection.
  - Every missed relevant document costs ST Engineering ~S$450K in
    duplicated R&D (industry benchmark).
  - For 180K reports, the 14% improvement is ~900 fewer misses/year =
    ~S$400M in avoided duplicated R&D cost.
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
  [x] Built a bag-of-words and TF-IDF matrix with sklearn
  [x] Explained why rare terms get higher IDF (discriminative power)
  [x] Implemented BM25 with k1 (saturation) and b (length normalisation)
  [x] Observed BM25's sub-linear growth in TF empirically
  [x] Mapped the technique to ST Engineering's document-search problem

  KEY INSIGHT: TF-IDF and BM25 require NO training. They are statistical
  summaries of the corpus, computed once, reused at every query. This
  makes them ideal baselines — if BM25 is already 90% as accurate as
  your neural retriever at 0.001% the latency, the neural model is
  probably not worth the cost.

  Next: 02_nmf_topics.py — factorise the TF-IDF matrix into topics.
"""
)
