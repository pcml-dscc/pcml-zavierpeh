# MLFP04 — Task 3: NLP Topic Discovery with NMF

**Weight**: 25 marks · **Difficulty**: Hard · **Dataset**:
`data/mlfp04/sg_domain_qa.parquet` — **real** Singapore-domain question/answer
text (loaded via `shared.MLFPDataLoader`). This task uses the four most
lexically distinct domains: **finance, food, geography, transport** (616
documents).

## Scenario

A Singapore civic-information portal has a large pile of unlabelled
question/answer pairs and wants to auto-organise them into topics for its help
centre. The portal does **not** know which document belongs to which domain —
the domain labels exist only in the grader, which uses them to score how well
your topics match the real structure. Your job is unsupervised **topic
modelling**: turn each document into a TF-IDF vector and factor the
document-term matrix with **Non-negative Matrix Factorisation (NMF)** so each
document lands in one of four topics.

Use the kailash-ml **`DimReductionEngine`** with `algorithm="nmf"`
(`from kailash_ml.engines.dim_reduction import DimReductionEngine`). NMF
produces a non-negative `documents × topics` weight matrix; the topic with the
largest weight is the document's dominant topic. Raw sklearn is not permitted —
the engine is the framework-first surface.

Implement `solve() -> dict`.

## Required pipeline

1. **Load + select**: load `sg_domain_qa.parquet`, filter to the four domains,
   and sort by `["category", "instruction"]`. This fixed order is the canonical
   document order the grader aligns against — do **not** reorder.
2. **Vectorise**: build the TF-IDF document-term matrix. The helper
   `build_tfidf(...)` is **provided** in the starter (sublinear TF, L2-normalised
   rows, stopword removal, document-frequency vocabulary pruning) — use it as
   given so the matrix is deterministic.
3. **Factor**: wrap the matrix in a Polars DataFrame and call
   `DimReductionEngine().reduce(matrix_df, algorithm="nmf", n_components=4,
seed=42)`. Read the `transformed` field — the `documents × 4` topic-weight
   matrix.
4. **Assign**: each document's dominant topic is the `argmax` over its 4 topic
   weights.

## Output contract — `solve()` returns a `dict` with exactly these keys

| Key            | Type        | Meaning                                                |
| -------------- | ----------- | ------------------------------------------------------ |
| `doc_topics`   | `list[int]` | dominant topic id per document, in the canonical order |
| `n_topics`     | `int`       | number of topics (4)                                   |
| `topic_purity` | `float`     | cluster purity of `doc_topics` vs the true domains     |

`len(doc_topics)` must equal the number of selected documents (616).

## Visible sanity checks

- `result["n_topics"] == 4`
- all four topics are non-empty and no single topic swallows > 65% of documents
- the topics line up with the real domains: the grader checks **purity ≥ 0.65**,
  **adjusted Rand index ≥ 0.45**, and **normalised mutual information ≥ 0.55**
  (random assignment scores purity ≈ 0.25, ARI ≈ 0)

## Grading (10 automated checks, all must pass)

returns a dict · required keys present · `doc_topics` length 616 · `n_topics == 4`
· exactly 4 non-empty topics · no topic > 65% of documents · **purity ≥ 0.65** ·
**adjusted Rand index ≥ 0.45** · **NMI ≥ 0.55** · self-reported `topic_purity`
matches the grader (±0.05).

## Rules

- **kailash-ml `DimReductionEngine` (NMF) only** — raw sklearn is blocked.
- **Polars only** — no pandas. Load via `shared.MLFPDataLoader`.
- Deterministic — use the provided `build_tfidf` helper, keep `seed=42`, and the
  given sort order.
- The placeholder in `starter.py` fails grading by design.
