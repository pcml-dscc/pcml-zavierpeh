# MLFP04 — Task 1: Customer Segmentation by Clustering

**Weight**: 25 marks · **Difficulty**: Hard · **Dataset**: deterministic synthetic
loyalty cohort (fixed seed `20260401`, 1,200 customers, 5 RFM-style features —
generated inside the task, no file needed)

## Scenario

A Singapore e-commerce loyalty team hands you a behavioural table with five
numeric features per customer: `recency_days`, `frequency`, `monetary_sgd`,
`tenure_months`, `avg_basket_sgd`. They believe there are a handful of distinct
spending personas hidden in the data but have **no labels** — this is pure
unsupervised discovery. Four genuine personas were planted in the data
generator (champions, new-low-value, dormant-at-risk, loyal-big-basket); your
job is to recover them without ever seeing the planted labels.

You must use the **kailash-ml `ClusteringEngine`** (`from kailash_ml.engines.clustering import ClusteringEngine`).
Raw `sklearn` clustering is not permitted — the engine is the framework-first
surface for this module.

Implement `solve() -> dict`.

## Required pipeline

1. **Generate** the deterministic customer table (the helper is given in the
   starter — do not change the seed or sizes).
2. **Standardise** every feature to a z-score (subtract mean, divide by std) in
   Polars. The raw features span wildly different scales (`monetary_sgd` ≈ 2000
   vs `frequency` ≈ 50); without standardisation distance is dominated by one
   column and recovery collapses. Standardisation is **load-bearing**.
3. **Select K** objectively with `ClusteringEngine.sweep_k(zdf, range(2, 9),
algorithm="kmeans", criterion="silhouette")`. Read `optimal_k` — do **not**
   hardcode the answer.
4. **Fit** `ClusteringEngine.fit(zdf, algorithm="kmeans", n_clusters=optimal_k)`
   and read `labels` and `silhouette_score` off the `ClusterResult`.

## Output contract — `solve()` returns a `dict` with exactly these keys

| Key          | Type        | Meaning                                             |
| ------------ | ----------- | --------------------------------------------------- |
| `labels`     | `list[int]` | cluster id per customer, in the generated row order |
| `n_clusters` | `int`       | the K recovered by the silhouette sweep             |
| `silhouette` | `float`     | silhouette score of the final fit                   |

`len(labels)` must equal 1,200. Each label is an integer in `[0, n_clusters)`.

## Visible sanity checks

After a correct implementation:

- `result["n_clusters"] == 4` (the sweep recovers the planted persona count)
- `result["silhouette"] > 0.55` (well-separated personas)
- all four clusters are non-empty
- the partition matches the planted personas (adjusted Rand index ≈ 1.0 — the
  grader checks this against the hidden labels)

## Grading (10 automated checks, all must pass)

returns a dict · required keys present · `labels` length 1,200 · `n_clusters == 4`
· exactly 4 distinct non-empty clusters · grader-recomputed silhouette ≥ 0.55 ·
self-reported silhouette matches the grader (±0.05) · **adjusted Rand index vs
the planted personas ≥ 0.90** · adjusted mutual information ≥ 0.85 · labels are
valid integers in range.

## Rules

- **kailash-ml `ClusteringEngine` only** — raw sklearn clustering is blocked.
- **Polars only** — no pandas.
- Deterministic — keep the given seed; no extra randomness.
- The placeholder in `starter.py` fails grading by design.
