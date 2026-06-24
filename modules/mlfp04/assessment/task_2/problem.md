# MLFP04 — Task 2: Dimensionality Reduction & Anomaly Detection

**Weight**: 25 marks · **Difficulty**: Hard · **Dataset**: deterministic synthetic
sensor matrix (fixed seed `20260402`, 1,000 rows × 24 features — generated inside
the task, no file needed)

## Scenario

A Singapore logistics operator streams 24 telemetry channels per delivery van.
The channels are highly redundant — they are driven by just **three** latent
factors (engine load, route congestion, driver behaviour) plus measurement
noise, so the data really lives on a low-dimensional manifold. Buried in the
fleet are **25 malfunctioning vans** whose telemetry sits far off that manifold.
You must (a) prove the redundancy by compressing the 24 channels with PCA and
(b) flag the off-manifold vans with an anomaly detector — all through
kailash-ml engines.

Use **`DimReductionEngine`** (`from kailash_ml.engines.dim_reduction import DimReductionEngine`)
for PCA and **`AnomalyDetectionEngine`**
(`from kailash_ml.engines.anomaly_detection import AnomalyDetectionEngine`) for
detection. Raw `sklearn` is not permitted.

Implement `solve() -> dict`.

## Required pipeline

1. **Generate** the deterministic 24-feature matrix (helper given in starter).
2. **Find the intrinsic dimensionality**: fit PCA at full rank
   (`reduce(df, algorithm="pca", n_components=24)`), read
   `explained_variance_ratio`, and compute the smallest number of components
   whose **cumulative** explained variance reaches **≥ 0.90**. Call it
   `n_components_90`.
3. **Compress**: re-run `reduce(df, algorithm="pca", n_components=n_components_90)`
   and read `reconstruction_error` off the `DimReductionResult`.
4. **Detect anomalies**: run
   `AnomalyDetectionEngine().detect(df, algorithm="isolation_forest",
contamination=0.025)`. Read `scores` (higher = more anomalous), `labels`,
   and `n_anomalies` off the `AnomalyResult`. Produce a binary
   `anomaly_labels` list (1 = flagged anomaly, 0 = normal).

## Output contract — `solve()` returns a `dict` with exactly these keys

| Key                    | Type          | Meaning                                           |
| ---------------------- | ------------- | ------------------------------------------------- |
| `n_components_90`      | `int`         | components needed for ≥90% cumulative variance    |
| `reconstruction_error` | `float`       | PCA reconstruction error at `n_components_90`     |
| `anomaly_scores`       | `list[float]` | anomaly score per row (higher = more anomalous)   |
| `anomaly_labels`       | `list[int]`   | 1 = flagged anomaly, 0 = normal, per row          |
| `n_anomalies`          | `int`         | number of rows flagged (== sum of anomaly_labels) |

Both lists have length 1,000, in the generated row order.

## Visible sanity checks

- `result["n_components_90"] == 3` (the three latent factors are recovered — an
  8× compression of the 24 channels)
- `result["n_components_90"] < 24`
- the anomaly detector ranks the 25 planted malfunctions near the top: the
  grader checks **ROC-AUC of the scores vs the hidden anomaly flags ≥ 0.85**
- `n_anomalies` is roughly 25 (≈ 2.5% contamination)

## Grading (10 automated checks, all must pass)

returns a dict · required keys present · `n_components_90 == 3` · compression is
real (`< 24` features) · `reconstruction_error` matches the engine reference
(within tolerance) and is positive · `anomaly_scores` length 1,000 ·
`anomaly_labels` length 1,000 · **ROC-AUC of scores vs planted anomalies ≥ 0.85**
· `n_anomalies` consistent (`== sum(labels)`, in `[10, 60]`) · flagged-set
precision ≥ 0.5.

## Rules

- **kailash-ml engines only** — `DimReductionEngine` + `AnomalyDetectionEngine`;
  raw sklearn is blocked.
- **Polars only** — no pandas.
- Deterministic — keep the given seed and `contamination=0.025`.
- The placeholder in `starter.py` fails grading by design.
