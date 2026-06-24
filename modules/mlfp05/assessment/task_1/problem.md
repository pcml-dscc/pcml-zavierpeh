# MLFP05 — Task 1: Autoencoder Anomaly Detection on Sensor Telemetry

**Weight**: 25 marks · **Difficulty**: Hard · **No GPU required** (trains on CPU in < 15s)

## Scenario

A Singapore precision-manufacturing line streams **12 sensor channels** (vibration,
temperature, current draw, acoustic, ...) per machine cycle. You have **no labelled
failures** — only the engineering assumption that _healthy_ cycles all "look alike"
and that an impending bearing fault produces telemetry the line has never emitted
before.

This is the classic unsupervised anomaly-detection setup from Exercise 1: train an
**undercomplete autoencoder on healthy-only data**, then score every incoming cycle
by its **reconstruction error**. Healthy cycles reconstruct well (low error);
anomalous cycles reconstruct poorly (high error) because the encoder never learned
their structure. The reconstruction error is the anomaly score.

Implement `solve()`. It must train the AE and return the scored test set so the
grader can verify the detector actually separates the planted anomalies.

## Dataset

**Synthetic, generated in-process with a fixed seed** (`numpy.random.default_rng(7)`)
— no download, fully deterministic. Provided to you by `make_dataset()` in the
starter:

- `X_train` — `(800, 12)` float32, **healthy cycles only** (a low-rank correlated
  signal + small Gaussian noise). Train the AE on this.
- `X_test` — `(400, 12)` float32, a mix of healthy and anomalous cycles.
- `y_test` — `(400,)` int, ground-truth labels (`0` = healthy, `1` = anomaly).
  **Use this ONLY to compute the final AUC** — never as a training signal.

Anomalies are planted by breaking the healthy correlation structure (off-manifold
shifts), so a bottleneck AE that learned the healthy manifold will score them high.

## Contract

```python
def solve() -> dict:
    ...
    return {
        "model":       <trained torch.nn.Module>,   # the autoencoder
        "scores":      <np.ndarray (400,) float>,   # per-test-row reconstruction MSE
        "y_test":      <np.ndarray (400,) int>,     # labels, passed straight through
        "input_dim":   12,                          # int
        "latent_dim":  <int>,                       # bottleneck size you chose, < 12
    }
```

Requirements baked into the grading:

1. **Bottleneck is undercomplete** — `latent_dim < input_dim` (a non-compressing AE
   can learn the identity and will fail to separate anomalies).
2. **`scores` are genuine reconstruction errors** — the grader re-runs your returned
   `model` on a fresh healthy batch and a fresh anomalous batch and checks that the
   model itself (not just your array) assigns higher error to anomalies.
3. **Detector quality** — ROC-AUC of `scores` vs `y_test` **>= 0.90**.
4. **Honest split** — mean reconstruction error on healthy test rows must be clearly
   below the mean on anomalous test rows (separation ratio >= 1.5x).

## Performance target

- ROC-AUC (`scores` vs `y_test`) **>= 0.90**
- Reconstruction-error separation (anomaly mean / healthy mean) **>= 1.5x**

A correctly-built undercomplete AE trained ~40 epochs reaches AUC ~0.98 here.

## Visible sanity check

`solution.py` prints, when run directly:

```
latent_dim=4  AUC=0.9xx  separation=x.xx
```

## Grading (8 automated checks, all must pass → 25 marks)

return type is a dict · required keys present · model is an `nn.Module` ·
`latent_dim < input_dim` (genuine bottleneck) · `scores` shape matches `y_test` ·
AUC >= 0.90 · separation ratio >= 1.5x · model re-run on fresh data still ranks
anomalies above healthy (the scores are not faked).

## Rules

- **No GPU.** CPU only; the reference trains in well under 15 seconds.
- Raw **PyTorch is allowed** — this is the deep-learning module and Exercise 1 builds
  AEs directly in `torch.nn`.
- **Polars** for any tabular work; **no pandas**.
- Fix all seeds (`torch.manual_seed`) so your run is reproducible.
- Do **not** read `y_test` during training — it is a held-out evaluation label only.
- No hardcoded API keys or model names.
