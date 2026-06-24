# MLFP05 — Task 3: GRU Time-Series Forecasting (beat the naive baseline)

**Weight**: 25 marks · **Difficulty**: Hard · **No GPU required** (trains on CPU in < 25s)

## Scenario

You are forecasting the **next value of a noisy oscillating process** — the kind of
signal you get from a vibrating machine component, a temperature controller, or a
seasonally-driven demand series. The bar to clear is the **naive last-value
forecast** ("next = most recent"). That baseline is genuinely hard to beat on noisy
data. Your job, exactly as in Exercise 3, is to build a small recurrent model (a
**GRU**) over fixed-length windows and **beat the naive baseline on held-out test
MSE**.

### CPU adaptation — and why the data is synthetic (read this)

Exercise 3 forecasts real STI / APAC stock prices. We tested a GRU on the bundled
`sti_prices.parquet`: **daily equity returns are a near-perfect random walk**, so
_no_ model — GRU, LSTM, transformer — beats "tomorrow = today" (we measured ratio
1.00). Grading a "beat the baseline" target on a random walk would be dishonest (it
is mathematically impossible). So this task uses a **synthetic series generated
in-process with a fixed seed** that has _genuine learnable structure_: a **damped
AR(2) oscillator** (its recurrence makes the next value depend on the _shape_ of the
recent window, not just the last value) plus a short **seasonal cycle** plus modest
noise. The skill tested is identical — windowed sequence forecasting that beats
last-value — but the target is now honestly achievable on CPU in seconds.

## Dataset

**Synthetic, deterministic** (`numpy.random.default_rng(13)`) — no download. Built by
`make_dataset()` in the starter:

- A length-3000 series: `y_t = 1.35·y_{t-1} − 0.55·y_{t-2} + 0.6·sin(2π t / 11) + ε_t`.
- Windowed into sequences of length `SEQ_LEN = 20` → predict the **next value**.
- Chronological split (first 80% train, last 20% test — no shuffling, no leakage).
- `naive_pred` = the **last observed value** in each test window (the last-value
  baseline). Because the series oscillates and turns at peaks/troughs, last-value is
  a weak predictor — but a real one a naive model would also lose to.

## Contract

```python
def solve() -> dict:
    ...
    return {
        "model":       <trained torch.nn.Module>,       # your GRU forecaster
        "test_pred":   <np.ndarray (Nte,) float>,       # your predicted next values
        "y_test":      <np.ndarray (Nte,) float>,       # true next values (passthrough)
        "naive_pred":  <np.ndarray (Nte,) float>,       # last-value baseline (passthrough)
        "uses_recurrent": <bool>,                        # True — model contains an RNN/GRU/LSTM
    }
```

Requirements baked into the grading:

1. **It is recurrent** — the model must contain an `nn.GRU`, `nn.LSTM`, or `nn.RNN`
   layer (the grader confirms by introspection; `uses_recurrent` must be `True`).
2. **Beat the naive baseline** — `MSE(test_pred, y_test) < MSE(naive_pred, y_test)`
   by a clear margin (model MSE <= **0.97 ×** naive MSE).
3. **`test_pred` comes from your model** — the grader re-runs the returned `model` on
   the re-derived `X_test` and checks it reproduces your predictions.
4. **Naive baseline is correct** — `naive_pred` must equal the last-value baseline the
   grader recomputes from the re-derived test windows.

## Performance target

- Test MSE <= **0.97 ×** naive-baseline test MSE (lower is better).

A small 1-layer GRU (hidden 16) trained ~60 epochs reaches ratio ~0.2 here — a huge
margin over the 0.97 ceiling.

## Visible sanity check

`solution.py` prints, when run directly:

```
gru  test_mse=x.xxe-01  naive_mse=x.xxe+00  ratio=0.2x
```

## Grading (8 automated checks, all must pass → 25 marks)

return type is a dict · required keys present · model is an `nn.Module` ·
model contains a recurrent layer (declared `uses_recurrent` matches introspection) ·
shapes of `test_pred`/`y_test`/`naive_pred` all match · model MSE beats naive MSE
(ratio <= 0.97) · re-running the model reproduces `test_pred` (anti-faking) ·
`naive_pred` is the correct last-value baseline.

## Rules

- **No GPU.** CPU only; the reference trains in well under 25 seconds.
- Raw **PyTorch is allowed** — this is the deep-learning module and Exercise 3 builds
  GRUs/LSTMs directly in `torch.nn`.
- Chronological split, **no leakage** — never let test windows inform training.
- Fix all seeds (`torch.manual_seed`) for reproducibility.
- No hardcoded API keys or model names.
