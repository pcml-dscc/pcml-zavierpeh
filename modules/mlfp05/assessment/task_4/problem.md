# MLFP05 — Task 4: Tiny Transformer for Text Classification (from scratch)

**Weight**: 25 marks · **Difficulty**: Hard · **No GPU required** (trains on CPU in < 35s)

## Scenario

A newsroom triage tool must route incoming headlines into four desks — **World,
Sports, Business, Sci/Tech**. You will build a **tiny Transformer encoder from
scratch** — token embeddings + a single **multi-head self-attention block** +
mean-pooling + a linear classifier — exactly the architecture pattern from Exercise 4
(self-attention from scratch → transformer encoder). Train it on a small bundled
slice of AG News and clear an accuracy floor that a bag-of-words guess cannot.

### CPU adaptation (read this)

Exercise 4 fine-tunes **BERT** (110M parameters, large download) and trains on the
full 120K AG News corpus. **That is not CPU-friendly and downloading a pretrained
backbone is forbidden here.** This task keeps the _same skill_ — implement scaled
dot-product self-attention and use it to classify text — but with a **from-scratch
tiny encoder** (a couple of attention blocks, small embedding) on the **bundled**
5K-row AG News slice. No HuggingFace download, no pretrained weights: you build the
attention mechanism yourself. The accuracy floor (0.72) is set so a genuine attention
classifier clears it comfortably while a constant/majority guess (~0.27) cannot — the
5K-row slice has a real accuracy ceiling (~0.80), so the floor leaves honest margin.

## Dataset

**`data/mlfp05/ag_news.parquet`** (5,000 rows) + **`ag_news_test.parquet`** (1,000
rows) — committed to the repo (no download). Columns: `text` (headline+blurb),
`label` (0=World, 1=Sports, 2=Business, 3=Sci/Tech). `make_dataset()` in the starter:

- Builds a deterministic word-level vocabulary (lowercased, capped at 8,000 tokens)
  from the **training** text only.
- Encodes each row to a fixed-length (`MAX_LEN = 40`) padded index sequence.
- Returns `X_train (5000,40) int64`, `y_train`, `X_test (1000,40) int64`, `y_test`,
  and `vocab_size`. **Use `y_test` only for the final accuracy score.**

## Contract

```python
def solve() -> dict:
    ...
    return {
        "model":      <trained torch.nn.Module>,        # your tiny transformer
        "preds":      <np.ndarray (1000,) int>,         # argmax predictions on X_test
        "y_test":     <np.ndarray (1000,) int>,         # labels, passed straight through
        "uses_attention": <bool>,                        # True — model uses self-attention
    }
```

Requirements baked into the grading:

1. **It uses attention** — the model must contain a `nn.MultiheadAttention` **or** a
   `nn.TransformerEncoderLayer`/`nn.TransformerEncoder` (the grader confirms by
   introspection; `uses_attention` must be `True`). A plain MLP/embedding-average
   without attention does not satisfy this.
2. **`preds` come from your model** — the grader re-runs your returned `model` on the
   re-derived `X_test` and checks the predictions match what you submitted.
3. **Test accuracy** of `preds` vs `y_test` **>= 0.72**.
4. **Beats the majority-class baseline** — accuracy must exceed the most-frequent-class
   rate by a clear margin (the grader recomputes the majority rate, ~0.30).

## Performance target

- Test accuracy **>= 0.72** (a correct 2-block tiny transformer reaches ~0.79 here).

## Visible sanity check

`solution.py` prints, when run directly:

```
transformer  test_acc=0.79  majority=0.27
```

## Grading (8 automated checks, all must pass → 25 marks)

return type is a dict · required keys present · model is an `nn.Module` ·
model uses self-attention (declared `uses_attention` matches introspection) ·
`preds` shape matches `y_test` · test accuracy >= 0.72 · re-running the model
reproduces the submitted `preds` (anti-faking) · accuracy beats the majority-class
baseline by >= 0.15.

## Rules

- **No GPU.** CPU only; the reference trains in well under 35 seconds.
- Raw **PyTorch is allowed** — this is the deep-learning module and Exercise 4 builds
  attention/transformers directly in `torch.nn`.
- **No pretrained backbones / downloads** (no BERT, no HuggingFace weights) — build
  the encoder from scratch.
- **Polars** for loading the parquet; **no pandas**.
- Build the vocabulary from **training text only** — no leakage from the test split.
- Fix all seeds (`torch.manual_seed`) for reproducibility.
- Do **not** train on `y_test`. No hardcoded API keys or model names.
