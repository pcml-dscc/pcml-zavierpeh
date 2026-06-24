# MLFP04 — Task 4: Neural Network Foundations

**Weight**: 25 marks · **Difficulty**: Hard · **Dataset**: deterministic synthetic
concentric circles (fixed seed `20260404`, 800 points, 2 features — generated
inside the task, no file needed)

## Scenario

This is the canonical "why hidden layers matter" problem. Two classes form
concentric rings: class 0 is a small inner ring, class 1 a larger outer ring.
Both rings are centred on the origin, so they share the same centroid — **no
straight line can separate them**. A linear model (logistic regression, a
single-layer perceptron) is stuck near 50% accuracy. A neural network with a
**hidden layer** carves a curved boundary and solves it.

You must train the network through a **kailash-ml Trainable adapter**, not a raw
PyTorch training loop. The reference uses
`SklearnTrainable(estimator=MLPClassifier(hidden_layer_sizes=(32, 16), ...))` —
a multi-layer perceptron driven by the kailash-ml engine. A linear model will
not pass: the grader independently confirms a linear classifier cannot clear the
accuracy floor on this data.

Implement `solve() -> dict`.

## Required pipeline

1. **Generate** the concentric-circles dataset (helper given in starter).
2. **Split** deterministically: the first 600 rows are train, the last 200 are
   test (the helpers `df.head(600)` / `df.tail(200)`).
3. **Train** a multi-layer perceptron through `SklearnTrainable` — at least one
   hidden layer, `random_state=SEED`. Call `.fit(train_df)`.
4. **Predict** on the held-out test rows. The kailash-ml prediction object
   exposes `.to_polars()` and `.column`; pull the predicted-label column out as
   integers.
5. **Score** test accuracy and train accuracy against the true labels.

## Output contract — `solve()` returns a `dict` with exactly these keys

| Key                | Type        | Meaning                                             |
| ------------------ | ----------- | --------------------------------------------------- |
| `test_predictions` | `list[int]` | predicted label (0/1) for each of the 200 test rows |
| `test_accuracy`    | `float`     | accuracy on the test rows                           |
| `train_accuracy`   | `float`     | accuracy on the training rows                       |

`len(test_predictions)` must equal 200, in test-row order.

## Visible sanity checks

- a linear model scores ≈ 0.5–0.6 on this data; your network should reach
  **≥ 0.90** test accuracy
- `test_predictions` contains both classes (not all one label)
- the grader recomputes your accuracy from `test_predictions` — self-reporting a
  number you did not actually achieve will fail the honesty check

## Grading (10 automated checks, all must pass)

returns a dict · required keys present · `test_predictions` length 200 ·
predictions are binary 0/1 · predictions use both classes · **grader-recomputed
test accuracy ≥ 0.90** · self-reported `test_accuracy` matches the grader
(±0.03) · `train_accuracy` in `[0.90, 1.0]` · problem is certified non-linear
(class centroids coincide) · **an independent linear classifier scores < 0.70**
while your model clears 0.90 (proves you built a non-linear model).

## Rules

- **kailash-ml Trainable only** — no raw `torch` training loop. The MLP must run
  through `SklearnTrainable` (or another kailash-ml Trainable).
- **Polars only** — no pandas.
- Deterministic — keep the given seed, sizes, split, and `random_state`.
- The placeholder in `starter.py` fails grading by design.
