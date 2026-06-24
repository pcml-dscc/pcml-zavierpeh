# MLFP03 — Task 2: The Model Zoo

**Weight**: 25 marks · **Difficulty**: Hard · **Dataset**: `data/mlfp03/ecommerce_customers.parquet` (50,000 rows, 16 columns)

## Scenario

The premium-upsell project (Task 1) needs a model. Before tuning anything, the
team wants a **fair head-to-head bake-off**: train a spread of classical and
ensemble classifiers on identical data and folds, then rank them on one table.
You must train each algorithm through the kailash-ml `TrainingPipeline` — no raw
`sklearn.fit()` in your code.

Implement `solve() -> pl.DataFrame`.

## Data contract (deterministic — given to you)

- First **10,000** rows (sorted by `customer_id`).
- Target `premium_response` — the derived premium-upsell label from Task 1
  (~25% positive). The full derivation is in `_model_frame()`; keep it intact.
- 8 base features: `satisfaction_score, avg_order_value, num_returns,
  order_count, loyalty_int, total_revenue, days_since_last_order,
  customer_tenure_days`.
- Split: `TrainingPipeline` holdout, `test_size=0.25` (deterministic seed-42
  shuffle handled by the engine).

## Required algorithms (train all six)

| `model` value | model_class | framework |
| ------------- | ----------- | --------- |
| `logistic_regression` | `sklearn.linear_model.LogisticRegression` | sklearn |
| `naive_bayes`         | `sklearn.naive_bayes.GaussianNB`          | sklearn |
| `decision_tree`       | `sklearn.tree.DecisionTreeClassifier`     | sklearn |
| `random_forest`       | `sklearn.ensemble.RandomForestClassifier` | sklearn |
| `extra_trees`         | `sklearn.ensemble.ExtraTreesClassifier`   | sklearn |
| `lightgbm`            | `lightgbm.LGBMClassifier`                 | lightgbm |

Hyperparameters are pre-filled in `MODEL_ZOO` — use them as given (all seeded).

## Exact return contract

A Polars DataFrame, **one row per algorithm**, columns in this exact order,
**sorted by `auc` descending**:

```
model (str) | accuracy (f64) | f1 (f64) | auc (f64)
```

Each metric comes from the `TrainingPipeline` evaluation (`EvalSpec` metrics
`["accuracy", "f1", "auc"]`).

## Visible sanity checks

After a correct implementation:

- `result.shape == (6, 4)`
- `result.columns == ["model", "accuracy", "f1", "auc"]`
- every `auc > 0.82`; the best `auc ≈ 0.91` (`logistic_regression` tends to top
  the table here — the target is mostly additive — with `lightgbm` close behind)
- at least one ensemble (`random_forest` / `extra_trees` / `lightgbm`) sits in
  the top 3.

## Performance target

Best model **ROC-AUC ≥ 0.88** and best **F1 ≥ 0.80** on the held-out split.

## Grading (12 automated checks, all must pass)

returns DataFrame · exact 4-column schema · ≥ 6 models · no duplicate model
names · all six required algorithms present · metrics in `[0,1]` · every model
auc > 0.82 · best auc ≥ 0.88 · best f1 ≥ 0.80 · sorted by auc descending · an
ensemble in the top-3 · **reported `random_forest` auc matches an independent
re-train** (within 0.02 — defeats hardcoded tables).

## Rules

- **Framework-first**: every model trains via `TrainingPipeline.train()` — raw
  `sklearn.fit()` is BLOCKED.
- **Polars only.** Load via `shared.MLFPDataLoader`. Deterministic (seeds fixed).
- `solve()` runs the async pipeline internally via `asyncio.run` and returns a
  plain DataFrame (the grader calls `solve()` synchronously).
