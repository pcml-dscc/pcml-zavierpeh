# MLFP03 — Task 3: Evaluation, Class Imbalance & Interpretability

**Weight**: 25 marks · **Difficulty**: Hard · **Dataset**: `data/mlfp03/ecommerce_customers.parquet` (50,000 rows, 16 columns)

## Scenario

Your premium-upsell model will drive a marketing spend, so the business cares
about **catching the responders** (the ~25% positive minority), not raw
accuracy. A model that predicts "no one responds" is 75% accurate and useless.
You must: evaluate per class, show that class-imbalance handling lifts
minority-class recall, and explain *which features* drive the model.

Implement `solve() -> dict`.

## Data contract (deterministic — given to you)

- First **10,000** rows; derived target `premium_response` (~25% positive — the
  minority/positive class). Code given in `_model_frame()`; keep it intact.
- 8 base features (no engineered interaction this time — the point is honest
  evaluation, not feature work): `satisfaction_score, avg_order_value,
  num_returns, order_count, loyalty_int, total_revenue, days_since_last_order,
  customer_tenure_days`.
- `TrainingPipeline` holdout, `test_size=0.25`. `_holdout_test()` reproduces the
  exact test split so `km.diagnose` lines up with the engine's evaluation.

## Required pipeline (framework-first)

1. **Train two RandomForests** via `TrainingPipeline.train()`:
   - **baseline**: `{n_estimators:150, random_state:42, n_jobs:-1}`
   - **balanced**: same **plus** `class_weight="balanced"`
2. **Evaluate per class** with `km.diagnose(model, kind="classical_classifier",
   data=(X_test, y_test), show=False)`. The minority class is key `"1.0"`:
   `report.per_class["1.0"]["recall"]`. Macro recall and accuracy are in
   `report.metrics` (`recall_macro`, `accuracy`).
3. **Interpret** the balanced model with `ModelExplainer` (SHAP):
   `explain_global(max_display=6)["feature_importance"]` returns a dict ordered
   by importance — take the first 6 keys as `top_features`.

## Exact return contract

```python
{
  "baseline_minority_recall": float,   # class "1.0" recall, no balancing
  "balanced_minority_recall": float,   # class "1.0" recall, class_weight balanced
  "baseline_recall_macro":   float,
  "balanced_recall_macro":   float,
  "baseline_accuracy":       float,
  "balanced_accuracy":       float,
  "roc_auc":                 float,    # balanced model held-out ROC-AUC
  "top_features":            list[str],# top-6 features by SHAP global importance
  "n_features":              8,
}
```

## Visible sanity checks

After a correct implementation:

- `balanced_minority_recall` ≈ `0.72` **>** `baseline_minority_recall` ≈ `0.62`
- `balanced_accuracy` ≈ `0.84` **<** `baseline_accuracy` ≈ `0.86`
  (the accuracy-vs-recall tradeoff is the whole lesson)
- `balanced_recall_macro` **>** `baseline_recall_macro`
- `roc_auc` ≈ `0.90`
- `top_features[:3]` contains `satisfaction_score` (the dominant driver)

## Performance target

Balanced-model **minority-class recall ≥ 0.68** and held-out **ROC-AUC ≥ 0.85**.

## Grading (13 automated checks, all must pass)

returns dict with all keys · values finite & in `[0,1]` · balancing lifts
minority recall (> baseline + 0.02) · balanced minority recall ≥ 0.68 ·
baseline minority recall in `[0.55, 0.67]` · macro recall improves · accuracy
tradeoff (balanced ≤ baseline) · roc_auc ≥ 0.85 · `top_features` shape (6 valid
names) · SHAP surfaces `satisfaction_score` in the top-3 · `n_features == 8` ·
**baseline + balanced minority recall each match an independent re-train**
(within 0.03 — defeats hardcoded dicts).

## Rules

- **Framework-first**: training via `TrainingPipeline`, evaluation via
  `km.diagnose`, interpretability via `ModelExplainer`. Raw `sklearn.fit()` and
  hand-rolled metrics are BLOCKED.
- **Polars only.** Load via `shared.MLFPDataLoader`. Deterministic (seeds fixed).
- `solve()` wraps the async work in `asyncio.run` and returns a plain dict.
