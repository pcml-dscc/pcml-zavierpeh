# MLFP03 — Task 1: Feature Engineering & Leakage-Free Selection

**Weight**: 20 marks · **Difficulty**: Hard · **Dataset**: `data/mlfp03/ecommerce_customers.parquet` (50,000 rows, 16 columns)

## Scenario

A Southeast-Asia e-commerce operator wants to target customers for a paid
**premium-membership upsell**. You are handed the raw customer table. Your job:
engineer business-meaningful features, then let the kailash-ml
`FeatureEngineer` rank them so the modelling team starts from the highest-signal
inputs — **without leaking the test set into the ranking**.

Implement `solve() -> dict`.

## The derived target — `premium_response`

The native `churned` column is a near-deterministic function of recency, so it
is useless for teaching. Instead the assessment derives a realistic target:
whether a customer accepts the premium upsell. It is built from a documented
logit over satisfaction, loyalty, spend, returns, plus a
loyalty x high-satisfaction **interaction** and seeded Gaussian noise (≈25%
positive — a 3:1 imbalance). **The full target code is given to you in
`_load_base()` — keep it byte-for-byte so your output matches the grader.**

## Required engineered features (exact formulas)

Add these six columns. `loyalty_int` is `loyalty_member` cast to `Int64`
(already created in `_load_base()`).

| Column | Formula |
| ------ | ------- |
| `revenue_per_order`    | `total_revenue / order_count` |
| `returns_per_order`    | `num_returns / order_count` |
| `is_satisfied`         | `Int64(satisfaction_score >= 4)` |
| `loyal_and_satisfied`  | `loyalty_int * Int64(satisfaction_score >= 4)` |
| `tenure_years`         | `customer_tenure_days / 365.0` |
| `spend_per_tenure_day` | `total_revenue / customer_tenure_days` |

## Candidate pool (14 features)

```
BASE (8): total_revenue, order_count, avg_order_value, days_since_last_order,
          customer_tenure_days, satisfaction_score, num_returns, loyalty_int
ENGINEERED (6): the six above
```

## Required pipeline

1. **Engineer** the six features (exact formulas above).
2. **Assemble** `feature_matrix` = the 14 candidate columns **plus** the target
   `premium_response`, in original row order (no shuffle). Customer IDs, raw
   `review_text`, `ltv_tier`, `product_categories`, and the native `churned`
   label MUST NOT appear.
3. **Select leakage-free**: take the **first 75%** of rows (the train split) and
   rank with `kailash_ml` `FeatureEngineer.select(..., method="importance",
   top_k=8)`. Build the candidate set with `GeneratedFeatures` /
   `GeneratedColumn` (originals = the 8 base, generated = the 6 engineered).
4. **Return** the dict described below.

## Exact return contract

```python
{
  "feature_matrix":    pl.DataFrame,  # 14 candidate cols + "premium_response", 10,000 rows
  "engineered_columns": list[str],    # the 6 engineered names
  "selected_features":  list[str],    # top-8 features by importance (leakage-free)
  "target_column":     "premium_response",
}
```

## Visible sanity checks

After a correct implementation:

- `result["feature_matrix"].shape == (10000, 15)`
- positive rate `result["feature_matrix"]["premium_response"].mean()` ≈ `0.254`
- `len(result["selected_features"]) == 8`
- the **top-ranked** feature is `loyal_and_satisfied` (the engineered
  interaction) — engineering it surfaces the single strongest signal.

## Performance target

A RandomForest trained on **only** your `selected_features` clears
**ROC-AUC ≥ 0.84** on a held-out split (the reference clears ≈ 0.87).

## Grading (12 automated checks, all must pass)

return dict shape · types valid · no id/text/native-label leakage · engineered
names exact · all 14 candidates present · derived target re-derived
element-wise · engineered interaction + ratio correct · engineered spend +
revenue correct · selection shape (exactly 8, valid, target excluded) · top
driver is the interaction or satisfaction · selected overlaps independently
re-derived importance top-8 by ≥ 6 · selected features clear the AUC floor.

## Rules

- **Polars only** — no pandas. Framework-first: selection via `FeatureEngineer`.
- Load via `shared.MLFPDataLoader`. Fully deterministic (seeds fixed).
- Selection MUST be fit on the train split only — fitting on all rows leaks
  the test distribution and fails the leakage checks.
