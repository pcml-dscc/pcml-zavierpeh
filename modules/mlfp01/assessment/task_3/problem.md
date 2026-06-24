# MLFP01 — Task 3: Window Functions & Price Trends

**Weight**: 20 marks · **Difficulty**: Hard · **Dataset**: `data/mlfp01/hdb_resale.parquet` (50,150 rows)

## Scenario

A housing analyst needs a per-town price-trend table to spot which estates are
heating up. This is a pure window-function exercise: the marks are in correct
**partitioning** (within town vs within year) and **ordering** (rolling and
lag computations require sorted years).

Implement `solve() -> pl.DataFrame`.

## Required computation

1. Derive `sale_year` (Int) from the `"YYYY-MM"` `month` string.
2. Aggregate to one row per **(town, sale_year)**:
   - `median_price` = median `resale_price`
   - `n_sales` = number of transactions
3. `yoy_pct` (Float) = `100 * (median_price − prev_year_median) / prev_year_median`,
   computed **within each town**, years in ascending order. The first year for
   each town has **no prior year → null**.
4. `rolling_3yr_avg` (Float) = trailing 3-year mean of `median_price`
   **within each town** (window size 3, `min_periods=1`, so years 1 and 2 are
   partial averages).
5. `price_rank_in_year` (Int) = rank of `median_price` **within each year**,
   descending, so `1` = the most expensive town that year (ties share the
   lower rank, i.e. `method="min"`).
6. Return these **7 columns in this exact order**, sorted by `[town, sale_year]`:

   ```
   town, sale_year, n_sales, median_price, yoy_pct, rolling_3yr_avg, price_rank_in_year
   ```

## Visible sanity checks

- `result.shape == (270, 7)` (27 towns × 10 years, 2015–2024)
- `result["yoy_pct"].null_count() == 27` (one first-year null per town)
- `price_rank_in_year` ranges `1`–`27`

## Grading (automated, all checks must pass)

return type · exact 7-column schema · row count 270 · (town, sale_year) keys
match reference · `median_price` correct · `n_sales` correct · `yoy_pct` correct
(values + first-year nulls) · `rolling_3yr_avg` correct · `price_rank_in_year`
correct · rank range 1–27.

## Rules

- **Polars only** — no pandas. Load via `shared.MLFPDataLoader`. Deterministic.
- Use Polars window expressions (`.over(...)`, `.shift(...)`, `.rolling_mean(...)`,
  `.rank(...)`) — do not hand-roll loops.
