# MLFP01 — Task 4: Profile, Clean & Integrate with DataExplorer

**Weight**: 30 marks · **Difficulty**: Hard · **Dataset**: `data/mlfp01/economic_indicators.csv` (401 rows, 8 columns)

## Scenario

Singapore's quarterly macro indicators arrive in a single messy file: numbers
stored as text with thousands separators, three incompatible period formats,
and scattered missing values. Use the kailash-ml **`DataExplorer`** engine to
_discover_ the quality issues, fix them deterministically, and then prove the
fix worked by re-profiling.

Implement `solve() -> dict` returning **exactly** these keys:

```python
{
    "cleaned": pl.DataFrame,    # the cleaned quarterly table (8 columns)
    "raw_alert_count": int,     # DataExplorer alerts on the RAW quarterly slice
    "clean_alert_count": int,   # DataExplorer alerts on your cleaned frame
}
```

## Required pipeline

1. Keep only rows where `period_type == "quarterly"` (101 rows).
2. Parse `period` into `period_year` (Int) and `period_quarter` (Int, 1–4). It
   appears in **three** formats you must all handle:
   - `"Q1 2000"` · `"2001-Q1"` · `"2001-2"` (year-dash-quarternumber)
3. `tourist_arrivals` is stored as **text**, some with thousands separators
   (`"5,246,242"`). Strip separators and cast to `Int64`.
4. Impute `inflation_rate` and `trade_balance_sgd_bn` nulls with the **quarterly
   median** of each column.
5. `cleaned` = these **8 columns in this exact order**, sorted by
   `[period_year, period_quarter]`:

   ```
   period_year, period_quarter, gdp_growth_pct, unemployment_rate,
   inflation_rate, trade_balance_sgd_bn, property_price_index, tourist_arrivals
   ```

6. Profile the **raw quarterly slice** and your **cleaned frame** with
   `DataExplorer` (`await explorer.profile(df)`; the count is
   `len(profile.alerts)`). Your cleaning must **reduce** the alert count.

## Visible sanity checks

- `result["cleaned"].shape == (101, 8)`
- `result["cleaned"]["tourist_arrivals"].dtype == pl.Int64`
- `result["clean_alert_count"] < result["raw_alert_count"]`
- no nulls in `inflation_rate`, `trade_balance_sgd_bn`, `tourist_arrivals`

## Grading (automated, all checks must pass)

returns the dict with the 3 keys · `cleaned` is a DataFrame · exact 8-column
schema · row count 101 · `tourist_arrivals` is Int64 · quarter range 1–4 ·
no nulls in imputed columns · `tourist_arrivals` values correct · `inflation_rate`
values correct · (year, quarter) keys match reference · `raw_alert_count` matches
the independently-measured ground truth · cleaning reduced the alert count.

## Rules

- **Polars only** for data — no pandas. Use the kailash-ml `DataExplorer` engine
  for profiling. Load via `shared.MLFPDataLoader`. Deterministic — no sampling.
