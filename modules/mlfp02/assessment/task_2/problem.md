# MLFP02 — Task 2: Hypothesis Testing, Bootstrap & CUPED

**Weight**: 25 marks · **Difficulty**: Hard · **Dataset**: `data/mlfp02/experiment_data.parquet` (500,000 rows, 9 columns)

## Scenario

You have the same e-commerce experiment as Task 1. The team now wants a
rigorous read on whether `treatment_a` beats `control` on the continuous
`metric_value`, with a confidence interval that survives a sceptic, plus a
variance-reduction pass (CUPED) using the pre-experiment covariate. Finally,
the company ran five tests at once and needs a correct multiple-testing
verdict.

Work the **control + treatment_a** cohort only.

Implement `solve() -> dict`.

## Required computation

1. **Welch t-test** — `metric_value`, treatment_a vs control, unequal variances:
   `welch_t, welch_p = scipy.stats.ttest_ind(t, c, equal_var=False)` (two-sided).
   `mean_diff = mean(t) − mean(c)`. Build `t` and `c` as float numpy arrays in
   file order.
2. **Seeded percentile bootstrap** of `mean_diff` — follow this protocol
   **exactly** (it is bit-reproducible and graded tightly):

   ```python
   rng = np.random.default_rng(2024)
   diffs = np.empty(2000)
   for b in range(2000):
       bt = rng.choice(t, size=t.size, replace=True)   # treatment FIRST
       bc = rng.choice(c, size=c.size, replace=True)   # control SECOND
       diffs[b] = bt.mean() - bc.mean()
   boot_ci_low, boot_ci_high = np.percentile(diffs, [2.5, 97.5])
   ```

3. **CUPED** — using `pre_metric_value` as covariate over the full cohort:
   - `cuped_theta = Cov(metric, pre) / Var(pre)` with `ddof=1`
   - `metric_adj = metric − theta · (pre − mean(pre))`
   - `var_metric = Var(metric, ddof=1)`, `var_adj = Var(metric_adj, ddof=1)`
   - `cuped_var_reduction = 1 − var_adj / var_metric`
4. **CUPED-adjusted test** — re-run the Welch test on `metric_adj`
   (treatment_a vs control): `welch_t_cuped`, `welch_p_cuped`. The adjusted
   `|t|` must come out **larger** than the unadjusted `|t|`.
5. **Multiple-testing correction** — over `MT_P_VALUES = [0.03, 0.012, 0.04,
   0.65, 0.009]` at `alpha = 0.05`:
   - `bonferroni_n_sig` = count of `p < alpha / m` (m = 5)
   - `bh_n_sig` = Benjamini-Hochberg step-up count: sort the p-values, set
     `threshold_i = alpha · i / m`, reject every hypothesis up to the largest
     rank `i` with `p_(i) ≤ threshold_i`.

## Return contract — `dict` with these exact keys

```
welch_t, welch_p, mean_diff, boot_ci_low, boot_ci_high,
cuped_theta, var_metric, var_adj, cuped_var_reduction,
welch_t_cuped, welch_p_cuped, bonferroni_n_sig (int), bh_n_sig (int)
```

## Visible sanity checks

- `boot_ci_low > 0` — the bootstrap CI excludes zero (significant uplift)
- `0 < cuped_var_reduction < 1`
- `abs(welch_t_cuped) > abs(welch_t)` — CUPED lowers the noise floor
- `bh_n_sig >= bonferroni_n_sig` — BH is at least as powerful

## Grading (12 automated checks, all must pass)

return type is `dict` · all 13 keys · Welch test · mean diff · seeded bootstrap
CI (tight) · CI excludes zero · CUPED theta · CUPED variances · CUPED reduction
(in range) · CUPED-adjusted test · CUPED increases power · Bonferroni count ·
BH count · BH ≥ Bonferroni.

## Rules

- **Polars only** for data wrangling — no pandas. `numpy` / `scipy.stats` are
  allowed for the statistics.
- Load via `shared.MLFPDataLoader`.
- The only randomness is the bootstrap, fixed by `np.random.default_rng(2024)`
  and the exact resample order above. Match it precisely.
