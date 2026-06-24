# MLFP02 — Task 3: Regression Modelling & Interpretation

**Weight**: 25 marks · **Difficulty**: Hard · **Dataset**: `data/mlfp02/sg_credit_scoring.parquet` (100,000 rows, 36 columns)

## Scenario

A Singapore lender wants to understand what drives the **loan amount** it
extends, and separately what drives **default**. You will fit a multiple linear
regression with full inference, test whether non-linear terms are justified, and
fit a logistic model for odds-ratio interpretation. Everything is closed-form
and deterministic — no train/test split, no random solver.

Implement `solve() -> dict`.

## Preprocessing (no rows dropped — `n_obs == 100000`)

- `income_imp` = `income_sgd` with nulls (30,000 of them) filled by the
  **median** of `income_sgd`.
- `edu_ord` = `education` ordinal-encoded:
  `primary→1, secondary→2, diploma→3, degree→4, postgraduate→5` (Float64).

## Required computation

1. **OLS — predict `loan_amount_sgd`** from `OLS_FEATURES`
   = `[income_imp, age, employment_years, debt_to_income, credit_age_years,
   num_dependents, edu_ord]`. **Standardise each predictor** (z-score,
   population sd `ddof=0`), prepend an intercept column of ones, and solve
   `beta` with `np.linalg.lstsq` against the **raw** target. Standardising keeps
   the design matrix well-conditioned once the squared term is added.
2. **Inference** — `r_squared`, `adj_r_squared`, `f_statistic`, `f_p_value`,
   plus per-coefficient `t_stats` and two-sided `p_values`:
   - `sigma2 = rss / (n − p)`, `se = sqrt(diag(sigma2 · (XᵀX)⁻¹))`
   - `t = beta / se`, `p = 2 · stats.t.sf(|t|, df=n−p)`
   - `f_statistic = (r² / (p−1)) / ((1−r²) / (n−p))`
   - `coefficients`, `t_stats`, `p_values` are dicts keyed by feature name plus
     `"intercept"`.
3. **Partial F-test** — add two terms built from the **standardised** base
   columns: `income_std²` and `age_std · employment_std`. Refit, then with
   `q = 2`:
   `partial_f = ((rss − rss_full)/q) / (rss_full/(n − p_full))`,
   `partial_f_p_value = stats.f.sf(partial_f, q, n − p_full)`,
   `delta_r_squared = r²_full − r²`. (Expect: significant F but a **negligible**
   ΔR² — significance is not the same as practical importance.)
4. **Logistic — predict `default`** from `LOGIT_FEATURES`
   = `[credit_utilization, num_late_payments, previous_defaults, debt_to_income,
   num_hard_inquiries]`. Standardise the features, prepend an intercept, fit by
   Newton-Raphson / IRLS to convergence. Report `odds_ratios = exp(beta)` (dict
   incl. `"intercept"`) and `strongest_logit_predictor` = the feature (excluding
   intercept) with the largest `|beta|`.

## Return contract — `dict` with these exact keys

```
n_obs (int), coefficients (dict), t_stats (dict), p_values (dict),
r_squared, adj_r_squared, f_statistic, f_p_value,
partial_f, partial_f_p_value, delta_r_squared,
odds_ratios (dict), strongest_logit_predictor (str)
```

The three OLS dicts are keyed by `"intercept"` + the 7 `OLS_FEATURES`;
`odds_ratios` is keyed by `"intercept"` + the 5 `LOGIT_FEATURES`.

## Visible sanity checks

- `0.85 < r_squared < 0.87` (loan amount is highly predictable here)
- `partial_f_p_value < 0.01` **but** `delta_r_squared < 1e-3`
- `odds_ratios["debt_to_income"] > 1` (more leverage → higher default odds)
- `strongest_logit_predictor in LOGIT_FEATURES`

## Grading (12 automated checks, all must pass)

return type is `dict` · all 13 keys · `n_obs` · OLS coefficients · t-stats ·
p-values · R² · adjusted R² · F-statistic · partial F · partial-F p-value ·
ΔR² (negligible) · odds ratios · strongest logistic predictor.

## Rules

- **Polars only** for data wrangling — no pandas. `numpy` / `scipy.stats` for the
  linear algebra and distributions. **No raw `sklearn` model training** — these
  are closed-form statistical fits.
- Load via `shared.MLFPDataLoader`.
- Fully **deterministic** — closed-form OLS and a convex logistic MLE.
