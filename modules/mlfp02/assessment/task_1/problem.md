# MLFP02 — Task 1: Probability, Bayes & Experiment Validation

**Weight**: 20 marks · **Difficulty**: Hard · **Dataset**: `data/mlfp02/experiment_data.parquet` (500,000 rows, 9 columns)

## Scenario

A Southeast-Asia e-commerce team ran a multi-arm experiment on a new app flow.
The raw log has columns `user_id, experiment_group, metric_value,
pre_metric_value, revenue, timestamp, segment, platform, country`. Before you
trust any uplift number you must (a) reason about the conversion event
probabilistically, (b) prove the allocation was not silently corrupted, and
(c) update a prior belief about the treatment conversion rate.

Work the **primary A/B comparison only**: the `control` and `treatment_a`
arms (ignore `treatment_b` and `variant_c`). Define the conversion event
deterministically as `converted := metric_value >= 50.0`.

Implement `solve() -> dict`.

## Required computation

1. **Conditional probabilities** — over the control + treatment_a cohort:
   - `p_convert_overall` = P(converted)
   - `p_convert_control` = P(converted | control)
   - `p_convert_treatment` = P(converted | treatment_a)
2. **Bayes inversion** — `p_treatment_given_convert` = P(treatment_a | converted)
   = `P(converted|treatment_a) · P(treatment_a) / P(converted)`, where
   `P(treatment_a)` is treatment_a's share of the **cohort** (not the full file).
3. **Sample Ratio Mismatch** — the experiment was designed for a 50/50 split.
   Run a chi-square goodness-of-fit on the observed counts
   `[n_control, n_treatment]` against `expected = n_total / 2` per cell:
   - `srm_chi2` = Σ (observed − expected)² / expected
   - `srm_p_value` = `scipy.stats.chi2.sf(srm_chi2, df=1)`
   - `srm_flag` = `srm_p_value < 1e-3` (bool)
4. **Base-rate fallacy (fraud detector)** — pure scalar Bayes using the fixed
   constants `base=0.02`, `sensitivity=0.95`, `fpr=0.03`:
   `p_fraud_given_flagged = sens·base / (sens·base + fpr·(1−base))`.
5. **Beta-Binomial conjugate update** — on the **treatment_a** arm, with prior
   `Beta(2, 20)`: `successes = Σ converted`, `failures = n − successes`,
   posterior `Beta(2+successes, 20+failures)`:
   - `beta_post_alpha`, `beta_post_beta`
   - `posterior_mean` = α/(α+β)
   - `cred_int_low`, `cred_int_high` = `stats.beta.ppf(0.025/0.975, α, β)`

## Return contract — `dict` with these exact keys (full-precision floats)

```
p_convert_overall, p_convert_control, p_convert_treatment,
p_treatment_given_convert, srm_chi2, srm_p_value, srm_flag (bool),
p_fraud_given_flagged, beta_post_alpha, beta_post_beta,
posterior_mean, cred_int_low, cred_int_high
```

## Visible sanity checks

After a correct implementation:

- `p_convert_control < p_convert_treatment` (treatment lifts conversion)
- `srm_flag is True` — the arms are **not** 50/50 (a real SRM)
- `0.39 < p_fraud_given_flagged < 0.40` — only ~39% of flagged are truly fraud
- `cred_int_low < posterior_mean < cred_int_high`

## Grading (12 automated checks, all must pass)

return type is `dict` · all 13 keys present · three conditional probabilities ·
Bayes inversion · SRM chi-square · SRM p-value · SRM flag · fraud base-rate
Bayes · Beta posterior parameters · posterior mean · 95% credible interval.

## Rules

- **Polars only** for data wrangling — no pandas. `scipy.stats` is allowed for
  the chi-square / Beta quantiles.
- Load via `shared.MLFPDataLoader`.
- Fully **deterministic** — no randomness anywhere in this task.
