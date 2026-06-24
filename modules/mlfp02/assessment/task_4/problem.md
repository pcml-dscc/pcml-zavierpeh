# MLFP02 — Task 4: Feature Engineering & Feature Store

**Weight**: 30 marks · **Difficulty**: Hard · **Datasets**: five ICU tables in `data/mlfp02/` — `icu_admissions.parquet` (8,000), `icu_patients.parquet` (5,000), `icu_vitals.parquet` (69,421), `icu_labs.parquet` (30,000), `icu_medications.parquet` (20,000)

## Scenario

You are building the feature layer for an ICU length-of-stay model. The raw data
is spread across five tables and is messy: vitals and labs only exist for a
fraction of admissions, lab values contain analyser junk (`HAEMOLYSED`,
`CLOTTED`, `<0.1`), unit casing is inconsistent, and dose strings look like
`"34.8MG"` and `"29.0 mg"`. Assemble a single **admission-level feature table**
(entity key = `admission_id`) that is model-ready: numeric, no nulls, with a
point-in-time anchor and the outcome label kept separate.

Implement `solve() -> pl.DataFrame`.

## Required feature table (one row per admission_id, 8,000 rows)

1. **Base + anchor** — from `icu_admissions`: `admission_id`, `diagnosis`,
   `icu_type`, `los_days` (the label), and `feature_timestamp` = `admit_time`
   parsed to `Datetime` (format `"%Y-%m-%d %H:%M:%S"`).
2. **Demographics** — left-join `icu_patients` on `patient_id`: `age`, `gender`,
   `bmi`. (About 2% of admissions have no matching patient.)
3. **Vitals** (group `icu_vitals` by `admission_id`):
   - `mean_heart_rate`, `mean_systolic_bp`, `min_spo2`, `max_temperature`
     (the aggregates ignore nulls inside a column)
   - `n_vitals` = number of vitals rows
4. **Labs** (parse, then group `icu_labs` by `admission_id`):
   - parse `value` → Float64 with `strict=False` so junk strings become null
   - lowercase `flag`
   - `n_labs` = number of lab rows, `n_abnormal_labs` = count where
     `flag == "abnormal"`, `mean_creatinine` = mean parsed `value` where
     `test_name == "Creatinine"`
5. **Medications** (parse, then group `icu_medications` by `admission_id`):
   - parse the leading number of `dose` via regex `([0-9]+\.?[0-9]*)` → Float64 mg
   - `n_distinct_drugs` = unique `drug_name`, `n_iv_meds` = count where
     `route == "IV"`, `total_dose_mg` = sum of parsed dose
6. **Imputation policy** (apply after the joins; medians computed **before**
   filling):
   - `gender` null → `"Unknown"`; `total_dose_mg` null → `0.0`
   - the five count columns (`n_vitals, n_labs, n_abnormal_labs,
     n_distinct_drugs, n_iv_meds`) null → `0` (Int64)
   - `age, bmi, mean_heart_rate, mean_systolic_bp, min_spo2, max_temperature,
     mean_creatinine` null → that **column's median** (Float64)

## Output schema (19 columns, this exact order), sorted by `admission_id`

```
admission_id, feature_timestamp, age, gender, bmi, diagnosis, icu_type,
mean_heart_rate, mean_systolic_bp, min_spo2, max_temperature,
n_vitals, n_labs, n_abnormal_labs, mean_creatinine,
n_distinct_drugs, n_iv_meds, total_dose_mg, los_days
```

## Visible sanity checks

- `result.shape == (8000, 19)` (one row per admission, no rows dropped)
- total null count across all columns is `0`
- `result.schema["feature_timestamp"] == pl.Datetime`
- about 1,000 admissions have `n_vitals > 0`; the rest are median-imputed
- `los_days` is unchanged from the source (it is the label, not a feature)

## Grading (12 automated checks, all must pass)

return type · exact 19-column schema · row count 8,000 · sorted by
`admission_id` · no nulls anywhere · `feature_timestamp` is Datetime · entity
keys match · demographics imputed correctly · vitals aggregates · labs aggregates
(value parsing + flag + Creatinine) · medication aggregates (dose parsing) ·
categoricals + `los_days` label preserved.

## Rules

- **Polars only** — no pandas. Load via `shared.MLFPDataLoader`.
- Feature engineering is **deterministic** — no sampling, no row dropping.
- No leakage: do **not** build features from `discharge_time` (post-outcome);
  `los_days` is the label and is carried through untouched.
