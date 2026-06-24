# MLFP03 — Task 4: Production Pipeline — Registry, Drift, Deploy

**Weight**: 30 marks · **Difficulty**: Hard · **Dataset**: `data/mlfp03/ecommerce_customers.parquet` (50,000 rows, 16 columns)

## Scenario

The premium-upsell model is ready to ship. Production ML is not "the pickle on
S3" — it needs a **versioned registry** with an audit-grade promotion trail and
**drift monitoring** so you find out the week the world changes, not at the
quarterly review. Build the full lifecycle: train -> register -> promote ->
monitor.

Implement `solve() -> dict`.

## Data contract (deterministic — given to you)

- First **10,000** rows; derived target `premium_response` (~25% positive).
  Code given in `_model_frame()`; keep it intact.
- 8 base features (same as Task 3).
- **Reference distribution** = first 7,500 feature rows. **Incoming batches** =
  the remaining 2,500 rows, evaluated twice:
  - `clean` — the raw 2,500 rows (same distribution as reference)
  - `shifted` — an economic-downturn shift built by `_shift_slice()` (spend
    x0.6, recency x1.5 + 60 days, satisfaction − 1). Keep it intact.

## Required pipeline (framework-first)

1. **Train** a LightGBM model via `TrainingPipeline.train()` —
   `model_class="lightgbm.LGBMClassifier"`, framework `lightgbm`,
   `{n_estimators:200, random_state:42, verbose:-1}`; EvalSpec metrics
   `["accuracy","f1","auc"]`, holdout, `test_size=0.25`. Training registers the
   model at **staging**.
2. **Promote** the registered version `staging -> production` with an audit
   reason, then `get_model(name, stage="production")` to confirm.
3. **Monitor**: arm a `DriftMonitor` (`psi_threshold=0.2`, `ks_threshold=0.05`),
   `set_reference_data(name, reference, BASE_FEATURES)`, then `check_drift` on
   the `clean` batch and on the `shifted` batch.
4. **Return** the dict below.

### Two databases (MUST)

Give the `ModelRegistry` and the `DriftMonitor` **separate SQLite files** — the
realistic production posture, since a model registry and a monitoring store are
distinct systems with independent lifecycles. Using fresh, separate files per
store also avoids reusing a stale database whose schema predates your installed
kailash-ml version.

## Exact return contract

```python
{
  "registered_version":       int,    # >= 1
  "production_stage":         "production",
  "reference_auc":            float,  # held-out ROC-AUC of the registered model
  "clean_drift_detected":     bool,   # False — same distribution, no alarm
  "shift_drift_detected":     bool,   # True  — downturn trips the monitor
  "n_drifted_features_clean": int,    # 0
  "n_drifted_features_shift": int,    # >= 3
  "shift_severity":           str,    # not "none"
}
```

## Visible sanity checks

After a correct implementation:

- `registered_version == 1`, `production_stage == "production"`
- `reference_auc ≈ 0.90`
- `clean_drift_detected is False` and `n_drifted_features_clean == 0`
  (no false alarm on a same-distribution batch)
- `shift_drift_detected is True`, `n_drifted_features_shift == 4`,
  `shift_severity == "severe"`

## Performance target

Registered model **ROC-AUC ≥ 0.85**; drift monitor flags the shifted batch
(≥ 3 features) while staying silent on the clean batch (0 features).

## Grading (11 automated checks, all must pass)

returns dict with all keys · registered_version ≥ 1 · promoted to production ·
reference_auc ≥ 0.85 · clean batch → no drift · shifted batch → drift detected ·
clean drifted-feature count == 0 · shifted drifted-feature count ≥ 3 · shift
severity signals drift · **reference_auc matches an independent re-train**
(within 0.02) · **clean/shift drift outcomes + shifted feature count match an
independent re-run** (defeats hardcoded dicts).

## Rules

- **Framework-first**: training via `TrainingPipeline`, versioning via
  `ModelRegistry`, monitoring via `DriftMonitor`. Raw SQL, manual pickling to
  disk, and hand-rolled PSI/KS are BLOCKED.
- **Polars only.** Load via `shared.MLFPDataLoader`. Deterministic (seeds fixed).
- `solve()` wraps the async work in `asyncio.run` and returns a plain dict.
- Use separate SQLite files for the registry and the drift monitor; clean them
  up before returning.
