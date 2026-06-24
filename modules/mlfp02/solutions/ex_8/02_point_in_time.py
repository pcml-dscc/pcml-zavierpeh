# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 8.2: Point-in-Time Retrieval — Leakage Prevention
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Retrieve features at specific points in time to prevent leakage
#   - Demonstrate how future data inflates model performance
#   - Compare FeatureStore PIT retrieval with Polars temporal filtering
#   - Quantify the impact of leakage on regression coefficients
#   - Apply PIT correctness to Singapore property market forecasting
#
# PREREQUISITES: Exercise 8.1 (FeatureSchema v1, feature computation)
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — what data leakage is and why it destroys models
#   2. Build — PIT retrieval via FeatureStore and Polars fallback
#   3. Train — compare leaked vs correct model performance
#   4. Visualise — side-by-side leaked vs correct predictions
#   5. Apply — mortgage approval model for DBS Bank Singapore
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from shared.mlfp02.ex_8 import (
    OUTPUT_DIR,
    as_of,
    build_schema_v1,
    compute_v1_features,
    fit_ols,
    load_hdb_resale,
    prepare_design_matrix,
    setup_feature_store,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — What Data Leakage Is and Why It Destroys Models
# ════════════════════════════════════════════════════════════════════════
# Data leakage occurs when information from the future (or from the test
# set) bleeds into the training data. The model learns patterns that
# will not exist at prediction time, producing artificially high
# validation metrics and catastrophic production failures.
#
# Three common leakage types:
#
#   1. TEMPORAL LEAKAGE — Using 2024 transaction prices to train a model
#      that predicts 2023 prices. The model "knows" the future.
#
#   2. TARGET LEAKAGE — Using a feature derived from the target variable.
#      Example: using "price_per_sqm" (which includes resale_price) to
#      predict resale_price. Circular reasoning.
#
#   3. TRAIN-TEST CONTAMINATION — Same transaction appears in both
#      training and test sets (duplication or random split ignoring time).
#
# Point-in-time (PIT) retrieval prevents temporal leakage by enforcing
# a hard cutoff: at prediction time T, only data from before T is used.
#
# Singapore analogy: URA publishes quarterly property price indices.
# If you build a Q1-2024 forecast using data up to Q3-2024, your
# "forecast" is just reading the answer sheet. PIT retrieval is the
# discipline of covering the answer sheet during the exam.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Point-in-Time retrieval via FeatureStore and Polars
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Exercise 8.2 — Point-in-Time Retrieval: Leakage Prevention")
print("=" * 70)

# --- 2a. Prepare features ---
hdb = load_hdb_resale()
features_v1 = compute_v1_features(hdb)
property_schema_v1 = build_schema_v1()

print(f"\n  Features computed: {features_v1.shape[0]:,} rows")
print(
    f"  Date range: {features_v1['transaction_date'].min()} to "
    f"{features_v1['transaction_date'].max()}"
)

# --- 2b. FeatureStore PIT retrieval ---
factory, fs, tracker, has_backend = asyncio.run(setup_feature_store())

CUTOFF_2023 = datetime(2023, 1, 1)
CUTOFF_2024 = datetime(2024, 1, 1)

if has_backend:
    try:

        async def pit_demo():
            await fs.register_features(property_schema_v1)
            await fs.store(features_v1, property_schema_v1)
            f_2023 = await fs.get_training_set(
                schema=property_schema_v1,
                start=datetime(2000, 1, 1),
                end=CUTOFF_2023,
            )
            f_2024 = await fs.get_training_set(
                schema=property_schema_v1,
                start=datetime(2000, 1, 1),
                end=CUTOFF_2024,
            )
            return f_2023, f_2024

        features_2023, features_2024 = asyncio.run(pit_demo())
        delta = features_2024.height - features_2023.height
        print(f"\n  [FeatureStore PIT]")
        print(f"    Features as of 2023-01-01: {features_2023.height:,} rows")
        print(f"    Features as of 2024-01-01: {features_2024.height:,} rows")
        print(f"    2023 transactions added: {delta:,}")
    except Exception as e:
        has_backend = False
        print(f"  [Skipped: PIT retrieval ({type(e).__name__}: {e})]")

if not has_backend:
    # Polars fallback — same logic, same guarantees
    features_2023 = as_of(features_v1, CUTOFF_2023)
    features_2024 = as_of(features_v1, CUTOFF_2024)
    delta = features_2024.height - features_2023.height
    print(f"\n  [Polars PIT]")
    print(f"    Features before 2023: {features_2023.height:,}")
    print(f"    Features before 2024: {features_2024.height:,}")
    print(f"    Delta: {delta:,}")

print(f"\n  --- Why Point-in-Time Matters ---")
print(f"  To predict prices at T=2023-01-01, you must ONLY use data before T.")
print(f"  Using 2024 data would leak future info -> over-optimistic evaluation.")


# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert features_2023.height > 0, "Task 2: must have pre-2023 features"
assert (
    features_2024.height > features_2023.height
), "Task 2: 2024 cutoff must include more rows than 2023"
print("\n[ok] Checkpoint 1 passed — PIT retrieval demonstrated\n")

# INTERPRETATION: The delta between 2023 and 2024 cutoffs represents
# an entire year of transactions that would LEAK into a 2023 model
# if we used a naive "use all data" approach. In production, this
# means the model would appear to predict perfectly for 2023 but fail
# on genuinely future data.


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Compare leaked vs PIT-correct model performance
# ════════════════════════════════════════════════════════════════════════
# We build two OLS models:
#   - CORRECT: trained on data before 2023, evaluated on 2023 data
#   - LEAKED:  trained on ALL data including 2023, evaluated on 2023
#
# The leaked model will show inflated R² because it already "knows"
# the 2023 prices it's being asked to predict.

print("--- Comparing Leaked vs Correct Models ---")

# Correct model: train on pre-2023, test on 2023
FEATURE_COLS = [
    "floor_area_sqm",
    "storey_midpoint",
    "remaining_lease_years",
]

train_correct = features_2023.drop_nulls(subset=[*FEATURE_COLS, "resale_price"])
test_2023 = features_v1.filter(
    (pl.col("transaction_date") >= pl.lit(CUTOFF_2023.date()))
    & (pl.col("transaction_date") < pl.lit(CUTOFF_2024.date()))
).drop_nulls(subset=[*FEATURE_COLS, "resale_price"])

# Build design matrices
X_train = np.column_stack(
    [
        np.ones(train_correct.height),
        train_correct.select(FEATURE_COLS).to_numpy().astype(np.float64),
    ]
)
y_train = train_correct["resale_price"].to_numpy().astype(np.float64)

X_test = np.column_stack(
    [
        np.ones(test_2023.height),
        test_2023.select(FEATURE_COLS).to_numpy().astype(np.float64),
    ]
)
y_test = test_2023["resale_price"].to_numpy().astype(np.float64)

# Correct model: train on pre-2023
ols_correct = fit_ols(X_train, y_train)
y_pred_correct = X_test @ ols_correct["beta"]
resid_correct = y_test - y_pred_correct
rmse_correct = float(np.sqrt(np.mean(resid_correct**2)))
ss_res_correct = float(np.sum(resid_correct**2))
ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
r2_test_correct = 1 - ss_res_correct / ss_tot

# Leaked model: train on ALL data (including 2023 test set)
X_all = np.column_stack(
    [
        np.ones(features_v1.height),
        features_v1.drop_nulls(subset=[*FEATURE_COLS, "resale_price"])
        .select(FEATURE_COLS)
        .to_numpy()
        .astype(np.float64),
    ]
)
y_all = (
    features_v1.drop_nulls(subset=[*FEATURE_COLS, "resale_price"])["resale_price"]
    .to_numpy()
    .astype(np.float64)
)
ols_leaked = fit_ols(X_all, y_all)
y_pred_leaked = X_test @ ols_leaked["beta"]
resid_leaked = y_test - y_pred_leaked
rmse_leaked = float(np.sqrt(np.mean(resid_leaked**2)))
ss_res_leaked = float(np.sum(resid_leaked**2))
r2_test_leaked = 1 - ss_res_leaked / ss_tot

print(f"\n  Correct model (trained pre-2023, tested on 2023):")
print(f"    Training R²: {ols_correct['r2']:.4f}")
print(f"    Test RMSE:   ${rmse_correct:,.0f}")
print(f"    Test R²:     {r2_test_correct:.4f}")

print(f"\n  Leaked model (trained on ALL data, tested on 2023):")
print(f"    Training R²: {ols_leaked['r2']:.4f}")
print(f"    Test RMSE:   ${rmse_leaked:,.0f}")
print(f"    Test R²:     {r2_test_leaked:.4f}")

leakage_gap = rmse_correct - rmse_leaked
print(f"\n  Leakage gap: ${leakage_gap:,.0f} RMSE difference")
print(f"  The leaked model APPEARS ${abs(leakage_gap):,.0f} better per prediction")
print(f"  but this improvement is fictional — it used future data.")


# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert ols_correct["r2"] > 0.1, "Task 3: correct model R² should be reasonable"
assert rmse_correct > 0, "Task 3: RMSE must be positive"
print("\n[ok] Checkpoint 2 passed — leaked vs correct comparison complete\n")

# INTERPRETATION: Even a small RMSE gap from leakage is dangerous.
# In production, the leaked model's confidence intervals are too
# narrow (it thinks it knows more than it does), and every decision
# based on those intervals is over-confident.


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Side-by-side leaked vs correct predictions
# ════════════════════════════════════════════════════════════════════════

print("--- Visualising Leakage Impact ---")

rng = np.random.default_rng(42)
n_sample = min(2000, len(y_test))
idx = rng.choice(len(y_test), size=n_sample, replace=False)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        f"CORRECT (PIT) — RMSE ${rmse_correct:,.0f}",
        f"LEAKED — RMSE ${rmse_leaked:,.0f}",
    ],
)

# Correct model
fig.add_trace(
    go.Scatter(
        x=y_test[idx].tolist(),
        y=y_pred_correct[idx].tolist(),
        mode="markers",
        marker={"size": 3, "opacity": 0.4, "color": "steelblue"},
        name="Correct",
    ),
    row=1,
    col=1,
)
# Leaked model
fig.add_trace(
    go.Scatter(
        x=y_test[idx].tolist(),
        y=y_pred_leaked[idx].tolist(),
        mode="markers",
        marker={"size": 3, "opacity": 0.4, "color": "firebrick"},
        name="Leaked",
    ),
    row=1,
    col=2,
)

# Perfect prediction lines
for col in [1, 2]:
    fig.add_trace(
        go.Scatter(
            x=[float(y_test.min()), float(y_test.max())],
            y=[float(y_test.min()), float(y_test.max())],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            showlegend=False,
        ),
        row=1,
        col=col,
    )

fig.update_layout(
    title="Data Leakage Impact — Actual vs Predicted (2023 Test Set)",
    height=500,
    width=1000,
)
fig.update_xaxes(title_text="Actual Price ($)", row=1, col=1)
fig.update_xaxes(title_text="Actual Price ($)", row=1, col=2)
fig.update_yaxes(title_text="Predicted Price ($)", row=1, col=1)

fig.write_html(str(OUTPUT_DIR / "02_leakage_comparison.html"))
print(f"\n  Saved: {OUTPUT_DIR / '02_leakage_comparison.html'}")

# Residual comparison
fig2 = go.Figure()
fig2.add_trace(
    go.Histogram(
        x=resid_correct[idx].tolist(),
        name=f"Correct (RMSE ${rmse_correct:,.0f})",
        opacity=0.6,
        nbinsx=50,
    )
)
fig2.add_trace(
    go.Histogram(
        x=resid_leaked[idx].tolist(),
        name=f"Leaked (RMSE ${rmse_leaked:,.0f})",
        opacity=0.6,
        nbinsx=50,
    )
)
fig2.update_layout(
    title="Residual Distributions — Correct vs Leaked",
    xaxis_title="Residual ($)",
    yaxis_title="Count",
    barmode="overlay",
)
fig2.write_html(str(OUTPUT_DIR / "02_residual_comparison.html"))
print(f"  Saved: {OUTPUT_DIR / '02_residual_comparison.html'}")


# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 3 passed — leakage visualisations saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Mortgage Approval Model for DBS Bank Singapore
# ════════════════════════════════════════════════════════════════════════
# Scenario: DBS Bank builds a mortgage risk model using HDB resale
# prices. The model estimates "loan-to-value" (LTV) ratios to decide
# mortgage approval. If the model leaks future prices, it
# overestimates property values, approves mortgages for overvalued
# flats, and the bank absorbs losses when prices correct.
#
# DBS processes ~8,000 HDB mortgages per month in Singapore. Average
# mortgage: S$350,000. If future-price leakage inflates valuations
# by 5%, the bank over-lends by ~$17,500 per mortgage.
#
#   Monthly exposure: 8,000 * $17,500 = S$140M in excess lending
#   If 3% of these correct downward: S$4.2M monthly write-offs
#   Annual impact: ~S$50M
#
# PIT retrieval eliminates this by ensuring the valuation model only
# uses prices that existed BEFORE the mortgage application date.

print("=== APPLY: DBS Mortgage Approval — PIT-Correct Valuation ===")
print()
print("  Scenario: DBS Bank HDB mortgage risk model")
print()
print("  WITHOUT PIT retrieval:")
print("    - Model trained on all data, including future prices")
print("    - Overestimates property values by ~5%")
print("    - Over-lends S$17,500 per mortgage on average")
print("    - 8,000 mortgages/month * 3% correction rate")
print("    - Annual write-off exposure: ~S$50M")
print()
print("  WITH PIT retrieval:")
print("    - Model only uses prices before application date")
print("    - Conservative valuations that reflect actual market")
print("    - No systematic over-lending from future data")
print()
print(f"  Your PIT model performance on 2023 data:")
print(f"    R² = {r2_test_correct:.4f} (honest, no leakage)")
print(f"    RMSE = ${rmse_correct:,.0f} (true prediction error)")
print(f"    vs leaked RMSE = ${rmse_leaked:,.0f} (fictionally low)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n[ok] Checkpoint 4 passed — PIT mortgage application demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [ok] Point-in-time retrieval: hard temporal cutoffs for training data
  [ok] Leakage detection: comparing leaked vs correct model performance
  [ok] FeatureStore PIT API: get_training_set(start, end) enforcement
  [ok] Polars temporal filtering: as_of() fallback for Polars-only mode
  [ok] Quantified leakage impact: RMSE gap and dollar-value consequences

  KEY INSIGHT: A model that "works great in development" but fails in
  production almost always has a leakage bug. PIT retrieval is the
  structural fix — it makes leakage impossible, not just unlikely.

  Next: In 03_rolling_features.py, you'll extend the feature schema
  with rolling market statistics that capture town-level price trends
  and transaction volumes over time.
"""
)
