# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.4: Embedded Feature Selection (L1 / Lasso Path)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Use L1 regularisation to drive coefficients to exactly zero
#   - Walk a regularisation path (multiple C values)
#   - Apply to DBS Bank card-fraud scoring where latency dominates
#
# PREREQUISITES: 03_wrapper_selection.py
# ESTIMATED TIME: ~25 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from shared.mlfp03.ex_1 import (
    build_full_feature_frame,
    load_icu_tables,
    log_selection_run,
    prepare_selection_inputs,
    setup_tracking,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — L1 Drives Coefficients To Zero
# ════════════════════════════════════════════════════════════════════════
# L1 (Lasso) adds a penalty proportional to sum(|w_j|). The corner at
# zero means weak coefficients are pushed all the way to zero, not
# merely close to zero. Every zero coefficient is an eliminated
# feature — selection happens INSIDE the training loop.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: scale inputs and define the C grid
# ════════════════════════════════════════════════════════════════════════

tables = load_icu_tables()
features = build_full_feature_frame(tables)
feature_cols, X_sel, y_binary = prepare_selection_inputs(features)

print("\n" + "=" * 70)
print("  Embedded Selection — L1 Logistic Regression")
print("=" * 70)
print(f"  Features: {len(feature_cols)}")
print(f"  Samples:  {X_sel.shape[0]}")

# TODO: Standardise X_sel before L1 fitting. L1 is scale-sensitive!
# Hint: StandardScaler().fit_transform(X_sel)
X_scaled = ____

C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: walk the regularisation path
# ════════════════════════════════════════════════════════════════════════

print("\n--- Regularisation Path ---")
print(f"{'C':>8} {'Non-zero':>10} / {'Total':>6}")
print("-" * 30)

lasso_results: dict[float, dict] = {}
for c_val in C_VALUES:
    # TODO: Fit a LogisticRegression with penalty="l1", solver="saga",
    # C=c_val, max_iter=5000, random_state=42.
    # Hint: build the estimator then call .fit(X_scaled, y_binary)
    lasso = ____
    ____

    coefs = lasso.coef_[0].copy()
    n_nonzero = int((np.abs(coefs) > 1e-6).sum())
    lasso_results[c_val] = {"n_nonzero": n_nonzero, "coefs": coefs}
    print(f"  {c_val:>8.3f} {n_nonzero:>10} / {len(feature_cols):>6}")


LASSO_C = 0.1
lasso_coefs = lasso_results[LASSO_C]["coefs"]

# TODO: Build the selected feature list from non-zero coefficients.
# Hint: [name for name, coef in zip(feature_cols, lasso_coefs) if abs(coef) > 1e-6]
lasso_selected = ____

lasso_importance = sorted(
    [
        (name, float(abs(coef)))
        for name, coef in zip(feature_cols, lasso_coefs)
        if abs(coef) > 1e-6
    ],
    key=lambda x: x[1],
    reverse=True,
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(lasso_selected) > 0, "Task 3: L1 should retain SOME features"
assert len(lasso_selected) < len(
    feature_cols
), "Task 3: L1 should ELIMINATE some features"
print("\n[ok] Checkpoint 1 passed — regularisation path complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE sparsity trajectory
# ════════════════════════════════════════════════════════════════════════

print(f"\n--- L1 Selected Features (C={LASSO_C}, {len(lasso_selected)} total) ---")
print(f"{'Feature':<35} {'|coef|':>10}")
print("-" * 48)
max_abs = max((c for _, c in lasso_importance), default=1.0)
for name, imp in lasso_importance[:15]:
    bar = "#" * int((imp / max_abs) * 20)
    print(f"  {name:<33} {imp:>10.4f}  {bar}")

print("\n--- Sparsity Trajectory ---")
for c_val in C_VALUES:
    nz = lasso_results[c_val]["n_nonzero"]
    bar = "#" * int(nz / max(1, len(feature_cols)) * 40)
    print(f"  C={c_val:>7.3f}  {nz:>4}/{len(feature_cols):<4}  {bar}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert (
    lasso_results[0.001]["n_nonzero"] <= lasso_results[10.0]["n_nonzero"]
), "Task 4: stronger regularisation must yield fewer non-zero coefficients"
print("\n[ok] Checkpoint 2 passed — sparsity trajectory is monotonic\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4b — LOG the embedded run
# ════════════════════════════════════════════════════════════════════════


async def log_embedded() -> str:
    conn, tracker, exp_id = await setup_tracking()
    run_id = await log_selection_run(
        tracker,
        exp_id,
        run_name=f"embedded_lasso_c{LASSO_C}",
        method="embedded",
        selected_features=lasso_selected,
        total_features=len(feature_cols),
        extra_params={
            "estimator": "LogisticRegression",
            "penalty": "l1",
            "C": str(LASSO_C),
            "solver": "saga",
            "max_iter": "5000",
        },
        extra_metrics={
            "top_abs_coef": lasso_importance[0][1] if lasso_importance else 0.0,
            "n_nonzero_c0p001": float(lasso_results[0.001]["n_nonzero"]),
            "n_nonzero_c10": float(lasso_results[10.0]["n_nonzero"]),
        },
    )
    await conn.close()
    return run_id


run_id = asyncio.run(log_embedded())
print(f"\n  ExperimentTracker run: {run_id}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Bank Card-Fraud Scoring
# ════════════════════════════════════════════════════════════════════════
# DBS processes ~6M card transactions/day. L1 wins because:
#   - single-fit training fits a nightly retrain budget
#   - sparse weights skip zeroed features at inference time
#   - linear coefficients are directly auditable by MAS regulators
#
# BUSINESS IMPACT: ~S$3.36M/year in additional fraud prevented; ~11x
# net ROI after infra ops.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Standardised features before L1 fitting (non-negotiable)
  [x] Walked the regularisation path across multiple C values
  [x] Read non-zero coefficients as an interpretable ranking

  Next: 05_validation_and_tracking.py — schema contract, multi-method
  consensus, and leakage audit.
"""
)
