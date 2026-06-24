# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 8 — Production ML + Drift +
Deployment.

Contains: data loading (Singapore credit scoring via MLFPDataLoader),
baseline LightGBM + isotonic calibration training, PSI/KS drift helpers,
and common output directory setup.

Technique-specific code (conformal quantile logic, dashboard rendering,
readiness checklist) stays in the per-technique files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl  # noqa: F401  # re-exported for students
import lightgbm as lgb
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from kailash_ml import PreprocessingPipeline
from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════

setup_environment()

OUTPUT_DIR = Path("outputs") / "ex8_production"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore credit scoring (from MLFP02)
# ════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42


def load_credit_split() -> dict[str, Any]:
    """Load Singapore credit scoring data, preprocess, and return a split.

    Returns a dict with keys:
        X_train, y_train, X_test, y_test : numpy arrays
        feature_names                    : list[str]
        default_rate                     : float (train positive rate)
    """
    loader = MLFPDataLoader()
    credit = loader.load("mlfp02", "sg_credit_scoring.parquet")

    # Drop identifier columns before preprocessing — `customer_id` is a row
    # key, not a feature. Leaving it in the matrix causes drift noise to
    # dominate the top-variance feature list AND inflates AUC against the
    # model's pattern-recognition capability on IDs. Both are data-leakage
    # symptoms; the fix is to exclude the identifier at load time.
    id_columns = [c for c in ("customer_id", "application_id") if c in credit.columns]
    if id_columns:
        credit = credit.drop(id_columns)

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        credit,
        target="default",
        seed=RANDOM_SEED,
        normalize=False,
        categorical_encoding="ordinal",
    )

    feature_cols = [c for c in result.train_data.columns if c != "default"]
    X_train, y_train, col_info = to_sklearn_input(
        result.train_data,
        feature_columns=feature_cols,
        target_column="default",
    )
    X_test, y_test, _ = to_sklearn_input(
        result.test_data,
        feature_columns=feature_cols,
        target_column="default",
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": col_info["feature_columns"],
        "default_rate": float(y_train.mean()),
    }


# ════════════════════════════════════════════════════════════════════════
# BASELINE MODEL — LightGBM + isotonic calibration
# ════════════════════════════════════════════════════════════════════════
#
# Hyperparameters come from Exercise 7's Bayesian optimisation run. These
# are frozen here so every technique file trains an identical baseline
# model and can be compared apples-to-apples.


def build_baseline_model(y_train: np.ndarray) -> lgb.LGBMClassifier:
    """Return an unfit LightGBM classifier configured for credit default."""
    base_rate = float(y_train.mean())
    scale_pos_weight = (1.0 - base_rate) / max(base_rate, 1e-6)
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        verbose=-1,
    )


def train_calibrated_model(
    X_train: np.ndarray, y_train: np.ndarray
) -> CalibratedClassifierCV:
    """Train LightGBM and wrap with isotonic calibration (cv=5)."""
    base = build_baseline_model(y_train)
    base.fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def evaluate_classification(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return the full classification metric bundle used across ex_8."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


# ════════════════════════════════════════════════════════════════════════
# DRIFT STATISTICS — PSI + KS
# ════════════════════════════════════════════════════════════════════════
#
# PSI (Population Stability Index):
#     PSI = Σ (p_new - p_ref) * ln(p_new / p_ref)
#     < 0.1 no shift, 0.1-0.2 moderate, > 0.2 significant
# KS (Kolmogorov-Smirnov): two-sample test on the empirical CDFs.


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index on 1-D arrays."""
    _, bin_edges = np.histogram(reference, bins=bins)
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    # Laplace-smooth to avoid log(0)
    ref_props = (ref_counts + 1) / (len(reference) + bins)
    cur_props = (cur_counts + 1) / (len(current) + bins)
    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Return (KS statistic, p-value) on 1-D arrays."""
    ks_stat, p_value = stats.ks_2samp(reference, current)
    return float(ks_stat), float(p_value)


def drift_row(
    reference: np.ndarray, current: np.ndarray, psi_threshold: float = 0.1
) -> dict[str, float | str]:
    """Return {psi, ks_stat, ks_pval, drift} for a single feature."""
    psi = compute_psi(reference, current)
    ks_stat, ks_pval = compute_ks(reference, current)
    drift = "YES" if (psi > psi_threshold or ks_pval < 0.05) else "No"
    return {"psi": psi, "ks_stat": ks_stat, "ks_pval": ks_pval, "drift": drift}


def simulate_gradual_drift(
    X_ref: np.ndarray, X_new: np.ndarray, n_features: int = 3, shift: float = 0.5
) -> np.ndarray:
    """Return a copy of X_new with mean-shifted top features (drift sim)."""
    drifted = X_new.copy()
    for i in range(n_features):
        drifted[:, i] += shift * X_ref[:, i].std()
    return drifted


def simulate_sudden_drift(
    X_ref: np.ndarray,
    X_new: np.ndarray,
    feature_idx: int = 0,
    sigma_shift: float = 3.0,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Replace one column in X_new with a heavily shifted Gaussian."""
    rng = np.random.default_rng(seed)
    drifted = X_new.copy()
    drifted[:, feature_idx] = rng.normal(
        loc=X_ref[:, feature_idx].mean() + sigma_shift * X_ref[:, feature_idx].std(),
        scale=X_ref[:, feature_idx].std(),
        size=drifted.shape[0],
    )
    return drifted
