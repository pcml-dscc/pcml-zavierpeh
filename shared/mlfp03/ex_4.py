# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 4 — Gradient Boosting Deep Dive.

Contains: Singapore credit-scoring data loader, preprocessing pipeline
wrapper, numpy/polars conversion, model-zoo factories (XGBoost/LightGBM/
CatBoost with identical defaults), evaluation metric helpers, and the
output directory used by every technique file.

Technique-specific code (from-scratch boosting loops, split-gain formulas,
hyperparameter sweeps, early-stopping analysis) does NOT belong here — it
lives in the per-technique files under `modules/mlfp03/solutions/ex_4/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from kailash_ml import PreprocessingPipeline
from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# CONFIG — output directory, random seeds, dataset tag
# ════════════════════════════════════════════════════════════════════════

SEED = 42
DATASET_MODULE = "mlfp02"
DATASET_FILE = "sg_credit_scoring.parquet"
TARGET_COLUMN = "default"

OUTPUT_DIR = Path("outputs") / "mlfp03" / "ex_4_boosting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore credit-scoring, shared across all four techniques
# ════════════════════════════════════════════════════════════════════════


def load_credit_data() -> pl.DataFrame:
    """Load the Singapore credit-scoring dataset via MLFPDataLoader."""
    loader = MLFPDataLoader()
    return loader.load(DATASET_MODULE, DATASET_FILE)


def prepare_credit_split() -> dict[str, Any]:
    """Load credit data and return a train/test split ready for boosting.

    Returns a dict with:
      X_train, y_train, X_test, y_test : numpy arrays
      feature_names                    : list[str]
      default_rate                     : float (positive-class prevalence)

    Tree models do not need normalisation, so we set ``normalize=False``.
    Categoricals are ordinal-encoded because XGBoost/LightGBM expect numeric
    input; CatBoost would accept raw categoricals but we keep the pipeline
    consistent across all three libraries for a fair comparison.
    """
    credit = load_credit_data()

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        data=credit,
        target=TARGET_COLUMN,
        train_size=0.8,
        seed=SEED,
        normalize=False,
        categorical_encoding="ordinal",
        imputation_strategy="median",
    )

    feature_cols = [c for c in result.train_data.columns if c != TARGET_COLUMN]
    X_train, y_train, col_info = to_sklearn_input(
        result.train_data,
        feature_columns=feature_cols,
        target_column=TARGET_COLUMN,
    )
    X_test, y_test, _ = to_sklearn_input(
        result.test_data,
        feature_columns=feature_cols,
        target_column=TARGET_COLUMN,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": col_info["feature_columns"],
        "default_rate": float(credit[TARGET_COLUMN].mean()),
    }


# ════════════════════════════════════════════════════════════════════════
# MODEL FACTORIES — identical defaults for fair comparison
# ════════════════════════════════════════════════════════════════════════


def make_xgboost(
    n_estimators: int = 500,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    **extra: Any,
):
    """Build an XGBoost classifier with course-standard defaults."""
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0,
        **extra,
    )


def make_lightgbm(
    n_estimators: int = 500,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    **extra: Any,
):
    """Build a LightGBM classifier with course-standard defaults."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=31,
        random_state=SEED,
        verbose=-1,
        **extra,
    )


def make_catboost(
    iterations: int = 500,
    learning_rate: float = 0.1,
    depth: int = 6,
    **extra: Any,
):
    """Build a CatBoost classifier with course-standard defaults."""
    import catboost as cb

    return cb.CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=SEED,
        verbose=0,
        **extra,
    )


# ════════════════════════════════════════════════════════════════════════
# EVALUATION — shared metric helpers for imbalanced binary classification
# ════════════════════════════════════════════════════════════════════════


def evaluate_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return the full boosting-eval metric bundle.

    AUC-PR is the primary metric — with a 12% default rate, AUC-ROC rewards
    models that rank negatives correctly even when they miss every default.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def print_metrics(
    name: str, metrics: dict[str, float], train_time: float | None = None
) -> None:
    """One-line headline: AUC-ROC | AUC-PR | Log Loss | F1 | Time."""
    time_str = f" | time={train_time:.2f}s" if train_time is not None else ""
    print(
        f"  {name}: "
        f"AUC-ROC={metrics['auc_roc']:.4f} | "
        f"AUC-PR={metrics['auc_pr']:.4f} | "
        f"log_loss={metrics['log_loss']:.4f} | "
        f"F1={metrics['f1']:.4f}"
        f"{time_str}"
    )


# ════════════════════════════════════════════════════════════════════════
# FROM-SCRATCH GRADIENT BOOSTING (used by 01_boosting_theory.py)
# ════════════════════════════════════════════════════════════════════════


def xgb_split_gain(
    g_left: float,
    h_left: float,
    g_right: float,
    h_right: float,
    lambda_reg: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """XGBoost split-gain formula from 2nd-order Taylor expansion of log-loss.

    Gain = ½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
    """
    left_score = g_left**2 / (h_left + lambda_reg)
    right_score = g_right**2 / (h_right + lambda_reg)
    parent_score = (g_left + g_right) ** 2 / (h_left + h_right + lambda_reg)
    return 0.5 * (left_score + right_score - parent_score) - gamma


def make_1d_demo(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 1D logistic-shaped binary classification demo.

    Used by the from-scratch boosting loop to show that residual-fitting
    converges in just a handful of rounds.
    """
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 1, n).reshape(-1, 1)
    true_proba = 1 / (1 + np.exp(-10 * (x.ravel() - 0.5)))
    y = rng.binomial(1, true_proba)
    return x, y
