# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 3 — The Classical ML Zoo.

Contains: e-commerce churn data loading, preprocessing, CV strategy,
2D PCA projection for decision boundary plots, model comparison helpers,
and a shared ModelVisualizer-backed plot utility.

Technique-specific code (model fitting, parameter sweeps, from-scratch
Gini, OOB convergence, decision guide) does NOT belong here — it lives
in the per-technique files under modules/mlfp03/solutions/ex_3/.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from kailash_ml import ModelVisualizer, PreprocessingPipeline
from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════

setup_environment()
np.random.seed(42)

# Output directory for comparison artifacts (HTML plots, tables)
OUTPUT_DIR = Path("outputs") / "ex3_model_zoo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# E-commerce churn dataset — Singapore APAC retail churn scenario
DATASET_MODULE = "mlfp03"
DATASET_FILE = "ecommerce_customers.parquet"
TARGET_COL = "churned"

# SVM with RBF kernel is O(n²) — cap the training set so every technique
# in the zoo fits in a few seconds on a laptop.
SUBSAMPLE_N = 5000
RANDOM_SEED = 42

# Drop columns that are text/ID or leak the target
DROP_COLS = ["customer_id", "review_text", "product_categories"]


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING + PREPROCESSING
# ════════════════════════════════════════════════════════════════════════


def load_ecommerce_churn() -> pl.DataFrame:
    """Load the Singapore e-commerce churn dataset (polars DataFrame).

    Drops text/ID columns and subsamples for SVM tractability.
    """
    loader = MLFPDataLoader()
    df = loader.load(DATASET_MODULE, DATASET_FILE)
    df = df.sample(n=min(SUBSAMPLE_N, df.height), seed=RANDOM_SEED)
    keep = [c for c in df.columns if c not in DROP_COLS]
    return df.select(keep)


def build_train_test_split() -> dict[str, Any]:
    """Return a fully prepared dict: X_train, X_test, y_train, y_test, feature_names, cv.

    Uses kailash_ml PreprocessingPipeline with z-score normalisation and
    ordinal categorical encoding. Every technique file calls this so all
    models share identical folds and identical preprocessing.
    """
    df = load_ecommerce_churn()

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        data=df,
        target=TARGET_COL,
        train_size=0.8,
        seed=RANDOM_SEED,
        normalize=True,
        normalize_method="zscore",
        categorical_encoding="ordinal",
        imputation_strategy="median",
    )

    feature_cols = [c for c in result.train_data.columns if c != TARGET_COL]
    X_train, y_train, col_info = to_sklearn_input(
        result.train_data,
        feature_columns=feature_cols,
        target_column=TARGET_COL,
    )
    X_test, y_test, _ = to_sklearn_input(
        result.test_data,
        feature_columns=feature_cols,
        target_column=TARGET_COL,
    )
    feature_names = col_info["feature_columns"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "cv": cv,
        "churn_rate": float(np.mean(y_train)),
    }


# ════════════════════════════════════════════════════════════════════════
# 2D PCA PROJECTION — shared so every technique plots on the same axes
# ════════════════════════════════════════════════════════════════════════


def project_2d(X_train: np.ndarray, X_test: np.ndarray) -> dict[str, Any]:
    """Fit PCA(2) on X_train and project both train and test.

    Returns {X_train_2d, X_test_2d, explained_variance, pca}.
    """
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)
    return {
        "X_train_2d": X_train_2d,
        "X_test_2d": X_test_2d,
        "explained_variance": pca.explained_variance_ratio_,
        "pca": pca,
    }


# ════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION HELPER — keep every parameter sweep one line
# ════════════════════════════════════════════════════════════════════════


def cv_accuracy_f1(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Any,
) -> tuple[float, float]:
    """Return (mean_accuracy, mean_f1) for a 5-fold CV."""
    acc = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy").mean()
    f1 = cross_val_score(estimator, X, y, cv=cv, scoring="f1").mean()
    return float(acc), float(f1)


# ════════════════════════════════════════════════════════════════════════
# EVALUATION — train on full set, measure timing, return pred/prob/metrics
# ════════════════════════════════════════════════════════════════════════


def fit_and_evaluate(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> dict[str, Any]:
    """Fit, predict, score, and time a single model.

    Returns a dict with keys: name, model, pred, prob, train_time,
    accuracy, f1, auc_roc.
    """
    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    pred = estimator.predict(X_test)
    if hasattr(estimator, "predict_proba"):
        prob = estimator.predict_proba(X_test)[:, 1]
    else:
        # Decision-function fallback (never used by the zoo but keeps contract)
        prob = estimator.decision_function(X_test)

    return {
        "name": name,
        "model": estimator,
        "pred": pred,
        "prob": prob,
        "train_time": float(train_time),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "auc_roc": float(roc_auc_score(y_test, prob)),
    }


def print_classification_report(y_test: np.ndarray, pred: np.ndarray) -> None:
    """Print sklearn classification report with churn-friendly target names."""
    print(
        classification_report(
            y_test,
            pred,
            target_names=["Retained", "Churned"],
        )
    )


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — Plotly via kailash_ml.ModelVisualizer
# ════════════════════════════════════════════════════════════════════════


def get_visualizer() -> ModelVisualizer:
    """Return a ModelVisualizer instance (polars-native plots)."""
    return ModelVisualizer()


def save_metric_comparison(
    metric_dict: dict[str, dict[str, float]], fname: str
) -> Path:
    """Save a metric_comparison plot to OUTPUT_DIR/fname and return the path."""
    viz = get_visualizer()
    fig = viz.metric_comparison(metric_dict)
    fig.update_layout(title="Classical ML Zoo — Performance Comparison")
    out = OUTPUT_DIR / fname
    fig.write_html(str(out))
    return out


# ════════════════════════════════════════════════════════════════════════
# DECISION BOUNDARY MESH — shared helper so every technique file uses
# the same axes, grid, and figure style.
# ════════════════════════════════════════════════════════════════════════


def decision_boundary_mesh(
    X_2d: np.ndarray,
    step: float = 0.1,
    pad: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (xx, yy) meshgrid covering the 2D PCA projection."""
    x_min, x_max = X_2d[:, 0].min() - pad, X_2d[:, 0].max() + pad
    y_min, y_max = X_2d[:, 1].min() - pad, X_2d[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step),
    )
    return xx, yy


# ════════════════════════════════════════════════════════════════════════
# SINGAPORE E-COMMERCE CHURN — business-impact constants
# ════════════════════════════════════════════════════════════════════════
# Public industry figures used for the "Apply" phases. Sources in reading
# notes (SGX retail analyst reports, Shopee/Lazada 2024 ops reviews).

AVG_CUSTOMER_LIFETIME_VALUE_SGD = 420.0  # avg 12-month CLV per retained SG customer
RETENTION_OFFER_COST_SGD = 18.0  # targeted promo cost per flagged customer
MONTHLY_ACTIVE_CUSTOMERS = 250_000  # typical mid-market SG e-commerce platform
ANNUAL_CHURN_BASELINE = 0.22  # industry baseline annual churn


def churn_saved_dollars(true_positives: int) -> float:
    """Dollar value of correctly identified churners (retention offer accepted).

    Assumes a 40% offer-acceptance rate and the retained lifetime value
    net of offer cost. Public industry benchmarks — not proprietary data.
    """
    accept_rate = 0.40
    net_value_per_save = AVG_CUSTOMER_LIFETIME_VALUE_SGD - RETENTION_OFFER_COST_SGD
    return round(true_positives * accept_rate * net_value_per_save, 2)
