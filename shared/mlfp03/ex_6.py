# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 6 — Interpretability and Fairness.

Contains: Singapore credit scoring data load, LightGBM model training,
TreeSHAP explainer setup, output directory, and common helper utilities.

Technique-specific code (permutation importance loops, LIME wrappers,
fairness audit reports) lives in the per-technique files under
`modules/mlfp03/solutions/ex_6/`.

Import pattern (solutions and local both):

    from shared.mlfp03.ex_6 import (
        FEATURE_NAMES,
        OUTPUT_DIR,
        load_credit_scoring,
        train_credit_model,
        build_shap_explainer,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import lightgbm as lgb
import shap
from sklearn.metrics import roc_auc_score

from kailash_ml import PreprocessingPipeline
from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader


# ════════════════════════════════════════════════════════════════════════
# PATHS / CONSTANTS
# ════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs") / "mlfp03_ex6_interpretability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Singapore credit scoring: Monetary Authority of Singapore (MAS) requires
# explainability for credit decisions under the Model Risk Management
# guideline. This dataset simulates a retail-bank default prediction task
# used throughout MLFP02/MLFP03.
DATASET_MODULE = "mlfp02"
DATASET_FILE = "sg_credit_scoring.parquet"
TARGET_COLUMN = "default"
RANDOM_SEED = 42

# Protected attribute candidates we audit for disparate impact.
PROTECTED_CANDIDATES: list[str] = ["age", "gender", "ethnicity", "marital_status"]


# ════════════════════════════════════════════════════════════════════════
# DATA LOAD + MODEL TRAIN
# ════════════════════════════════════════════════════════════════════════

# Populated on first call so every technique file sees the same split.
_CACHE: dict[str, Any] = {}


def load_credit_scoring() -> dict[str, Any]:
    """Load the Singapore credit scoring dataset and run the M3 preprocessing
    pipeline. Returns a dict with X_train, y_train, X_test, y_test, feature_names.

    The return value is cached so repeated calls from different technique
    files re-use the same split (essential for interpretability comparisons).
    """
    if _CACHE:
        return _CACHE

    loader = MLFPDataLoader()
    credit: pl.DataFrame = loader.load(DATASET_MODULE, DATASET_FILE)

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        credit,
        target=TARGET_COLUMN,
        seed=RANDOM_SEED,
        normalize=False,
        categorical_encoding="ordinal",
    )

    feature_columns = [c for c in result.train_data.columns if c != TARGET_COLUMN]
    X_train, y_train, col_info = to_sklearn_input(
        result.train_data,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
    )
    X_test, y_test, _ = to_sklearn_input(
        result.test_data,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
    )
    feature_names: list[str] = col_info["feature_columns"]

    _CACHE.update(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )
    return _CACHE


def train_credit_model() -> dict[str, Any]:
    """Train the LightGBM credit default model. Cached per-process.

    Returns a dict with model, y_proba, y_pred, auc, and all data from
    `load_credit_scoring()`.
    """
    if "model" in _CACHE:
        return _CACHE

    data = load_credit_scoring()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)

    _CACHE.update(model=model, y_proba=y_proba, y_pred=y_pred, auc=auc)
    return _CACHE


# ════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINER
# ════════════════════════════════════════════════════════════════════════


def build_shap_explainer() -> dict[str, Any]:
    """Construct the TreeSHAP explainer and compute SHAP values for X_test.

    Returns the full bundle: model, data, explainer, shap_vals, expected_value.
    """
    if "shap_vals" in _CACHE:
        return _CACHE

    bundle = train_credit_model()
    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer.shap_values(bundle["X_test"])

    # TreeSHAP for binary classifiers may return [class_0, class_1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    expected_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, list)
        else explainer.expected_value
    )

    _CACHE.update(
        explainer=explainer,
        shap_vals=shap_vals,
        expected_value=expected_value,
    )
    return _CACHE


# ════════════════════════════════════════════════════════════════════════
# REUSABLE UTILITIES
# ════════════════════════════════════════════════════════════════════════


def rank_features_by_mean_abs_shap(
    shap_vals: np.ndarray, feature_names: list[str]
) -> list[tuple[str, float]]:
    """Return [(feature, mean_abs_shap), ...] sorted descending."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)


def feature_index(feature_names: list[str], name: str) -> int:
    """Lookup a feature column index by name, raising a clear error."""
    if name not in feature_names:
        raise KeyError(
            f"Feature '{name}' not found. Available: {feature_names[:10]}..."
        )
    return feature_names.index(name)


def synthetic_group_split(
    X: np.ndarray, feature_idx: int = 0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split X into two groups on a median cut of `feature_idx`.

    Returns (group_a_mask, group_b_mask, median_value).
    Used as a fallback when no protected attribute is present in features.
    """
    vals = X[:, feature_idx]
    median_val = float(np.median(vals))
    group_a = vals <= median_val
    group_b = ~group_a
    return group_a, group_b, median_val


def print_section(title: str, char: str = "=") -> None:
    """Print a standardised section banner."""
    line = char * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(line)
