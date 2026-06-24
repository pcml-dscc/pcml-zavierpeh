# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 2 — Regularisation and
Cross-Validation.

Contains: data loading for the Singapore credit scoring dataset,
feature preparation, synthetic 1D bias-variance problem, alpha grids,
plotting helpers, and a shared OUTPUT_DIR for generated artefacts.

Technique-specific code (Ridge/Lasso model construction, nested CV
loops, learning curves) lives in the per-technique solution files —
this module holds only the helpers those files share.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from kailash_ml import PreprocessingPipeline
from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════

# Deterministic seed for every random operation in this exercise
SEED = 42

# Alpha sweep used by Ridge / Lasso / ElasticNet and the regularisation path
ALPHAS: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Output directory for HTML plots, summary CSVs, etc.
OUTPUT_DIR = Path("outputs") / "mlfp03_ex2_regularisation_cv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore credit scoring
# ════════════════════════════════════════════════════════════════════════


def load_credit_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]
):
    """Load + preprocess the MLFP02 Singapore credit scoring parquet.

    Returns:
        X_train, y_train, X_test, y_test, feature_names

    Target: predicts ``credit_utilization`` (continuous ratio). Falls
    back to ``income_sgd`` if the utilisation column is absent in older
    dataset revisions.

    Uses ``kailash_ml.PreprocessingPipeline`` for normalisation +
    ordinal encoding + median imputation. All regularised models
    REQUIRE normalised features (otherwise the penalty is unevenly
    distributed across the coefficient vector).
    """
    loader = MLFPDataLoader()
    credit = loader.load("mlfp02", "sg_credit_scoring.parquet")

    target_col = (
        "credit_utilization" if "credit_utilization" in credit.columns else "income_sgd"
    )

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        data=credit,
        target=target_col,
        train_size=0.8,
        seed=SEED,
        normalize=True,
        categorical_encoding="ordinal",
        imputation_strategy="median",
    )

    feature_cols = [c for c in result.train_data.columns if c != target_col]
    X_train, y_train, col_info = to_sklearn_input(
        result.train_data,
        feature_columns=feature_cols,
        target_column=target_col,
    )
    X_test, y_test, _ = to_sklearn_input(
        result.test_data,
        feature_columns=feature_cols,
        target_column=target_col,
    )

    return X_train, y_train, X_test, y_test, col_info["feature_columns"]


# ════════════════════════════════════════════════════════════════════════
# SYNTHETIC 1D PROBLEM — for bias/variance and polynomial fits
# ════════════════════════════════════════════════════════════════════════


def make_sine_dataset(
    n: int = 100,
    noise_sigma: float = 0.2,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a 1D noisy-sine regression problem.

    Returns:
        x_train_2d (n_train, 1), y_train (n_train,),
        x_test_2d (n_test, 1),   y_test  (n_test,),
        noise_variance (float, σ² — the irreducible error floor)

    The true function is ``y = sin(2πx) + ε`` with ε ~ N(0, σ²).
    The noise variance is returned so callers can use it in the
    bias-variance decomposition (σ² is the "irreducible noise" term).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, noise_sigma, n)

    split = int(n * 0.8)
    x_train = x[:split].reshape(-1, 1)
    x_test = x[split:].reshape(-1, 1)
    y_train = y[:split]
    y_test = y[split:]
    return x_train, y_train, x_test, y_test, noise_sigma**2


def make_poly_pipeline(degree: int) -> Pipeline:
    """Polynomial-features + scaler + linear-regression pipeline."""
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]
    )


# ════════════════════════════════════════════════════════════════════════
# REPORTING HELPERS
# ════════════════════════════════════════════════════════════════════════


def print_header(title: str) -> None:
    """Print a banner so each phase of a technique file is easy to spot."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def format_results_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    """Render a small table of dicts as fixed-width text."""
    header = "  ".join(f"{c:>12}" for c in cols)
    sep = "-" * len(header)
    body = "\n".join(
        "  ".join(
            f"{row[c]:>12.4f}" if isinstance(row[c], float) else f"{row[c]:>12}"
            for c in cols
        )
        for row in rows
    )
    return f"{header}\n{sep}\n{body}"


def save_html_plot(fig: Any, name: str) -> Path:
    """Write a plotly figure into OUTPUT_DIR and return the path."""
    path = OUTPUT_DIR / name
    fig.write_html(str(path))
    return path
