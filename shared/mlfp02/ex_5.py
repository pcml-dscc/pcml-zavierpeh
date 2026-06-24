# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP02 Exercise 5 — Linear Regression.

Contains: HDB resale data loading, feature engineering, OLS fitting utilities,
diagnostic helpers, and visualisation helpers. Technique-specific code (the
derivation walk-throughs, the WLS weight construction, the polynomial/dummy
matrices) lives in the per-technique files, not here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from shared.data_loader import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════

NUMERIC_FEATURES: list[str] = [
    "floor_area_sqm",
    "storey_midpoint",
    "remaining_lease_years",
]
TARGET: str = "resale_price"
BASE_FLAT_TYPE: str = "3 ROOM"

OUTPUT_DIR = Path("outputs") / "ex5_linear_regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — HDB resale flat transactions
# ════════════════════════════════════════════════════════════════════════


def load_hdb_clean() -> pl.DataFrame:
    """Load HDB resale data, engineer numeric features, drop nulls.

    Returns a polars DataFrame with columns:
      - floor_area_sqm (Float)
      - storey_midpoint (Float, midpoint of '07 TO 09' range)
      - remaining_lease_years (Float, 99 - years_elapsed_since_commence)
      - resale_price (target, SGD)
      - flat_type (categorical)
      - town (categorical)
    """
    loader = MLFPDataLoader()
    hdb = loader.load("mlfp01", "hdb_resale.parquet")

    hdb = hdb.with_columns(
        pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
    )
    hdb_recent = hdb.filter(pl.col("transaction_date") >= pl.date(2020, 1, 1))

    hdb_recent = hdb_recent.with_columns(
        (
            (
                pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
                + pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
            )
            / 2
        ).alias("storey_midpoint"),
        (99 - (pl.col("transaction_date").dt.year() - pl.col("lease_commence_date")))
        .cast(pl.Float64)
        .alias("remaining_lease_years"),
    )

    return hdb_recent.drop_nulls(
        subset=[*NUMERIC_FEATURES, TARGET],
    )


def build_design_matrix(
    df: pl.DataFrame, features: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X_with_intercept, y, feature_names) from a cleaned HDB frame."""
    features = features or NUMERIC_FEATURES
    y = df[TARGET].to_numpy().astype(np.float64)
    X_raw = df.select(features).to_numpy().astype(np.float64)
    n_obs = X_raw.shape[0]
    X = np.column_stack([np.ones(n_obs), X_raw])
    feature_names = ["intercept"] + list(features)
    return X, y, feature_names


# ════════════════════════════════════════════════════════════════════════
# OLS CORE — the workhorse every technique reuses
# ════════════════════════════════════════════════════════════════════════


def fit_ols(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit OLS via the normal equation and return core statistics.

    Returns a dict with keys: beta, y_hat, residuals, XtX_inv, SSR, SST, SSE,
    R2, adj_R2, sigma_hat, se_beta, t_stats, p_values, f_stat, f_p_value, n, k.
    """
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y

    y_hat = X @ beta
    residuals = y - y_hat

    SSR = float(np.sum(residuals**2))
    SST = float(np.sum((y - y.mean()) ** 2))
    SSE = float(np.sum((y_hat - y.mean()) ** 2))

    sigma_sq = SSR / (n - k)
    sigma_hat = float(np.sqrt(sigma_sq))
    se_beta = np.sqrt(sigma_sq * np.diag(XtX_inv))
    t_stats = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    r2 = 1 - SSR / SST
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
    f_stat = (SSE / (k - 1)) / (SSR / (n - k))
    f_p = 1 - stats.f.cdf(f_stat, dfn=k - 1, dfd=n - k)

    return {
        "beta": beta,
        "y_hat": y_hat,
        "residuals": residuals,
        "XtX_inv": XtX_inv,
        "SSR": SSR,
        "SST": SST,
        "SSE": SSE,
        "R2": float(r2),
        "adj_R2": float(adj_r2),
        "sigma_hat": sigma_hat,
        "se_beta": se_beta,
        "t_stats": t_stats,
        "p_values": p_values,
        "f_stat": float(f_stat),
        "f_p_value": float(f_p),
        "n": int(n),
        "k": int(k),
    }


def print_coef_table(names: list[str], fit: dict[str, Any]) -> None:
    """Print coefficient / SE / t / p table for an OLS fit."""
    beta = fit["beta"]
    se = fit["se_beta"]
    t = fit["t_stats"]
    p = fit["p_values"]
    print(f"\n{'Feature':<25} {'β':>14} {'SE(β)':>12} {'t':>8} {'p':>10} {'Sig':>4}")
    print("-" * 78)
    for i, name in enumerate(names):
        if p[i] < 0.001:
            sig = "***"
        elif p[i] < 0.01:
            sig = "**"
        elif p[i] < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(
            f"{name:<25} {beta[i]:>14,.2f} {se[i]:>12,.2f} "
            f"{t[i]:>8.2f} {p[i]:>10.2e} {sig:>4}"
        )


# ════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS — VIF, Breusch-Pagan, residual shape
# ════════════════════════════════════════════════════════════════════════


def compute_vif(X_raw: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Variance Inflation Factor for each feature (no intercept column)."""
    n = X_raw.shape[0]
    results: dict[str, float] = {}
    for j in range(X_raw.shape[1]):
        other = [i for i in range(X_raw.shape[1]) if i != j]
        Xo = np.column_stack([np.ones(n), X_raw[:, other]])
        yj = X_raw[:, j]
        beta_j = np.linalg.lstsq(Xo, yj, rcond=None)[0]
        yhat_j = Xo @ beta_j
        ss_res = np.sum((yj - yhat_j) ** 2)
        ss_tot = np.sum((yj - yj.mean()) ** 2)
        r2_j = 1 - ss_res / ss_tot
        results[feature_names[j]] = (
            float(1.0 / (1.0 - r2_j)) if r2_j < 1 else float("inf")
        )
    return results


def breusch_pagan(residuals: np.ndarray, X_raw: np.ndarray) -> tuple[float, float]:
    """Breusch-Pagan test for heteroscedasticity. Returns (BP statistic, p-value)."""
    n = X_raw.shape[0]
    e_sq = residuals**2
    Xbp = np.column_stack([np.ones(n), X_raw])
    beta = np.linalg.lstsq(Xbp, e_sq, rcond=None)[0]
    pred = Xbp @ beta
    sse = np.sum((e_sq - pred) ** 2)
    sst = np.sum((e_sq - e_sq.mean()) ** 2)
    r2 = 1 - sse / sst
    bp_stat = n * r2
    p = 1 - stats.chi2.cdf(bp_stat, df=X_raw.shape[1])
    return float(bp_stat), float(p)


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — residual plots + actual-vs-predicted
# ════════════════════════════════════════════════════════════════════════


def save_residual_diagnostics(
    y_hat: np.ndarray,
    residuals: np.ndarray,
    feature_col: np.ndarray,
    feature_label: str,
    filename: str,
) -> Path:
    """Four-panel diagnostic figure: residuals vs fitted, histogram, Q-Q, vs feature."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Residuals vs Fitted",
            "Residual Histogram",
            "Q-Q Plot",
            f"Residuals vs {feature_label}",
        ],
    )
    sample = min(3000, len(residuals))
    fig.add_trace(
        go.Scatter(
            x=y_hat[:sample],
            y=residuals[:sample],
            mode="markers",
            marker={"size": 2, "opacity": 0.3},
            name="Residuals",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, row=1, col=1, line_dash="dash")
    fig.add_trace(go.Histogram(x=residuals, nbinsx=50, name="Residuals"), row=1, col=2)
    sorted_resid = np.sort(residuals)
    step = max(1, len(sorted_resid) // 2000)
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, len(sorted_resid)))
    fig.add_trace(
        go.Scatter(
            x=theoretical[::step],
            y=sorted_resid[::step],
            mode="markers",
            marker={"size": 2},
            name="Q-Q",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=feature_col[:sample],
            y=residuals[:sample],
            mode="markers",
            marker={"size": 2, "opacity": 0.3},
            name=f"vs {feature_label}",
        ),
        row=2,
        col=2,
    )
    fig.update_layout(height=600, title="Residual Diagnostics", showlegend=False)
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    return path


def save_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str,
) -> Path:
    """Actual-vs-predicted scatter with the perfect-prediction diagonal."""
    sample = min(2000, len(y_true))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true[:sample],
            y=y_pred[:sample],
            mode="markers",
            marker={"size": 3, "opacity": 0.4},
            name="Predictions",
        )
    )
    lo, hi = float(y_true.min()), float(y_true.max())
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            name="Perfect",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Actual Price (SGD)",
        yaxis_title="Predicted Price (SGD)",
        height=500,
    )
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    return path
