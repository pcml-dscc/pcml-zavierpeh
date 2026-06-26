# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 3: Regression Modelling & Interpretation

Complete the `solve()` function. Read problem.md for the full specification.
Every regression is solved in closed form (OLS via least squares; logistic via
Newton-Raphson to the unique MLE) so your numbers must match the independently
re-derived reference. Standardise predictors (z-score) before fitting.

    python grader.py starter.py
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats
from scipy.special import expit

from shared import MLFPDataLoader

# --- Fixed problem constants (do not change) ---
OLS_FEATURES = [
    "income_imp",
    "age",
    "employment_years",
    "debt_to_income",
    "credit_age_years",
    "num_dependents",
    "edu_ord",
]
LOGIT_FEATURES = [
    "credit_utilization",
    "num_late_payments",
    "previous_defaults",
    "debt_to_income",
    "num_hard_inquiries",
]
EDU_MAP = {
    "primary": 1.0,
    "secondary": 2.0,
    "diploma": 3.0,
    "degree": 4.0,
    "postgraduate": 5.0,
}
TARGET = "loan_amount_sgd"


def solve() -> dict:
    """Return the regression / interpretation answer dict.

    See problem.md for the exact 13 keys and how each is defined.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "sg_credit_scoring.parquet")

    # TODO 1: Preprocess — median-impute income_sgd -> income_imp;
    #         ordinal-encode education via EDU_MAP -> edu_ord (Float64).
    #         No rows are dropped: n_obs == df.height.
    income_median = float(df["income_sgd"].median())
    df = df.with_columns(
        pl.when(pl.col("income_sgd").is_null())
        .then(income_median)
        .otherwise(pl.col("income_sgd"))
        .alias("income_imp"),
        pl.col("education")
        .map_elements(lambda v: EDU_MAP.get(v, np.nan), return_dtype=pl.Float64)
        .alias("edu_ord"),
    )

    n_obs = df.height

    def zscore(column: str | pl.Expr) -> pl.Expr:
        expr = pl.col(column) if isinstance(column, str) else column
        values = df.select(expr).to_numpy().flatten().astype(np.float64)
        mean = values.mean()
        std = values.std(ddof=0)
        if std == 0:
            return pl.lit(0.0)
        return ((expr - mean) / std).cast(pl.Float64)

    def build_design_matrix(features: list[str], add_terms: bool = False) -> tuple[np.ndarray, list[str]]:
        cols = []
        names = ["intercept"]
        intercept = np.ones((n_obs, 1), dtype=np.float64)

        for feature in features:
            expr = zscore(feature).alias(feature + "_std")
            df_feature = df.select(expr).to_numpy().flatten().astype(np.float64)
            cols.append(df_feature)
            names.append(feature)

        X = np.hstack([intercept, np.column_stack(cols)])

        if add_terms:
            income_std = X[:, names.index("income_imp")]
            age_std = X[:, names.index("age")]
            emp_std = X[:, names.index("employment_years")]
            extra = np.column_stack([income_std**2, age_std * emp_std])
            X = np.hstack([X, extra])
            names.extend(["income_std_sq", "age_emp_std"])

        return X, names

    def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float, float, int]:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        rss = float(np.sum((y - y_pred) ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - rss / tss
        p = X.shape[1]
        return beta, rss, tss, r_squared, p

    def inference_stats(X: np.ndarray, y: np.ndarray, beta: np.ndarray, rss: float, r_squared: float, p: int) -> tuple[dict, dict, dict, float, float, float, float]:
        n = X.shape[0]
        sigma2 = rss / (n - p)
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * xtx_inv))
        t_stats = beta / se
        p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=n - p)
        f_statistic = (r_squared / (p - 1)) / ((1.0 - r_squared) / (n - p))
        f_p_value = float(stats.f.sf(f_statistic, p - 1, n - p))

        names = ["intercept"] + [col for col in OLS_FEATURES]
        coef_dict = {name: float(beta[i]) for i, name in enumerate(names)}
        t_dict = {name: float(t_stats[i]) for i, name in enumerate(names)}
        p_dict = {name: float(p_values[i]) for i, name in enumerate(names)}
        return coef_dict, t_dict, p_dict, float(r_squared), float(1.0 - (1.0 - r_squared) * (n - 1) / (n - p)), float(f_statistic), f_p_value

    y = np.asarray(df[TARGET].to_numpy(), dtype=np.float64)
    X, names = build_design_matrix(OLS_FEATURES)
    beta, rss, tss, r_squared, p = fit_ols(X, y)
    coefficients, t_stats, p_values, r_squared, adj_r_squared, f_statistic, f_p_value = inference_stats(X, y, beta, rss, r_squared, p)

    X_full, names_full = build_design_matrix(OLS_FEATURES, add_terms=True)
    beta_full, rss_full, tss_full, r_squared_full, p_full = fit_ols(X_full, y)
    q = 2
    partial_f = float(((rss - rss_full) / q) / (rss_full / (n_obs - p_full)))
    partial_f_p_value = float(stats.f.sf(partial_f, q, n_obs - p_full))
    delta_r_squared = float(r_squared_full - r_squared)

    def fit_logistic(features: list[str]) -> tuple[dict, str]:
        feature_arrays = [np.asarray(df.select(zscore(feature)).to_numpy().flatten(), dtype=np.float64) for feature in features]
        Z = np.column_stack([np.ones(n_obs, dtype=np.float64), *feature_arrays])
        beta_logit = np.zeros(Z.shape[1], dtype=np.float64)
        y_logit = np.asarray(df["default"].to_numpy(), dtype=np.float64)

        for _ in range(100):
            eta = Z @ beta_logit
            mu = expit(eta)
            W = mu * (1 - mu)
            W = np.where(W == 0, 1e-8, W)
            z = eta + (y_logit - mu) / W
            WX = Z * W[:, np.newaxis]
            xtwx = Z.T @ WX
            xTwz = Z.T @ (W * z)
            beta_logit_new = np.linalg.solve(xtwx, xTwz)
            if np.max(np.abs(beta_logit_new - beta_logit)) < 1e-8:
                beta_logit = beta_logit_new
                break
            beta_logit = beta_logit_new

        odds_ratios = {"intercept": float(np.exp(beta_logit[0]))}
        for i, feature in enumerate(features, start=1):
            odds_ratios[feature] = float(np.exp(beta_logit[i]))
        strongest = max(features, key=lambda f: abs(beta_logit[features.index(f) + 1]))
        return odds_ratios, strongest

    odds_ratios, strongest_logit_predictor = fit_logistic(LOGIT_FEATURES)

    return {
        "n_obs": int(n_obs),
        "coefficients": coefficients,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
        "partial_f": partial_f,
        "partial_f_p_value": partial_f_p_value,
        "delta_r_squared": delta_r_squared,
        "odds_ratios": odds_ratios,
        "strongest_logit_predictor": strongest_logit_predictor,
    }


if __name__ == "__main__":
    print(solve())
