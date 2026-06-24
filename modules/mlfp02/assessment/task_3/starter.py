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
    # TODO 2: OLS — build the predictor matrix from OLS_FEATURES, z-score each
    #         column (population sd, ddof=0), prepend an intercept column of 1s,
    #         and solve beta via np.linalg.lstsq against the RAW target.
    # TODO 3: Inference — rss, tss, r_squared, adj_r_squared,
    #         sigma2 = rss/(n-p), se = sqrt(diag(sigma2 * inv(X'X))),
    #         t = beta/se, p = 2*stats.t.sf(|t|, df=n-p),
    #         f_statistic = (r2/(p-1)) / ((1-r2)/(n-p)),
    #         f_p_value = stats.f.sf(f_statistic, p-1, n-p).
    #         Return coefficients / t_stats / p_values as dicts keyed by feature
    #         name plus "intercept".
    # TODO 4: Partial F-test — add two terms built from the STANDARDISED base
    #         columns: income_std**2 and age_std*employment_std. Refit, then
    #         partial_f = ((rss - rss_full)/q) / (rss_full/(n - p_full)) with q=2;
    #         partial_f_p_value = stats.f.sf(partial_f, q, n - p_full);
    #         delta_r_squared = r2_full - r2.
    # TODO 5: Logistic — z-score LOGIT_FEATURES, prepend intercept, fit default
    #         via Newton-Raphson/IRLS to convergence; odds_ratios = exp(beta);
    #         strongest_logit_predictor = feature (excl. intercept) with max |beta|.
    # TODO 6: Return the dict with all 13 keys (see problem.md).

    return {}  # <- replace with the completed answer dict


if __name__ == "__main__":
    print(solve())
