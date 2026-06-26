# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 1: Probability, Bayes & Experiment Validation

Complete the `solve()` function. Read problem.md for the full specification.
Your submission is auto-graded: every probability, the SRM chi-square, the
base-rate Bayes scalar, and the Beta posterior must match the independently
re-derived reference within tight tolerances.

    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl
from scipy import stats

from shared import MLFPDataLoader

# --- Fixed problem constants (do not change) ---
COHORT = ["control", "treatment_a"]
CONVERT_THRESHOLD = 50.0          # converted := metric_value >= 50.0
FRAUD_BASE_RATE = 0.02            # P(fraud)
FRAUD_SENSITIVITY = 0.95          # P(flagged | fraud)
FRAUD_FPR = 0.03                  # P(flagged | not fraud)
BETA_PRIOR_ALPHA = 2.0
BETA_PRIOR_BETA = 20.0


def solve() -> dict:
    """Return the probability / Bayes / experiment-validation answer dict.

    See problem.md for the exact 13 keys and how each is defined.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "experiment_data.parquet")

    cohort = df.filter(pl.col("experiment_group").is_in(COHORT)).with_columns(
        converted=(pl.col("metric_value") >= CONVERT_THRESHOLD)
    )

    n_total = float(cohort.height)
    control_df = cohort.filter(pl.col("experiment_group") == "control")
    treatment_df = cohort.filter(pl.col("experiment_group") == "treatment_a")

    n_control = float(control_df.height)
    n_treatment = float(treatment_df.height)

    converted_total = float(cohort.filter(pl.col("converted")).height)
    converted_control = float(control_df.filter(pl.col("converted")).height)
    converted_treatment = float(treatment_df.filter(pl.col("converted")).height)

    p_convert_overall = converted_total / n_total
    p_convert_control = converted_control / n_control
    p_convert_treatment = converted_treatment / n_treatment

    p_treatment = n_treatment / n_total
    p_treatment_given_convert = (
        p_convert_treatment * p_treatment / p_convert_overall
        if p_convert_overall > 0
        else 0.0
    )

    expected_each = n_total / 2.0
    srm_chi2 = ((n_control - expected_each) ** 2) / expected_each + (
        (n_treatment - expected_each) ** 2
    ) / expected_each
    srm_p_value = float(stats.chi2.sf(srm_chi2, df=1))
    srm_flag = srm_p_value < 1e-3

    p_fraud_given_flagged = (
        FRAUD_SENSITIVITY * FRAUD_BASE_RATE
        / (
            FRAUD_SENSITIVITY * FRAUD_BASE_RATE
            + FRAUD_FPR * (1.0 - FRAUD_BASE_RATE)
        )
    )

    successes = converted_treatment
    failures = n_treatment - successes
    beta_post_alpha = BETA_PRIOR_ALPHA + successes
    beta_post_beta = BETA_PRIOR_BETA + failures
    posterior_mean = beta_post_alpha / (beta_post_alpha + beta_post_beta)
    cred_int_low = float(stats.beta.ppf(0.025, beta_post_alpha, beta_post_beta))
    cred_int_high = float(stats.beta.ppf(0.975, beta_post_alpha, beta_post_beta))

    return {
        "p_convert_overall": p_convert_overall,
        "p_convert_control": p_convert_control,
        "p_convert_treatment": p_convert_treatment,
        "p_treatment_given_convert": p_treatment_given_convert,
        "srm_chi2": srm_chi2,
        "srm_p_value": srm_p_value,
        "srm_flag": bool(srm_flag),
        "p_fraud_given_flagged": p_fraud_given_flagged,
        "beta_post_alpha": beta_post_alpha,
        "beta_post_beta": beta_post_beta,
        "posterior_mean": posterior_mean,
        "cred_int_low": cred_int_low,
        "cred_int_high": cred_int_high,
    }


if __name__ == "__main__":
    print(solve())
