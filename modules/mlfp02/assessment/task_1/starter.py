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

    # TODO 1: Restrict to the control + treatment_a cohort and add a boolean
    #         `converted` column = (metric_value >= CONVERT_THRESHOLD).
    # TODO 2: Compute p_convert_overall, p_convert_control, p_convert_treatment.
    # TODO 3: Bayes inversion p_treatment_given_convert =
    #         P(converted|treatment) * P(treatment) / P(converted), where
    #         P(treatment) is the treatment_a share of the cohort.
    # TODO 4: SRM check vs a designed 50/50 split — chi-square goodness-of-fit
    #         on [n_control, n_treatment] with expected = n_total/2 each
    #         (df=1). srm_p_value = stats.chi2.sf(chi2, df=1);
    #         srm_flag = (srm_p_value < 1e-3).
    # TODO 5: Fraud base-rate Bayes (use the FRAUD_* constants):
    #         P(fraud|flagged) = sens*base / (sens*base + fpr*(1-base)).
    # TODO 6: Beta-Binomial update on treatment_a: successes = sum(converted),
    #         failures = n - successes; posterior = Beta(prior_a+successes,
    #         prior_b+failures); posterior_mean = a/(a+b);
    #         95% credible interval via stats.beta.ppf(0.025/0.975, a, b).
    # TODO 7: Return the dict with all 13 keys (see problem.md).

    return {}  # <- replace with the completed answer dict


if __name__ == "__main__":
    print(solve())
