# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 2: Hypothesis Testing, Bootstrap & CUPED

Complete the `solve()` function. Read problem.md for the full specification.
The bootstrap is auto-graded against a bit-reproducible reference: you MUST
follow the seed / resample protocol exactly (treatment resampled before
control, every iteration).

    python grader.py starter.py
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader

# --- Fixed problem constants (do not change) ---
COHORT = ["control", "treatment_a"]
BOOT_SEED = 2024            # np.random.default_rng(BOOT_SEED)
BOOT_B = 2000              # number of bootstrap resamples
MT_P_VALUES = [0.03, 0.012, 0.04, 0.65, 0.009]   # five simultaneous tests
MT_ALPHA = 0.05


def solve() -> dict:
    """Return the hypothesis-testing / bootstrap / CUPED answer dict.

    See problem.md for the exact 13 keys and how each is defined.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "experiment_data.parquet")
    co = df.filter(pl.col("experiment_group").is_in(COHORT))

    # TODO 1: Extract metric_value for treatment_a (`t`) and control (`c`) as
    #         float numpy arrays, in file order. Welch two-sample t-test:
    #         welch_t, welch_p = stats.ttest_ind(t, c, equal_var=False).
    #         mean_diff = t.mean() - c.mean().
    # TODO 2: Seeded percentile bootstrap of mean_diff (EXACT protocol):
    #             rng = np.random.default_rng(BOOT_SEED)
    #             for b in range(BOOT_B):
    #                 bt = rng.choice(t, size=t.size, replace=True)   # treatment FIRST
    #                 bc = rng.choice(c, size=c.size, replace=True)   # control SECOND
    #                 diffs[b] = bt.mean() - bc.mean()
    #             boot_ci_low, boot_ci_high = np.percentile(diffs, [2.5, 97.5])
    # TODO 3: CUPED with pre_metric_value as covariate, over the full cohort:
    #             theta = np.cov(metric, pre, ddof=1)[0,1] / np.var(pre, ddof=1)
    #             metric_adj = metric - theta * (pre - pre.mean())
    #             var_metric = np.var(metric, ddof=1); var_adj = np.var(metric_adj, ddof=1)
    #             cuped_var_reduction = 1 - var_adj / var_metric
    # TODO 4: Re-run the Welch test on metric_adj (treatment vs control) ->
    #         welch_t_cuped, welch_p_cuped.
    # TODO 5: Multiple testing over MT_P_VALUES at MT_ALPHA:
    #           - Bonferroni: count pv < alpha/m  -> bonferroni_n_sig
    #           - Benjamini-Hochberg step-up: sort p, threshold_i = alpha*i/m,
    #             reject all up to the largest i with p_(i) <= threshold_i
    #             -> bh_n_sig
    # TODO 6: Return the dict with all 13 keys (see problem.md).

    return {}  # <- replace with the completed answer dict


if __name__ == "__main__":
    print(solve())
